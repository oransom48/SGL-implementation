"""
preprocess.py
=============
Data loading, preprocessing, negative sampling, and graph adjacency matrix construction
for Self-Supervised Graph Learning (SGL) for Recommendation (Wu et al., SIGIR 2021).
Reference implementation: https://github.com/wujcan/SGL-Torch

This module implements:
1. Loading interaction files (`.train` / `.test` CSV format `user_id,item_id` used in
   `SGL-Torch/dataset/` or space-separated adjacency list format `train.txt` / `test.txt`).
   If no dataset files exist in the target folder, a synthetic dataset with a power-law
   item degree distribution is automatically generated so the pipeline runs out-of-the-box.
2. Constructing the bipartite user-item interaction graph G = (V, E), where V = U union I
   (|V| = |U| + |I|) and E = O^+ represents observed user-item interactions.
3. Implementing `create_adj_mat` exactly as in `SGL-Torch/model/general_recommender/SGL.py`:
   - Full graph (`is_subgraph=False`): builds intact symmetric adjacency matrix A and
     normalizes it via symmetric Laplacian normalization:
         A_hat = D^{-1/2} A D^{-1/2}
   - Node Dropout (`aug_type='nd'`, Eq. 7): randomly drops `ssl_ratio` fraction of users
     and items using diagonal indicator matrices:
         R' = diag(indicator_user) * R * diag(indicator_item)
   - Edge Dropout (`aug_type='ed'`, Eq. 8) & Random Walk (`aug_type='rw'`, Eq. 9):
     randomly retains `(1 - ssl_ratio)` fraction of edges from the observed interactions.
4. Implementing `PairwiseSampler` matching `PairwiseSamplerV2` in `SGL-Torch/data/sampler.py`
   for sampling BPR training triplets O = {(u, i, j) | (u, i) in O^+, (u, j) in O^-}.
"""

import os
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch


def generate_synthetic_dataset(
    data_dir: str,
    dataset_name: str = "synthetic",
    num_users: int = 300,
    num_items: int = 500,
    interactions_per_user: int = 25,
    seed: int = 2021,
) -> None:
    """
    Generates a synthetic implicit-feedback dataset with a power-law item popularity
    distribution (reflecting the skewed long-tail distribution described in Section 1)
    and saves `<dataset_name>.train` and `<dataset_name>.test` in `SGL-Torch` CSV format
    (`user_id,item_id`).
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Power-law item popularity distribution: few head items, many long-tail items
    item_ranks = np.arange(1, num_items + 1, dtype=np.float64)
    item_probs = item_ranks ** (-0.8)
    item_probs /= item_probs.sum()

    train_pairs = []
    test_pairs = []

    for u in range(num_users):
        # Sample unique interacted items for user u
        n_inter = max(5, int(rng.poisson(interactions_per_user)))
        n_inter = min(n_inter, num_items // 2)
        items = rng.choice(num_items, size=n_inter, replace=False, p=item_probs)

        # Split 80% train, 20% test
        n_test = max(1, int(len(items) * 0.2))
        train_items = np.sort(items[:-n_test])
        test_items = np.sort(items[-n_test:])

        for i in train_items:
            train_pairs.append(f"{u},{int(i)}")
        for i in test_items:
            test_pairs.append(f"{u},{int(i)}")

    train_file = os.path.join(data_dir, f"{dataset_name}.train")
    test_file = os.path.join(data_dir, f"{dataset_name}.test")

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_pairs) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_pairs) + "\n")


def sp_mat_to_sp_tensor(sp_mat: sp.spmatrix) -> torch.Tensor:
    """
    Converts a SciPy sparse matrix into a coalesced PyTorch sparse COO tensor.
    Directly mirrors `sp_mat_to_sp_tensor` in `SGL-Torch/util/pytorch/func.py`.
    """
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col], dtype=np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


class SGLDataset:
    """
    Dataset loader and graph builder for SGL, combining the functionality of
    `data/dataset.py` and `SGL.create_adj_mat` from `SGL-Torch`.
    """

    def __init__(
        self,
        data_dir: str = "./dataset/synthetic",
        dataset_name: str = "synthetic",
        ssl_ratio: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            data_dir: Directory containing `<dataset_name>.train`/`.test` or `train.txt`/`test.txt`.
            dataset_name: Name of the dataset (e.g., 'yelp2018', 'amazon-book', 'ifashion', 'synthetic').
            ssl_ratio: Dropout probability rho for graph structure augmentations (ND, ED, RW).
            device: PyTorch device ('cpu' or 'cuda') where sparse graph tensors are placed.
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.ssl_ratio = ssl_ratio
        self.device = device

        # Locate train and test files (supports SGL-Torch `.train`/`.test` or `.txt` files)
        train_path, test_path = self._resolve_data_paths()

        # User -> list/set of interacted item IDs
        self.train_user_dict = defaultdict(list)
        self.test_user_dict = defaultdict(list)
        self.train_user_set = defaultdict(set)

        # Arrays of observed training interactions (u, i) in E = O^+
        self.train_users = []
        self.train_items = []

        self.num_users = 0
        self.num_items = 0

        # Parse training and testing interactions
        self._load_file(train_path, is_train=True)
        self._load_file(test_path, is_train=False)

        # Convert lists to NumPy arrays for fast indexing during graph augmentation
        self.train_users = np.asarray(self.train_users, dtype=np.int32)
        self.train_items = np.asarray(self.train_items, dtype=np.int32)
        self.num_ratings = len(self.train_users)

        # Convert user_dict values to numpy arrays (matching Interaction.to_user_dict in SGL-Torch)
        self.train_user_dict = {u: np.asarray(items, dtype=np.int32) for u, items in self.train_user_dict.items()}
        self.test_user_dict = {u: np.asarray(items, dtype=np.int32) for u, items in self.test_user_dict.items()}

        # Build the intact normalized adjacency matrix D^{-1/2} A D^{-1/2} for the main task
        adj_matrix = self.create_adj_mat(is_subgraph=False)
        self.norm_adj = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

    def _resolve_data_paths(self) -> tuple[str, str]:
        """Checks for existing `.train`/`.test` or `train.txt`/`test.txt`, or generates synthetic data."""
        sgl_train = os.path.join(self.data_dir, f"{self.dataset_name}.train")
        sgl_test = os.path.join(self.data_dir, f"{self.dataset_name}.test")
        txt_train = os.path.join(self.data_dir, "train.txt")
        txt_test = os.path.join(self.data_dir, "test.txt")

        if os.path.exists(sgl_train) and os.path.exists(sgl_test):
            return sgl_train, sgl_test
        if os.path.exists(txt_train) and os.path.exists(txt_test):
            return txt_train, txt_test

        print(f"[Preprocess] No dataset files found in '{self.data_dir}'. Generating synthetic dataset...")
        generate_synthetic_dataset(self.data_dir, self.dataset_name)
        return sgl_train, sgl_test

    def _load_file(self, file_path: str, is_train: bool) -> None:
        """
        Reads interaction data supporting both:
        1. SGL-Torch CSV format (`user_id,item_id` per line)
        2. Adjacency list / whitespace-separated format (`user_id item_1 item_2 ...`)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support comma-separated (SGL-Torch default) or whitespace-separated
                parts = line.split(",") if "," in line else line.split()
                if len(parts) < 2:
                    continue

                u = int(parts[0])
                items = [int(x) for x in parts[1:]]

                self.num_users = max(self.num_users, u + 1)
                self.num_items = max(self.num_items, max(items) + 1)

                for i in items:
                    if is_train:
                        if i not in self.train_user_set[u]:
                            self.train_user_dict[u].append(i)
                            self.train_user_set[u].add(i)
                            self.train_users.append(u)
                            self.train_items.append(i)
                    else:
                        self.test_user_dict[u].append(i)

    def create_adj_mat(self, is_subgraph: bool = False, aug_type: str = "ed") -> sp.spmatrix:
        """
        Builds and normalizes the bipartite user-item adjacency matrix.
        Directly mirrors `SGL.create_adj_mat` in `SGL-Torch/model/general_recommender/SGL.py`.

        Let n_nodes = |U| + |I|. User nodes are indexed 0..|U|-1 and item nodes are shifted
        by +|U| to |U|..|U|+|I|-1.

        Args:
            is_subgraph: If True, applies graph augmentation s(G) ~ S with drop probability `ssl_ratio`.
            aug_type: One of:
                - `'nd'` (Node Dropout, Eq. 7): Drops `ssl_ratio` fraction of user and item nodes
                  by zeroing out their entries in diagonal indicator matrices:
                      R' = diag(indicator_user) * R * diag(indicator_item)
                - `'ed'` (Edge Dropout, Eq. 8) or `'rw'` (Random Walk, Eq. 9):
                  Randomly retains `(1 - ssl_ratio)` fraction of user-item edges.

        Returns:
            Symmetrically normalized SciPy sparse matrix `D^{-1/2} A D^{-1/2}`.
        """
        n_nodes = self.num_users + self.num_items
        users_np, items_np = self.train_users, self.train_items
        aug_type = aug_type.lower()

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == "nd":
                # ------------------------------------------------------------------
                # Node Dropout (ND, Eq. 7): s(G) = (M' o V, E)
                # Drop `ssl_ratio` fraction of user nodes and item nodes
                # ------------------------------------------------------------------
                drop_user_size = int(self.num_users * self.ssl_ratio)
                drop_item_size = int(self.num_items * self.ssl_ratio)
                drop_user_idx = np.random.choice(self.num_users, size=drop_user_size, replace=False)
                drop_item_idx = np.random.choice(self.num_items, size=drop_item_size, replace=False)

                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.0
                indicator_item[drop_item_idx] = 0.0

                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)

                # User-item interaction matrix R
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                    shape=(self.num_users, self.num_items),
                )
                # Mask out edges incident to dropped users or dropped items
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                user_np_keep, item_np_keep = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix(
                    (ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                    shape=(n_nodes, n_nodes),
                )
            elif aug_type in ["ed", "rw"]:
                # ------------------------------------------------------------------
                # Edge Dropout (ED, Eq. 8) & Random Walk per-layer (RW, Eq. 9):
                # s(G) = (V, M o E) where (1 - ssl_ratio) of edges are kept
                # ------------------------------------------------------------------
                keep_size = int(len(users_np) * (1.0 - self.ssl_ratio))
                keep_idx = np.random.choice(len(users_np), size=keep_size, replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix(
                    (ratings, (user_np, item_np + self.num_users)),
                    shape=(n_nodes, n_nodes),
                )
            else:
                raise ValueError(f"Unsupported aug_type '{aug_type}'. Expected 'nd', 'ed', or 'rw'.")
        else:
            # Intact bipartite interaction graph G = (V, E)
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix(
                (ratings, (users_np, items_np + self.num_users)),
                shape=(n_nodes, n_nodes),
            )

        # Symmetrize the bipartite adjacency matrix: A = tmp_adj + tmp_adj^T
        adj_mat = tmp_adj + tmp_adj.T

        # Symmetric Laplacian normalization: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj_mat.sum(1))
        with np.errstate(divide="ignore"):
            d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def sample_subgraphs(
        self, aug_type: str = "ed", n_layers: int = 3
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        """
        Generates the two augmented graph views `sub_graph1` and `sub_graph2` per training epoch,
        matching `SGL.train_model` in `SGL-Torch/model/general_recommender/SGL.py`:
        - If `aug_type` is `'nd'` or `'ed'`: returns two sparse adjacency tensors shared across layers.
        - If `aug_type` is `'rw'`: returns two lists of `n_layers` sparse adjacency tensors
          so each GCN layer k = 0..L-1 propagates over a distinct random subgraph.
        """
        aug_type = aug_type.lower()
        if aug_type in ["nd", "ed"]:
            sub_graph1 = sp_mat_to_sp_tensor(
                self.create_adj_mat(is_subgraph=True, aug_type=aug_type)
            ).to(self.device)
            sub_graph2 = sp_mat_to_sp_tensor(
                self.create_adj_mat(is_subgraph=True, aug_type=aug_type)
            ).to(self.device)
            return sub_graph1, sub_graph2
        elif aug_type == "rw":
            sub_graph1, sub_graph2 = [], []
            for _ in range(n_layers):
                tmp1 = self.create_adj_mat(is_subgraph=True, aug_type="rw")
                sub_graph1.append(sp_mat_to_sp_tensor(tmp1).to(self.device))
                tmp2 = self.create_adj_mat(is_subgraph=True, aug_type="rw")
                sub_graph2.append(sp_mat_to_sp_tensor(tmp2).to(self.device))
            return sub_graph1, sub_graph2
        else:
            raise ValueError(f"Invalid aug_type '{aug_type}'. Choose from ['nd', 'ed', 'rw'].")


class PairwiseSampler:
    """
    Pairwise BPR negative sampler matching `PairwiseSamplerV2` in `SGL-Torch/data/sampler.py`.

    In each epoch, uniformly samples `num_ratings` training triples `(u, i, j)` where:
    - `u` is a user with observed interactions,
    - `i in N_u` is a positive item observed for user `u`,
    - `j not in N_u` is a negative item unobserved for user `u`.
    Yields mini-batches `(bat_users, bat_pos_items, bat_neg_items)` of size `batch_size`.
    """

    def __init__(self, dataset: SGLDataset, batch_size: int = 2048, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.user_pos_dict = dataset.train_user_dict
        self.user_pos_set = dataset.train_user_set
        self.num_trainings = dataset.num_ratings
        self.num_items = dataset.num_items
        self.users = list(self.user_pos_dict.keys())

    def __iter__(self):
        # Sample `num_trainings` users uniformly with replacement (as in PairwiseSamplerV2)
        sampled_users = np.random.choice(self.users, size=self.num_trainings, replace=True)
        unique_users, user_counts = np.unique(sampled_users, return_counts=True)

        all_users = []
        all_pos_items = []
        all_neg_items = []

        for u, count in zip(unique_users, user_counts):
            pos_items_u = self.user_pos_dict[u]
            pos_set_u = self.user_pos_set[u]

            # Sample `count` positive items for user u (with replacement)
            pos_sampled = np.random.choice(pos_items_u, size=count, replace=True)

            # Sample `count` negative items for user u not in `pos_set_u`
            neg_sampled = np.empty(count, dtype=np.int32)
            for idx in range(count):
                j = np.random.randint(0, self.num_items)
                while j in pos_set_u:
                    j = np.random.randint(0, self.num_items)
                neg_sampled[idx] = j

            all_users.append(np.full(count, u, dtype=np.int32))
            all_pos_items.append(pos_sampled)
            all_neg_items.append(neg_sampled)

        users_arr = np.concatenate(all_users)
        pos_arr = np.concatenate(all_pos_items)
        neg_arr = np.concatenate(all_neg_items)

        # Shuffle the generated triplets across the epoch
        if self.shuffle:
            perm = np.random.permutation(self.num_trainings)
            users_arr = users_arr[perm]
            pos_arr = pos_arr[perm]
            neg_arr = neg_arr[perm]

        # Yield mini-batches
        for start in range(0, self.num_trainings, self.batch_size):
            end = min(start + self.batch_size, self.num_trainings)
            yield users_arr[start:end], pos_arr[start:end], neg_arr[start:end]
