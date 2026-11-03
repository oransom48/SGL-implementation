"""SGL-ED recommender model for a user-item interaction dataset.

This is a self-contained model/training module based on the SGL paper and the
authors' SGL-Torch implementation. It implements the Edge Dropout (ED) variant
with a LightGCN encoder, BPR ranking loss, and two-view InfoNCE loss.

Data contract
-------------
``train_pairs`` is an N x 2 array-like object. Column 0 contains contiguous
user IDs in [0, n_users); column 1 contains contiguous item IDs in
[0, n_items). The data module must use the same ID-to-index mapping for train,
validation, and test data.

Typical use from main.py
------------------------
    model = SGLRecommender(n_users, n_items, ...)
    history = model.fit(train_pairs, epochs=2, batch_size=2048)
    scores = model.predict(user_ids)  # shape [len(user_ids), n_items]

The returned scores are dot-product ranking scores, not probabilities or
evaluation metrics. The evaluation module should mask items seen in training,
rank each row, and compare the ranking with held-out ground truth.

Paper-style hyperparameters are the defaults except ``epochs=2``, which is a
short smoke-test setting for resource-limited machines. Increase it for the
final experiment and clearly report the value actually used.

Compatibility: designed for the project's Python 3.7 / PyTorch 1.9 environment.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SGLRecommender(nn.Module):
    """SGL with edge-dropout graph augmentation and LightGCN propagation.

    Args:
        n_users: Number of users in the shared ID mapping.
        n_items: Number of items in the shared ID mapping.
        embedding_dim: User/item embedding width (paper setting: 64).
        n_layers: Number of LightGCN propagation layers (paper setting: 3).
        ssl_ratio: Fraction of interaction edges dropped in each augmented
            graph (the SGL-Torch ``ssl_ratio`` setting; paper experiments tune
            this value).
        ssl_weight: Weight of the self-supervised loss.
        ssl_temp: Temperature used by the InfoNCE objective.
        reg_weight: Weight of L2 regularization on sampled ID embeddings.
        learning_rate: Adam learning rate.
        device: Optional torch device. If omitted, CUDA is used only when
            ``torch.cuda.is_available()`` is true; otherwise CPU is used.
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        n_layers=3,
        ssl_ratio=0.1,
        ssl_weight=0.1,
        ssl_temp=0.2,
        reg_weight=1e-4,
        learning_rate=1e-3,
        device=None,
    ):
        super(SGLRecommender, self).__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be positive")
        if n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        if not 0.0 <= ssl_ratio < 1.0:
            raise ValueError("ssl_ratio must be in [0, 1)")
        if ssl_temp <= 0:
            raise ValueError("ssl_temp must be positive")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_nodes = self.n_users + self.n_items
        self.embedding_dim = int(embedding_dim)
        self.n_layers = int(n_layers)
        self.ssl_ratio = float(ssl_ratio)
        self.ssl_weight = float(ssl_weight)
        self.ssl_temp = float(ssl_temp)
        self.reg_weight = float(reg_weight)
        self.learning_rate = float(learning_rate)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.to(self.device)

        self._train_users = None
        self._train_items = None
        self._positive_items_by_user = None
        self._full_adj = None
        self._cached_user_repr = None
        self._cached_item_repr = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def _as_pairs(train_pairs):
        """Convert an N x 2 array-like interaction table to int64 NumPy."""
        if hasattr(train_pairs, "to_numpy"):
            train_pairs = train_pairs.to_numpy()
        pairs = np.asarray(train_pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("train_pairs must have shape [n_interactions, 2]")
        if pairs.shape[0] == 0:
            raise ValueError("train_pairs cannot be empty")
        if not np.issubdtype(pairs.dtype, np.integer):
            if not np.all(np.equal(pairs, np.floor(pairs))):
                raise ValueError("user and item IDs must be integers")
        return pairs.astype(np.int64, copy=False)

    def _normalize_graph(self, keep_edges=None):
        """Create D^-1/2 A D^-1/2 for the undirected user-item graph."""
        users = self._train_users
        items = self._train_items + self.n_users
        if keep_edges is not None:
            users = users[keep_edges]
            items = items[keep_edges]

        # Each interaction is an undirected edge, represented in both directions.
        row = torch.cat((users, items))
        col = torch.cat((items, users))
        indices = torch.stack((row, col), dim=0)
        values = torch.ones(row.numel(), dtype=self.user_embedding.weight.dtype,
                            device=self.device)
        degree = torch.zeros(self.n_nodes, dtype=values.dtype, device=self.device)
        degree.index_add_(0, row, values)
        inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
        values = values * inv_sqrt_degree[row] * inv_sqrt_degree[col]
        return torch.sparse_coo_tensor(
            indices, values, (self.n_nodes, self.n_nodes), device=self.device
        ).coalesce()

    def _make_augmented_graph(self):
        """Independently drop edges and normalize one SGL-ED graph view."""
        n_edges = self._train_users.numel()
        n_keep = int(n_edges * (1.0 - self.ssl_ratio))
        keep = torch.zeros(n_edges, dtype=torch.bool, device=self.device)
        if n_keep:
            keep_ids = torch.randperm(n_edges, device=self.device)[:n_keep]
            keep[keep_ids] = True
        return self._normalize_graph(keep)

    def _encode(self, adjacency):
        """LightGCN propagation; average embeddings from layer 0 through L."""
        ego = torch.cat((self.user_embedding.weight,
                         self.item_embedding.weight), dim=0)
        layer_embeddings = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adjacency, ego)
            layer_embeddings.append(ego)
        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        return (final_embeddings[:self.n_users],
                final_embeddings[self.n_users:])

    def _sample_negative_items(self, batch_users, rng):
        """Sample one item absent from each user's training interactions."""
        negatives = np.empty(batch_users.shape[0], dtype=np.int64)
        for row, user in enumerate(batch_users):
            candidate = int(rng.randint(self.n_items))
            positives = self._positive_items_by_user[int(user)]
            if len(positives) >= self.n_items:
                raise ValueError(
                    "Cannot sample a negative item: user {} interacted with every item".format(user)
                )
            while candidate in positives:
                candidate = int(rng.randint(self.n_items))
            negatives[row] = candidate
        return negatives

    def _batch_loss(self, users, positive_items, negative_items, view_a, view_b):
        """Compute BPR + paper-style two-view InfoNCE + embedding L2 loss."""
        user_repr, item_repr = self._encode(self._full_adj)
        batch_user_repr = user_repr[users]
        positive_repr = item_repr[positive_items]
        negative_repr = item_repr[negative_items]

        positive_scores = (batch_user_repr * positive_repr).sum(dim=1)
        negative_scores = (batch_user_repr * negative_repr).sum(dim=1)
        bpr_loss = -F.logsigmoid(positive_scores - negative_scores).sum()

        view_a_users, view_a_items = self._encode(view_a)
        view_b_users, view_b_items = self._encode(view_b)

        # SGL normalizes each augmented-view representation before computing
        # the contrastive similarities (the supervised full-graph embeddings
        # used by BPR are not normalized here).
        view_a_users = F.normalize(view_a_users, p=2, dim=1)
        view_a_items = F.normalize(view_a_items, p=2, dim=1)
        view_b_users = F.normalize(view_b_users, p=2, dim=1)
        view_b_items = F.normalize(view_b_items, p=2, dim=1)

        # Match the author's implementation: batch anchors contrast against
        # every user/item embedding in the opposite graph view.
        positive_user_sim = (view_a_users[users] * view_b_users[users]).sum(dim=1, keepdim=True)
        user_logits = torch.matmul(view_a_users[users], view_b_users.t())
        user_ssl = torch.logsumexp(
            (user_logits - positive_user_sim) / self.ssl_temp, dim=1
        ).sum()

        positive_item_sim = (view_a_items[positive_items] * view_b_items[positive_items]).sum(dim=1, keepdim=True)
        item_logits = torch.matmul(view_a_items[positive_items], view_b_items.t())
        item_ssl = torch.logsumexp(
            (item_logits - positive_item_sim) / self.ssl_temp, dim=1
        ).sum()
        ssl_loss = user_ssl + item_ssl

        # Regularize the sampled ID embeddings before graph propagation.
        reg_loss = 0.5 * (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(positive_items).pow(2).sum()
            + self.item_embedding(negative_items).pow(2).sum()
        )
        total_loss = bpr_loss + self.ssl_weight * ssl_loss + self.reg_weight * reg_loss
        return total_loss, bpr_loss.detach(), ssl_loss.detach(), reg_loss.detach()

    def fit(self, train_pairs, epochs=2, batch_size=2048, seed=2021, verbose=1):
        """Train on positive user-item pairs and return per-epoch loss history.

        Negative items are sampled uniformly and rejected if the user interacted
        with them in the training data. Two edge-dropped graph views are sampled
        once at the beginning of each epoch, as in SGL-ED.

        ``epochs=2`` is intended for a smoke test. The author's configuration
        used a much larger maximum training budget; use the paper/repo setting
        for a final run when practical, and record any reduced setting.
        """
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        pairs = self._as_pairs(train_pairs)
        if (pairs[:, 0].min() < 0 or pairs[:, 0].max() >= self.n_users
                or pairs[:, 1].min() < 0 or pairs[:, 1].max() >= self.n_items):
            raise ValueError("train_pairs has IDs outside the configured ranges")

        # Treat repeated user-item rows as one implicit-feedback edge.
        pairs = np.unique(pairs, axis=0)
        self._train_users = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=self.device)
        self._train_items = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=self.device)
        self._positive_items_by_user = [set() for _ in range(self.n_users)]
        for user, item in pairs:
            self._positive_items_by_user[int(user)].add(int(item))
        self._full_adj = self._normalize_graph()
        self._cached_user_repr = None
        self._cached_item_repr = None

        rng = np.random.RandomState(seed)
        history = []
        n_interactions = pairs.shape[0]

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            view_a = self._make_augmented_graph()
            view_b = self._make_augmented_graph()
            permutation = rng.permutation(n_interactions)
            sums = np.zeros(4, dtype=np.float64)
            batches = 0
            self.train()

            for start in range(0, n_interactions, batch_size):
                batch_rows = permutation[start:start + batch_size]
                batch_users_np = pairs[batch_rows, 0]
                batch_pos_np = pairs[batch_rows, 1]
                batch_neg_np = self._sample_negative_items(batch_users_np, rng)
                users = torch.as_tensor(batch_users_np, dtype=torch.long, device=self.device)
                positive_items = torch.as_tensor(batch_pos_np, dtype=torch.long, device=self.device)
                negative_items = torch.as_tensor(batch_neg_np, dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()
                total, bpr, ssl, reg = self._batch_loss(
                    users, positive_items, negative_items, view_a, view_b
                )
                total.backward()
                self.optimizer.step()
                self._cached_user_repr = None
                self._cached_item_repr = None
                sums += np.asarray([total.item(), bpr.item(), ssl.item(), reg.item()])
                batches += 1

            row = {
                "epoch": epoch,
                "loss": float(sums[0] / n_interactions),
                "bpr_loss": float(sums[1] / n_interactions),
                "ssl_loss_weighted": float(self.ssl_weight * sums[2] / n_interactions),
                "reg_loss_weighted": float(self.reg_weight * sums[3] / n_interactions),
                "seconds": float(time.time() - start_time),
            }
            history.append(row)
            if verbose:
                print(
                    "[epoch {epoch}: loss {loss:.4f} = {bpr_loss:.4f} + "
                    "{ssl_loss_weighted:.4f} + {reg_loss_weighted:.4f}, "
                    "time {seconds:.1f}s]".format(**row)
                )
        return history

    def _get_final_embeddings(self):
        if self._full_adj is None:
            raise RuntimeError("Call fit(train_pairs, ...) before prediction")
        if self._cached_user_repr is None or self._cached_item_repr is None:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                self._cached_user_repr, self._cached_item_repr = self._encode(self._full_adj)
            if was_training:
                self.train()
        return self._cached_user_repr, self._cached_item_repr

    def predict(self, user_ids):
        """Return scores shaped [number of users, n_items], in item-ID order.

        Column ``i`` always corresponds to ``item_id == i`` from the data
        module's shared mapping. Scores are dot products for ranking only.
        """
        if hasattr(user_ids, "detach"):
            user_ids = user_ids.detach().cpu().numpy()
        users_np = np.asarray(user_ids, dtype=np.int64).reshape(-1)
        if users_np.size and (users_np.min() < 0 or users_np.max() >= self.n_users):
            raise ValueError("user_ids contains an ID outside [0, n_users)")
        users = torch.as_tensor(users_np, dtype=torch.long, device=self.device)
        user_repr, item_repr = self._get_final_embeddings()
        with torch.no_grad():
            return torch.matmul(user_repr[users], item_repr.t()).cpu().numpy()


class _SGLGraphEncoder(nn.Module):
    """Encoder API used by the team's eval branch main.py.

    Graphs and triplets are supplied by sgl.preprocess, not rebuilt here.
    norm_adj is a buffer so model.to(device) also moves the graph.
    """

    def __init__(self, num_users, num_items, norm_adj, embed_dim, n_layers, param_init):
        super().__init__()
        if tuple(norm_adj.shape) != (num_users + num_items, num_users + num_items):
            raise ValueError("norm_adj shape must match num_users + num_items")
        self.num_users, self.num_items = num_users, num_items
        self.n_layers = n_layers
        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self._prediction_cache = None
        self.reset_parameters(init_method=param_init)

    def reset_parameters(self, pretrain=0, init_method="xavier_uniform", pretrain_dir=None):
        from pathlib import Path
        for name, embedding in (("user", self.user_embeddings), ("item", self.item_embeddings)):
            if pretrain:
                if pretrain_dir is None:
                    raise ValueError("pretrain_dir is required")
                values = torch.as_tensor(np.load(str(Path(pretrain_dir) / (name + "_embeddings.npy"))),
                                         dtype=embedding.weight.dtype, device=embedding.weight.device)
                if values.shape != embedding.weight.shape:
                    raise ValueError("Pretrained embedding shape does not match the ID mapping")
                with torch.no_grad():
                    embedding.weight.copy_(values)
            elif init_method == "xavier_uniform":
                nn.init.xavier_uniform_(embedding.weight)
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(embedding.weight)
            elif init_method == "normal":
                nn.init.normal_(embedding.weight, std=0.01)
            elif init_method == "uniform":
                nn.init.uniform_(embedding.weight, -0.01, 0.01)
            else:
                raise ValueError("Unsupported param_init: " + init_method)
        self._prediction_cache = None

    def train(self, mode=True):
        self._prediction_cache = None
        return super().train(mode)

    def _forward_gcn(self, graph):
        if isinstance(graph, (list, tuple)) and len(graph) != self.n_layers:
            raise ValueError("Layer-specific graphs must have n_layers entries")
        x = torch.cat((self.user_embeddings.weight, self.item_embeddings.weight))
        layers = [x]
        for k in range(self.n_layers):
            adjacency = graph[k] if isinstance(graph, (list, tuple)) else graph
            x = torch.sparse.mm(adjacency.to(x.device), x)
            layers.append(x)
        final = torch.stack(layers, dim=1).mean(dim=1)
        return final[:self.num_users], final[self.num_users:]

    @torch.no_grad()
    def predict(self, users):
        # Parameter versions change after optimizer steps and checkpoint loads.
        # This also prevents stale scores when evaluation follows a restore.
        key = (self.user_embeddings.weight._version, self.item_embeddings.weight._version,
               self.user_embeddings.weight.device, self.user_embeddings.weight.dtype)
        if self._prediction_cache is None or self._prediction_cache[0] != key:
            self._prediction_cache = (key, self._forward_gcn(self.norm_adj))
        user_repr, item_repr = self._prediction_cache[1]
        return user_repr[users] @ item_repr.t()


class SGLModel(nn.Module):
    """Drop-in model API for oransom48/SGL-implementation, branch eval.

    Reference interface inspected: sgl/main.py and sgl/evaluate.py, 2026-11-02.
    main.py owns sampling, optimizer, evaluation and stopping. calculate_loss
    returns (total, BPR, weighted_SSL, weighted_L2), all scalar tensors.
    predict returns NumPy scores [batch_users, num_items] in shared item-ID order.
    ED/ND take one adjacency per view; RW takes n_layers adjacencies per view.
    """

    def __init__(self, num_users, num_items, norm_adj, embed_dim=64, n_layers=3,
                 ssl_temp=0.2, ssl_reg=0.1, reg=1e-4, ssl_mode="both_side",
                 param_init="xavier_uniform"):
        super().__init__()
        if num_users <= 0 or num_items <= 0 or embed_dim <= 0 or n_layers < 0:
            raise ValueError("Invalid model dimensions")
        if ssl_temp <= 0 or ssl_reg < 0 or reg < 0:
            raise ValueError("Invalid loss hyperparameters")
        if ssl_mode not in ("both_side", "user_side", "item_side"):
            raise ValueError("Unsupported ssl_mode")
        self.num_users, self.num_items = num_users, num_items
        self.ssl_temp, self.ssl_reg, self.reg = ssl_temp, ssl_reg, reg
        self.ssl_mode = ssl_mode
        self.lightgcn = _SGLGraphEncoder(num_users, num_items, norm_adj,
                                         embed_dim, n_layers, param_init)

    def calculate_loss(self, sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items):
        encoder = self.lightgcn
        device = encoder.user_embeddings.weight.device
        users, positives, negatives = [torch.as_tensor(x, dtype=torch.long, device=device)
                                      for x in (bat_users, bat_pos_items, bat_neg_items)]
        encoder._prediction_cache = None
        u, i = encoder._forward_gcn(encoder.norm_adj)
        margin = (u[users] * (i[positives] - i[negatives])).sum(dim=1)
        bpr = -F.logsigmoid(margin).sum()
        raw_reg = 0.5 * sum(x.pow(2).sum() for x in (
            encoder.user_embeddings(users), encoder.item_embeddings(positives),
            encoder.item_embeddings(negatives)))
        a_u, a_i = encoder._forward_gcn(sub_graph1)
        b_u, b_i = encoder._forward_gcn(sub_graph2)

        def contrastive(a, b, ids):
            a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
            anchors = a[ids]
            positive = (anchors * b[ids]).sum(dim=1, keepdim=True)
            return torch.logsumexp((anchors @ b.t() - positive) / self.ssl_temp, dim=1).sum()

        ssl = bpr.new_zeros(())
        if self.ssl_mode in ("user_side", "both_side"):
            ssl = ssl + contrastive(a_u, b_u, users)
        if self.ssl_mode in ("item_side", "both_side"):
            ssl = ssl + contrastive(a_i, b_i, positives)
        weighted_ssl, weighted_reg = self.ssl_reg * ssl, self.reg * raw_reg
        return bpr + weighted_ssl + weighted_reg, bpr, weighted_ssl, weighted_reg

    @torch.no_grad()
    def predict(self, users):
        users = torch.as_tensor(users, dtype=torch.long,
                                device=self.lightgcn.user_embeddings.weight.device)
        if users.ndim != 1:
            raise ValueError("users must be a one-dimensional sequence")
        return self.lightgcn.predict(users).cpu().numpy()


__all__ = ["SGLRecommender", "SGLModel"]
