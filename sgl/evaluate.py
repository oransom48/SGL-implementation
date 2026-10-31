"""
evaluate.py
===========
Top-K All-Ranking Evaluation Protocol for Self-Supervised Graph Learning (SGL)
(Wu et al., SIGIR 2021, Section 4.1).
Reference implementation: https://github.com/wujcan/SGL-Torch
(`reckit/evaluator.py` and `reckit/cython/include/metric.h`)

This module implements:
1. Full-Ranking Evaluation Protocol (Section 4.1):
   - Evaluates test users in mini-batches (`test_batch_size = 128`).
   - Computes preference scores y_hat_{ui} = z_u^T z_i across ALL items i in I.
   - Masks out each user's observed training items (`train_items`) by setting their
     scores to `-np.inf` so already-interacted items are never recommended.
   - Extracts the top-K highest-scoring items per user via `np.argpartition` / sorting.
2. Ranking Metrics (`Precision@K`, `Recall@K`, `NDCG@K` matching `metric.h`):
   - Precision@K(u) = |L_rec^u cap L_test^u| / K
   - Recall@K(u)    = |L_rec^u cap L_test^u| / |L_test^u|
   - NDCG@K(u)      = DCG@K(u) / iDCG@K(u), where:
         DCG@K(u)  = sum_{r=0}^{K-1} I(rank[r] in L_test^u) / log2(r + 2)
         iDCG@K(u) = sum_{r=0}^{min(K, |L_test^u|) - 1} 1.0 / log2(r + 2)
3. Long-Tail Item Group Recall Decomposition (Section 4.3.1, Figure 4):
   - Splits all items into 10 popularity groups G_1, ..., G_10 keeping the total number
     of training interactions in each group approximately equal (higher GroupID = higher degree).
   - Decomposes overall Recall@K into each group's contribution:
         Recall = sum_{g=1}^{10} Recall^{(g)}
"""

import numpy as np


def compute_user_metrics(
    ranked_items: np.ndarray,
    ground_truth: set[int],
    top_k_list: list[int],
) -> dict[str, np.ndarray]:
    """
    Computes Precision@K, Recall@K, and NDCG@K for a single user u given their
    top-max(K) ranked item IDs `ranked_items` and ground-truth test items `ground_truth`.

    Exact port of `precision`, `recall`, and `ndcg` in `SGL-Torch/reckit/cython/include/metric.h`.
    """
    max_k = len(ranked_items)
    truth_len = len(ground_truth)

    # Binary hit indicator at each rank position r = 0, 1, ..., max_k - 1
    hits = np.array([1.0 if item in ground_truth else 0.0 for item in ranked_items], dtype=np.float64)
    cum_hits = np.cumsum(hits)

    # Ranks 1, 2, ..., max_k
    ranks = np.arange(1, max_k + 1, dtype=np.float64)

    # 1. Precision@k = cum_hits[k-1] / k
    precisions = cum_hits / ranks

    # 2. Recall@k = cum_hits[k-1] / |L_test^u|
    recalls = cum_hits / truth_len if truth_len > 0 else np.zeros_like(cum_hits)

    # 3. NDCG@k = DCG@k / iDCG@k
    #    Discount factor at 0-based index r is 1 / log2(r + 2)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
    dcg = np.cumsum(hits * discounts)

    # Ideal DCG assumes all min(k, truth_len) relevant items are placed at the top of the list
    ideal_hits = np.zeros(max_k, dtype=np.float64)
    ideal_hits[: min(max_k, truth_len)] = 1.0
    idcg = np.cumsum(ideal_hits * discounts)
    ndcgs = np.where(idcg > 0, dcg / idcg, 0.0)

    # Slice out requested K values (e.g., K = 20)
    k_indices = [k - 1 for k in top_k_list]
    return {
        "Precision": precisions[k_indices],
        "Recall": recalls[k_indices],
        "NDCG": ndcgs[k_indices],
    }


class Evaluator:
    """
    All-ranking Top-K Evaluator matching `Evaluator` in `SGL-Torch/reckit/evaluator.py`.
    """

    def __init__(
        self,
        user_train_dict: dict[int, np.ndarray],
        user_test_dict: dict[int, np.ndarray],
        metrics: list[str] | None = None,
        top_k: int | list[int] = 20,
        batch_size: int = 128,
    ):
        """
        Args:
            user_train_dict: Mapping `user_id -> array of training item_ids` (masked during testing).
            user_test_dict: Mapping `user_id -> array of ground-truth test item_ids`.
            metrics: List of metric names to report (default: `["Precision", "Recall", "NDCG"]`).
            top_k: Integer K or list of K values for Top-K evaluation (default: `[20]`).
            batch_size: Number of test users evaluated per batch (default: 128 in `NeuRec.ini`).
        """
        if metrics is None:
            metrics = ["Precision", "Recall", "NDCG"]
        elif isinstance(metrics, str):
            metrics = [metrics]

        self.user_pos_train = user_train_dict if user_train_dict is not None else {}
        self.user_pos_test = user_test_dict
        self.metrics = metrics
        self.batch_size = batch_size

        if isinstance(top_k, int):
            self.top_show = [top_k]
        else:
            self.top_show = sorted(list(top_k))
        self.max_top = max(self.top_show)

    def metrics_info(self) -> str:
        """
        Returns a formatted header string of all (Metric@K) columns,
        matching `Evaluator.metrics_info` in `SGL-Torch`.
        """
        headers = []
        for k in self.top_show:
            for m in self.metrics:
                headers.append(f"{m}@{k}".ljust(12))
        return "\t".join(headers)

    def evaluate(self, model, test_users: list[int] | None = None) -> tuple[np.ndarray, str]:
        """
        Evaluates `model` on all test users under the all-ranking protocol (Section 4.1).

        Steps:
        1. Switches `model` to eval mode so final LightGCN representations are precomputed.
        2. Iterates over `test_users` in batches of `self.batch_size`.
        3. Predicts full-ranking scores `(B, num_items)` and sets training item scores to `-np.inf`.
        4. Extracts top-`max_top` items per user and computes Precision@K, Recall@K, and NDCG@K.

        Returns:
            final_result: 1D NumPy array of averaged metric values across all test users.
            buf: Formatted tab-separated string of metric values for logging.
        """
        model.eval()
        if test_users is None:
            test_users = [u for u, items in self.user_pos_test.items() if len(items) > 0]

        all_user_results = []

        for start in range(0, len(test_users), self.batch_size):
            batch_users = test_users[start : start + self.batch_size]
            ranking_score = model.predict(batch_users)  # shape: (B, num_items)
            ranking_score = np.array(ranking_score, dtype=np.float32, copy=True)

            # Mask observed training items with -inf so they are ranked at the very bottom
            for idx, u in enumerate(batch_users):
                if u in self.user_pos_train and len(self.user_pos_train[u]) > 0:
                    train_items = self.user_pos_train[u]
                    ranking_score[idx, train_items] = -np.inf

            # Extract top-`max_top` item indices for each user in the batch in descending score order
            # Using argpartition is O(N) per user, followed by sorting only the top-K slice
            top_k_part = np.argpartition(-ranking_score, self.max_top - 1, axis=1)[:, : self.max_top]
            row_indices = np.arange(len(batch_users))[:, None]
            top_k_scores = ranking_score[row_indices, top_k_part]
            top_k_order = np.argsort(-top_k_scores, axis=1)
            top_k_items = top_k_part[row_indices, top_k_order]  # shape: (B, max_top)

            # Compute metrics for each user in the batch
            for idx, u in enumerate(batch_users):
                truth_set = set(self.user_pos_test[u].tolist())
                user_metric_dict = compute_user_metrics(
                    ranked_items=top_k_items[idx],
                    ground_truth=truth_set,
                    top_k_list=self.top_show,
                )
                # Flatten in order: for each k in top_show, for each m in metrics
                user_vec = []
                for k_idx in range(len(self.top_show)):
                    for m in self.metrics:
                        user_vec.append(user_metric_dict[m][k_idx])
                all_user_results.append(user_vec)

        all_user_results = np.asarray(all_user_results, dtype=np.float64)
        final_result = np.mean(all_user_results, axis=0)
        buf = "\t".join([("%.8f" % x).ljust(12) for x in final_result])
        return final_result, buf

    def evaluate_long_tail_groups(
        self,
        model,
        train_items: np.ndarray,
        num_items: int,
        num_groups: int = 10,
        k: int = 20,
    ) -> np.ndarray:
        """
        Implements the Long-Tail Recommendation Decomposition from Section 4.3.1 (Figure 4):
            Recall = (1 / M) * sum_{u=1}^{M} sum_{g=1}^{10} |(l_rec^u)^{(g)} cap l_test^u| / |l_test^u|
                   = sum_{g=1}^{10} Recall^{(g)}

        Splits items into `num_groups` (10) groups sorted by item degree (popularity) such that
        the total number of training interactions in each group is approximately equal.
        Group 1 contains the least popular (long-tail) items; Group 10 contains the highest-degree
        (popular head) items.
        """
        model.eval()

        # 1. Compute item degrees in training set and partition items into 10 equal-interaction groups
        item_degrees = np.bincount(train_items, minlength=num_items)
        sorted_items = np.argsort(item_degrees)  # ascending by popularity
        sorted_degrees = item_degrees[sorted_items]
        cum_degrees = np.cumsum(sorted_degrees)
        total_interactions = cum_degrees[-1]

        item_to_group = np.zeros(num_items, dtype=np.int32)
        thresholds = np.linspace(0, total_interactions, num_groups + 1)[1:]
        group_ids = np.searchsorted(thresholds, cum_degrees, side="left")
        group_ids = np.clip(group_ids, 0, num_groups - 1)
        item_to_group[sorted_items] = group_ids

        # 2. Evaluate Recall^{(g)}@k across all test users
        test_users = [u for u, items in self.user_pos_test.items() if len(items) > 0]
        group_recalls = np.zeros(num_groups, dtype=np.float64)

        for start in range(0, len(test_users), self.batch_size):
            batch_users = test_users[start : start + self.batch_size]
            ranking_score = model.predict(batch_users)
            ranking_score = np.array(ranking_score, dtype=np.float32, copy=True)

            for idx, u in enumerate(batch_users):
                if u in self.user_pos_train and len(self.user_pos_train[u]) > 0:
                    ranking_score[idx, self.user_pos_train[u]] = -np.inf

            top_k_part = np.argpartition(-ranking_score, k - 1, axis=1)[:, :k]

            for idx, u in enumerate(batch_users):
                truth_set = set(self.user_pos_test[u].tolist())
                truth_len = len(truth_set)
                for rec_item in top_k_part[idx]:
                    if rec_item in truth_set:
                        g = item_to_group[int(rec_item)]
                        group_recalls[g] += 1.0 / truth_len

        group_recalls /= max(1, len(test_users))
        return group_recalls

