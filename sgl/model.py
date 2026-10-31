"""
model.py
========
PyTorch implementation of Self-Supervised Graph Learning (SGL) for Recommendation
built upon the LightGCN backbone (Wu et al., SIGIR 2021).
Reference implementation: https://github.com/wujcan/SGL-Torch (`model/general_recommender/SGL.py`)

Key components implemented in this module:
1. `_LightGCN` (Graph Encoder & Multi-View Forward Pass):
   - Maintains trainable 0-th layer ID embeddings `user_embeddings` and `item_embeddings`
     initialized with Xavier uniform (`xavier_uniform_`).
   - `_forward_gcn(norm_adj)` performs L layers of LightGCN message passing (Eq. 1-3):
         Z^{(k)} = A_hat * Z^{(k-1)}
         Z = (1 / (L + 1)) * sum_{k=0}^{L} Z^{(k)}
     Supports both a single sparse adjacency matrix (for main graph, Node Dropout 'nd',
     and Edge Dropout 'ed') and a list of L layer-specific sparse adjacency matrices
     (for Random Walk 'rw', Eq. 9).
   - `forward(sub_graph1, sub_graph2, users, items, neg_items)` computes:
     * Main view representations from `self.norm_adj` for BPR ranking logits:
           sup_logits = <e_u, e_i> - <e_u, e_j>
     * L2-normalized augmented view representations from `sub_graph1` and `sub_graph2`
       for InfoNCE contrastive learning logits:
           ssl_logits_user = E''_U * e'_u - <e'_u, e''_u>
           ssl_logits_item = E''_I * e'_i - <e'_i, e''_i>
2. Loss Functions (`inner_product`, `bpr_loss`, `l2_loss`, and InfoNCE loss):
   - Pairwise BPR loss (Eq. 5):
         L_main = -sum_{(u, i, j)} log(sigmoid(sup_logits))
   - InfoNCE Self-Supervised Loss (Eq. 10):
         L_ssl = sum_{u} log(sum_{v in U} exp((<s'_u, s''_v> - <s'_u, s''_u>) / tau))
               + sum_{i} log(sum_{v in I} exp((<s'_i, s''_v> - <s'_i, s''_i>) / tau))
   - L2 Regularization Loss on 0-th layer ego embeddings:
         L_reg = 0.5 * sum(||e_u^{(0)}||_2^2 + ||e_i^{(0)}||_2^2 + ||e_j^{(0)}||_2^2)
   - Multi-task joint objective (Eq. 11):
         L = L_main + ssl_reg * L_ssl + reg * L_reg
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as torch_sp


def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the row-wise inner product between two embedding tensors (Eq. 4).
    Mirrors `inner_product` in `SGL-Torch/util/pytorch/func.py`.
    """
    return torch.sum(a * b, dim=-1)


def bpr_loss(y_diff: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """
    Pairwise Bayesian Personalized Ranking (BPR) loss (Section 2, Eq. 5):
        L_main = sum_{(u, i, j) in O} -log(sigmoid(y_hat_{ui} - y_hat_{uj}))
    Mirrors `bpr_loss` in `SGL-Torch/util/pytorch/loss.py`.
    """
    loss = -F.logsigmoid(y_diff)
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def l2_loss(*weights: torch.Tensor) -> torch.Tensor:
    """
    Half L2-norm squared regularization loss over 0-th layer ego embeddings:
        0.5 * sum_w ||w||_2^2
    Mirrors `l2_loss` in `SGL-Torch/util/pytorch/loss.py`.
    """
    loss = 0.0
    for w in weights:
        loss = loss + torch.sum(torch.pow(w, 2))
    return 0.5 * loss


class _LightGCN(nn.Module):
    """
    LightGCN encoder module equipped with SGL dual-subgraph contrastive forward pass.
    Directly mirrors `_LightGCN` in `SGL-Torch/model/general_recommender/SGL.py`.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        norm_adj: torch.Tensor,
        n_layers: int = 3,
    ):
        """
        Args:
            num_users: Total number of users |U|.
            num_items: Total number of items |I|.
            embed_dim: Embedding dimension d (default: 64 in paper).
            norm_adj: Intact normalized bipartite graph adjacency tensor D^{-1/2} A D^{-1/2}.
            n_layers: Number of LightGCN propagation layers L (default: 3 in paper).
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers

        # Trainable 0-th layer ID embeddings Z^{(0)} (no extra parameters are added by SGL)
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)

        # Cached final representations used during fast full-ranking evaluation
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        self.reset_parameters(init_method="xavier_uniform")

    def reset_parameters(
        self,
        pretrain: int = 0,
        init_method: str = "xavier_uniform",
        pretrain_dir: str | None = None,
    ) -> None:
        """
        Initializes ego user and item embeddings using Xavier uniform initialization
        (Section 4.1.2) or loads pretrained embeddings for SGL-pre (Section 4.4.2).
        """
        if pretrain and pretrain_dir is not None:
            pretrain_user = np.load(pretrain_dir + "user_embeddings.npy")
            pretrain_item = np.load(pretrain_dir + "item_embeddings.npy")
            device = self.user_embeddings.weight.device
            self.user_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrain_user).to(device), freeze=False
            )
            self.item_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrain_item).to(device), freeze=False
            )
        else:
            if init_method == "xavier_uniform":
                nn.init.xavier_uniform_(self.user_embeddings.weight)
                nn.init.xavier_uniform_(self.item_embeddings.weight)
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(self.user_embeddings.weight)
                nn.init.xavier_normal_(self.item_embeddings.weight)
            elif init_method == "normal":
                nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.01)
                nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.01)
            else:
                nn.init.uniform_(self.user_embeddings.weight, a=-0.01, b=0.01)
                nn.init.uniform_(self.item_embeddings.weight, a=-0.01, b=0.01)

    def _forward_gcn(
        self, norm_adj: torch.Tensor | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagates user and item embeddings over L LightGCN layers (Section 2, Eq. 1-3):
            Z^{(k)} = H(Z^{(k-1)}, G) = A_hat * Z^{(k-1)}
            Z = f_readout({Z^{(0)}, ..., Z^{(L)}}) = mean(Z^{(0)}, ..., Z^{(L)})

        - If `norm_adj` is a `list` of L sparse tensors (Random Walk 'rw', Eq. 9),
          layer k multiplies with `norm_adj[k]`.
        - Otherwise (main graph G, Node Dropout 'nd', Edge Dropout 'ed', Eq. 7-8),
          all layers multiply with the same `norm_adj`.
        """
        # Concatenate user and item 0-th layer ID embeddings: shape (|U| + |I|, d)
        ego_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Readout function (Eq. 3): uniform average across all layers 0, 1, ..., L
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        # Split back into user representations (|U|, d) and item representations (|I|, d)
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_embeddings, item_embeddings

    def forward(
        self,
        sub_graph1: torch.Tensor | list[torch.Tensor],
        sub_graph2: torch.Tensor | list[torch.Tensor],
        users: torch.Tensor,
        items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass for both the main supervised recommendation task and
        the auxiliary self-supervised contrastive learning task.

        Directly mirrors `_LightGCN.forward` in `SGL-Torch/model/general_recommender/SGL.py`.

        Args:
            sub_graph1: First augmented graph view s_1(G).
            sub_graph2: Second augmented graph view s_2(G).
            users: Mini-batch user indices `u` of shape (B,).
            items: Mini-batch positive item indices `i` of shape (B,).
            neg_items: Mini-batch negative item indices `j` of shape (B,).

        Returns:
            sup_logits: Pairwise BPR score differences `y_hat_{ui} - y_hat_{uj}` of shape (B,).
            ssl_logits_user: Contrastive logits `s(z'_u, z''_v) - s(z'_u, z''_u)` of shape (B, |U|).
            ssl_logits_item: Contrastive logits `s(z'_i, z''_v) - s(z'_i, z''_i)` of shape (B, |I|).
        """
        # 1. Encode node representations on intact graph G and two augmented views s_1(G), s_2(G)
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # 2. L2-normalize representations from augmented views to compute cosine similarity s(.)
        #    s'_u = z'_u / ||z'_u||_2, s''_u = z''_u / ||z''_u||_2 (Section 3.2 & Section 3.4)
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        # 3. Look up mini-batch embeddings
        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)

        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        # 4. Main supervised task logits: y_hat_{ui} - y_hat_{uj} (Eq. 4-5)
        sup_pos_ratings = inner_product(user_embs, item_embs)      # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)  # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings             # [batch_size]

        # 5. Self-supervised InfoNCE logits (Eq. 10):
        #    Positive pair cosine similarity: s(z'_u, z''_u) and s(z'_i, z''_i)
        pos_ratings_user = inner_product(user_embs1, user_embs2)   # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)   # [batch_size]

        #    All-pair cosine similarity against the entire user/item space:
        #    s(z'_u, z''_v) for all v in U, and s(z'_i, z''_v) for all v in I
        tot_ratings_user = torch.matmul(
            user_embs1, torch.transpose(user_embeddings2, 0, 1)
        )  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(
            item_embs1, torch.transpose(item_embeddings2, 0, 1)
        )  # [batch_size, num_items]

        #    Subtract positive similarity so logsumexp(ssl_logits / tau) directly yields:
        #    -log( exp(pos / tau) / sum_v exp(tot / tau) )
        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]  # [batch_size, num_items]

        return sup_logits, ssl_logits_user, ssl_logits_item

    def eval(self):
        """
        Switches the module to evaluation mode and precomputes final multi-layer
        user and item representations on the intact graph G (`self.norm_adj`).
        """
        super().eval()
        with torch.no_grad():
            self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(
                self.norm_adj
            )
        return self

    @torch.no_grad()
    def predict(self, users: torch.Tensor) -> torch.Tensor:
        """
        Computes full-ranking inner-product preference scores y_hat_{ui} = z_u^T z_i
        for a batch of users against all items I.
        """
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(
                self.norm_adj
            )
        user_embs = F.embedding(users, self._user_embeddings_final)
        ratings = torch.matmul(user_embs, self._item_embeddings_final.T)
        return ratings


class SGLModel(nn.Module):
    """
    High-level SGL wrapper combining `_LightGCN` with the joint multi-task loss computation
    (BPR loss + InfoNCE SSL loss + L2 regularization) from `SGL.train_model` in `SGL-Torch`.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        embed_dim: int = 64,
        n_layers: int = 3,
        ssl_temp: float = 0.2,
        ssl_reg: float = 0.1,
        reg: float = 1e-4,
        ssl_mode: str = "both_side",
        param_init: str = "xavier_uniform",
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.reg = reg
        self.ssl_mode = ssl_mode.lower()

        self.lightgcn = _LightGCN(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            norm_adj=norm_adj,
            n_layers=n_layers,
        )
        self.lightgcn.reset_parameters(init_method=param_init)

    def calculate_loss(
        self,
        sub_graph1: torch.Tensor | list[torch.Tensor],
        sub_graph2: torch.Tensor | list[torch.Tensor],
        bat_users: torch.Tensor,
        bat_pos_items: torch.Tensor,
        bat_neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the multi-task SGL training loss for a mini-batch (Section 3.3, Eq. 11):
            loss = bpr_loss + ssl_reg * infonce_loss + reg * reg_loss

        Directly mirrors the loss calculation inside `SGL.train_model` in `SGL-Torch`.
        """
        sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
            sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items
        )

        # 1. Supervised BPR Loss (Eq. 5)
        batch_bpr_loss = bpr_loss(sup_logits, reduction="sum")

        # 2. L2 Regularization Loss on 0-th layer ego embeddings (Eq. 11)
        batch_reg_loss = l2_loss(
            self.lightgcn.user_embeddings(bat_users),
            self.lightgcn.item_embeddings(bat_pos_items),
            self.lightgcn.item_embeddings(bat_neg_items),
        )

        # 3. InfoNCE Contrastive Self-Supervised Loss (Eq. 10)
        #    logsumexp((tot_ratings - pos_ratings) / tau) = -log(exp(pos/tau) / sum(exp(tot/tau)))
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)

        if self.ssl_mode == "user_side":
            infonce_loss = torch.sum(clogits_user)
        elif self.ssl_mode == "item_side":
            infonce_loss = torch.sum(clogits_item)
        else:
            # Default 'both_side': L_ssl = L_ssl^user + L_ssl^item
            infonce_loss = torch.sum(clogits_user + clogits_item)

        weighted_ssl_loss = self.ssl_reg * infonce_loss
        weighted_reg_loss = self.reg * batch_reg_loss
        total_loss = batch_bpr_loss + weighted_ssl_loss + weighted_reg_loss

        return total_loss, batch_bpr_loss, weighted_ssl_loss, weighted_reg_loss

    def eval(self):
        """Switches the underlying LightGCN encoder to eval mode and caches embeddings."""
        self.lightgcn.eval()
        return super().eval()

    @torch.no_grad()
    def predict(self, users: np.ndarray | list[int] | torch.Tensor) -> np.ndarray:
        """
        Predicts full-ranking scores for `users` across all items and returns a NumPy array,
        matching `SGL.predict` in `SGL-Torch`.
        """
        if not isinstance(users, torch.Tensor):
            device = self.lightgcn.user_embeddings.weight.device
            users = torch.from_numpy(np.asarray(users)).long().to(device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
