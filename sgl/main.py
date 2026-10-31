"""
main.py
=======
Main entry point to preprocess data, build and train the SGL (Self-Supervised Graph Learning)
model, and evaluate Top-K recommendation performance (Wu et al., SIGIR 2021).
Reference implementation: https://github.com/wujcan/SGL-Torch

This script orchestrates all 4 modules:
1. `preprocess.py`: Loads the dataset (or generates a synthetic dataset if none is present),
                    constructs the normalized user-item bipartite graph D^{-1/2} A D^{-1/2},
                    generates stochastic augmented subgraphs per epoch (Node Dropout 'nd',
                    Edge Dropout 'ed', or Random Walk 'rw'), and samples BPR triplets.
2. `model.py`:      Initializes the LightGCN backbone (`_LightGCN`) and `SGLModel`, performs
                    multi-view graph convolutions, and computes the multi-task objective:
                        L = L_main (BPR) + lambda_1 * L_ssl (InfoNCE) + lambda_2 * ||Theta||_2^2
3. `evaluate.py`:   Evaluates all-ranking Top-K metrics (`Precision@20`, `Recall@20`, `NDCG@20`)
                    with training-item masking, tracks early stopping, and decomposes Recall@20
                    across 10 item popularity groups (Section 4.3.1).
4. `main.py`:       Parses CLI arguments (matching `conf/SGL.ini` and `NeuRec.ini`), sets random
                    seeds, runs the epoch training & evaluation loop, and prints the final summary.
"""

import argparse
import os
import random
from time import time
import numpy as np
import torch

from sgl.preprocess import SGLDataset, PairwiseSampler
from sgl.model import SGLModel
from sgl.evaluate import Evaluator


def set_seed(seed: int = 2021) -> None:
    """
    Fixes random seeds across Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.
    Default seed in `NeuRec.ini` is 2021.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    """
    Parses command-line hyperparameters matching `conf/SGL.ini` and `NeuRec.ini` in `SGL-Torch`.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Implementation of Self-Supervised Graph Learning (SGL, SIGIR 2021)"
    )

    # Dataset & Environment arguments (from NeuRec.ini)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/synthetic",
        help="Path to dataset directory containing <dataset>.train and <dataset>.test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset name (e.g., yelp2018, amazon-book, ifashion, synthetic)",
    )
    parser.add_argument("--seed", type=int, default=2021, help="Random seed (default: 2021)")
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[20],
        help="Top-K cutoff(s) for all-ranking evaluation (default: 20)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="Batch size for test user evaluation (default: 128)",
    )
    parser.add_argument(
        "--start_testing_epoch",
        type=int,
        default=0,
        help="Epoch after which evaluation starts (default: 0)",
    )

    # Model & Training arguments (from conf/SGL.ini)
    parser.add_argument(
        "--aug_type",
        type=str,
        default="ED",
        choices=["ND", "ED", "RW", "nd", "ed", "rw"],
        help="Graph augmentation operator: 'ND' (Node Dropout), 'ED' (Edge Dropout), 'RW' (Random Walk)",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=64,
        help="Embedding dimension size d (default: 64)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="Number of LightGCN propagation layers L (default: 3)",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient lambda_2 (default: 1e-4)",
    )
    parser.add_argument(
        "--ssl_reg",
        type=float,
        default=0.1,
        help="Self-supervised learning loss weight lambda_1 (default: 0.1)",
    )
    parser.add_argument(
        "--ssl_ratio",
        type=float,
        default=0.1,
        help="Dropout ratio rho for graph augmentation (default: 0.1)",
    )
    parser.add_argument(
        "--ssl_temp",
        type=float,
        default=0.2,
        help="Softmax temperature tau for InfoNCE contrastive loss (default: 0.2)",
    )
    parser.add_argument(
        "--ssl_mode",
        type=str,
        default="both_side",
        choices=["user_side", "item_side", "both_side"],
        help="Contrastive loss mode: 'user_side', 'item_side', or 'both_side'",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Adam optimizer learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs (default: 20 for quick demo; 1000 in SGL.ini)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Training mini-batch size B (default: 2048)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Evaluate model every `verbose` epochs (default: 1)",
    )
    parser.add_argument(
        "--stop_cnt",
        type=int,
        default=10,
        help="Early stopping patience counter (default: 10)",
    )
    parser.add_argument(
        "--param_init",
        type=str,
        default="xavier_uniform",
        help="Weight initialization method (default: xavier_uniform)",
    )
    parser.add_argument(
        "--save_flag",
        type=int,
        default=0,
        help="Whether to save the best model embeddings to disk (0 or 1)",
    )
    parser.add_argument(
        "--pretrain_flag",
        type=int,
        default=0,
        help="Whether to load pretrained embeddings (SGL-pre, Section 4.4.2)",
    )

    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> None:
    """
    Executes the complete SGL workflow:
    1. Data preprocessing & bipartite graph construction (`SGLDataset`).
    2. SGL model & Adam optimizer initialization (`SGLModel`).
    3. Per-epoch graph view augmentation (`sub_graph1`, `sub_graph2`), mini-batch optimization
       of `L = L_main + lambda_1 * L_ssl + lambda_2 * ||Theta||_2^2`, and periodic Top-K evaluation.
    """
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ssl_aug_type = args.aug_type.lower()

    print("=" * 80)
    print("Self-Supervised Graph Learning (SGL) for Recommendation (SIGIR '21)")
    print("=" * 80)
    print(
        f"[Config] dataset={args.dataset} | aug_type={args.aug_type.upper()} | "
        f"n_layers={args.n_layers} | embed_size={args.embed_size} | lr={args.lr} | "
        f"batch_size={args.batch_size}"
    )
    print(
        f"[SSL]    ssl_reg(lambda_1)={args.ssl_reg} | ssl_temp(tau)={args.ssl_temp} | "
        f"ssl_ratio(rho)={args.ssl_ratio} | ssl_mode={args.ssl_mode} | reg(lambda_2)={args.reg}"
    )
    print(f"[Device] Using {device}")

    # --------------------------------------------------------------------------
    # Step 1: Load and preprocess interaction data & construct intact graph G
    # --------------------------------------------------------------------------
    dataset = SGLDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        ssl_ratio=args.ssl_ratio,
        device=device,
    )
    print(
        f"[Data]   #Users: {dataset.num_users} | #Items: {dataset.num_items} | "
        f"#Train Interactions: {dataset.num_ratings}"
    )

    # Initialize the Pairwise BPR Sampler and All-Ranking Evaluator
    data_iter = PairwiseSampler(dataset, batch_size=args.batch_size, shuffle=True)
    evaluator = Evaluator(
        user_train_dict=dataset.train_user_dict,
        user_test_dict=dataset.test_user_dict,
        metrics=["Precision", "Recall", "NDCG"],
        top_k=args.top_k,
        batch_size=args.test_batch_size,
    )

    # --------------------------------------------------------------------------
    # Step 2: Initialize SGL model (LightGCN + Contrastive SSL) & Adam Optimizer
    # --------------------------------------------------------------------------
    model = SGLModel(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        norm_adj=dataset.norm_adj,
        embed_dim=args.embed_size,
        n_layers=args.n_layers,
        ssl_temp=args.ssl_temp,
        ssl_reg=args.ssl_reg,
        reg=args.reg,
        ssl_mode=args.ssl_mode,
        param_init=args.param_init,
    ).to(device)

    save_dir = os.path.join(args.data_dir, "saved_model")
    if args.pretrain_flag:
        model.lightgcn.reset_parameters(pretrain=1, pretrain_dir=save_dir + "/")

    optimizer = torch.optim.Adam(model.lightgcn.parameters(), lr=args.lr)

    # --------------------------------------------------------------------------
    # Step 3: Multi-task Training Loop & All-Ranking Top-K Evaluation
    #         Mirrors `SGL.train_model` in `SGL-Torch/model/general_recommender/SGL.py`
    # --------------------------------------------------------------------------
    print("-" * 80)
    print("Metrics:\t" + evaluator.metrics_info())
    print("-" * 80)

    best_epoch = 0
    best_result = np.zeros(3 * len(args.top_k), dtype=np.float64)
    best_state_dict = None
    stopping_step = 0

    for epoch in range(1, args.epochs + 1):
        total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
        training_start_time = time()

        # Generate two augmented graph views s_1(G) and s_2(G) at the start of the epoch (Section 3.1)
        # - For ND and ED: two single sparse adjacency matrices shared across all L layers
        # - For RW: two lists of L independent edge-dropped sparse adjacency matrices
        sub_graph1, sub_graph2 = dataset.sample_subgraphs(
            aug_type=ssl_aug_type, n_layers=args.n_layers
        )

        model.train()
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            bat_users = torch.from_numpy(bat_users).long().to(device)
            bat_pos_items = torch.from_numpy(bat_pos_items).long().to(device)
            bat_neg_items = torch.from_numpy(bat_neg_items).long().to(device)

            # Compute multi-task loss: L = L_main + ssl_reg * L_ssl + reg * L_reg
            loss, bpr_l, ssl_l, reg_l = model.calculate_loss(
                sub_graph1=sub_graph1,
                sub_graph2=sub_graph2,
                bat_users=bat_users,
                bat_pos_items=bat_pos_items,
                bat_neg_items=bat_neg_items,
            )

            total_loss += loss.item()
            total_bpr_loss += bpr_l.item()
            total_reg_loss += reg_l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log epoch loss breakdown normalized by total training interactions (as in SGL-Torch)
        ssl_component = total_loss - total_bpr_loss - total_reg_loss
        print(
            "[iter %d : loss : %.4f = %.4f (BPR) + %.4f (SSL) + %.4f (Reg), time: %.2fs]"
            % (
                epoch,
                total_loss / dataset.num_ratings,
                total_bpr_loss / dataset.num_ratings,
                ssl_component / dataset.num_ratings,
                total_reg_loss / dataset.num_ratings,
                time() - training_start_time,
            )
        )

        # Periodic Top-K evaluation on the test set
        if epoch % args.verbose == 0 and epoch > args.start_testing_epoch:
            current_result, buf = evaluator.evaluate(model)
            print(f"epoch {epoch}:\t{buf}")

            # Check if Recall@K (index 1 in [Precision, Recall, NDCG]) improved
            if best_result[1] < current_result[1]:
                best_result = current_result
                best_epoch = epoch
                stopping_step = 0
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.lightgcn.state_dict().items()
                }
                print("  --> Found a better model.")
                if args.save_flag:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.lightgcn.state_dict(), os.path.join(save_dir, "best_sgl.pt"))
            else:
                stopping_step += 1
                if stopping_step >= args.stop_cnt:
                    print(f"Early stopping triggered at epoch: {epoch}")
                    break

    # --------------------------------------------------------------------------
    # Step 4: Final Summary & Long-Tail Group Recall Analysis (Section 4.3.1)
    # --------------------------------------------------------------------------
    print("=" * 80)
    print(f"Best Result @ Epoch {best_epoch}:")
    print("Metrics:\t" + evaluator.metrics_info())
    best_buf = "\t".join([("%.8f" % x).ljust(12) for x in best_result])
    print(f"Values: \t{best_buf}")

    if best_state_dict is not None:
        model.lightgcn.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    group_recalls = evaluator.evaluate_long_tail_groups(
        model=model,
        train_items=dataset.train_items,
        num_items=dataset.num_items,
        num_groups=10,
        k=args.top_k[0],
    )
    print("-" * 80)
    print(
        f"Long-Tail Item Popularity Group Recall@{args.top_k[0]} Decomposition "
        "(Group 1 = Tail, Group 10 = Head):"
    )
    group_str = " | ".join([f"G{g+1}: {val:.4f}" for g, val in enumerate(group_recalls)])
    print(f"  {group_str}")
    print(f"  Sum of Group Recalls: {np.sum(group_recalls):.6f}")
    print("=" * 80)


if __name__ == "__main__":
    cli_args = parse_args()
    train_and_evaluate(cli_args)

