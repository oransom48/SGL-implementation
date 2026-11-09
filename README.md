# Self-Supervised Graph Learning (SGL) for Recommendation — Implementation & Comparison with NGCF and LightGCN

This repository provides a PyTorch implementation of **Self-Supervised Graph Learning for Recommendation (SGL)** (*SIGIR 2021*) and benchmarks it against two foundational Graph Collaborative Filtering (GCF) models—**NGCF** (*SIGIR 2019*) and **LightGCN** (*SIGIR 2020*)—under a **unified data preprocessing, splitting, and all-ranking evaluation pipeline**.

---

## 1. Project Overview & Motivation

Graph Convolutional Networks (GCNs) have become the de facto standard for collaborative filtering by modeling user–item interactions as a bipartite graph and propagating embeddings along higher-order connectivity paths. However, supervised GCN recommenders (such as **NGCF** and **LightGCN**) suffer from two key limitations:
1. **High-Degree Node Dominance (Popularity Bias)**: High-degree (popular) items exert disproportionately large influence during neighborhood aggregation, while long-tail items receive sparse supervisory signals.
2. **Vulnerability to Interaction Noise**: Real-world implicit feedback contains noisy interactions that compound across multi-hop message passing layers.

**Self-Supervised Graph Learning (SGL)** addresses these challenges by introducing an auxiliary **contrastive self-supervised learning (SSL)** task alongside the classical Bayesian Personalized Ranking (BPR) loss. By generating two stochastic structural views of the user–item graph via **Node Dropout (ND)**, **Edge Dropout (ED)**, or **Random Walk (RW)** and maximizing agreement between node representations across views via an **InfoNCE** loss, SGL improves both recommendation accuracy and long-tail item coverage.

### Core Objectives of This Project
- **Implement SGL from scratch in PyTorch** (supporting `SGL-ND`, `SGL-ED`, and `SGL-RW`) faithful to the [SGL paper](https://arxiv.org/abs/2010.10783) and the official [`wujcan/SGL-Torch`](https://github.com/wujcan/SGL-Torch) repository.
- **Implement NGCF and LightGCN baselines** following their official implementations ([`xiangwang1223/neural_graph_collaborative_filtering`](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) and [`gusye1234/LightGCN-PyTorch`](https://github.com/gusye1234/LightGCN-PyTorch)) and the baseline hyperparameter setup described in Section 4.1.2 of the SGL paper.
- **Compare SGL, LightGCN, and NGCF fairly** using an identical `7 : 1 : 2` (`train : val : test`) per-user data split on `yelp2018.train`, identical BPR negative sampling, and identical full-ranking Top-$K$ and **Long-Tail Group Recall@20** evaluation protocols.

---

## 2. Model Comparison Summary

| Feature / Component | **NGCF** (SIGIR '19) | **LightGCN** (SIGIR '20) | **SGL** (SIGIR '21) |
| :--- | :--- | :--- | :--- |
| **Graph Encoder Backbone** | Bi-Interaction GCN | Light Graph Convolution | Light Graph Convolution |
| **Feature Transformation ($`\mathbf{W}_1, \mathbf{W}_2`$)** | Yes (1st-order sum + 2nd-order element-wise product) | None (Identity) | None (Identity) |
| **Nonlinear Activation** | `LeakyReLU(0.2)` + Message Dropout | None | None |
| **Layer Combination (Readout)** | Concatenation: $`\mathbf{e}_u^* = [\mathbf{e}_u^{(0)} \parallel \dots \parallel \mathbf{e}_u^{(L)}]`$ | Uniform Average: $`\mathbf{e}_u^* = \frac{1}{L+1}\sum_{l=0}^L \mathbf{e}_u^{(l)}`$ | Uniform Average: $`\mathbf{e}_u^* = \frac{1}{L+1}\sum_{l=0}^L \mathbf{e}_u^{(l)}`$ |
| **Graph Augmentation** | Optional Sparse Laplacian Dropout | Optional Edge Dropout | Multi-View Structural Augmentation (**ND**, **ED**, **RW**) |
| **Training Objective** | $`\mathcal{L}_{\text{BPR}} + \lambda_2 \Vert\Theta\Vert_2^2`$ | $`\mathcal{L}_{\text{BPR}} + \lambda_2 \Vert\Theta\Vert_2^2`$ | $`\mathcal{L}_{\text{BPR}} + \lambda_1 \mathcal{L}_{\text{ssl}} + \lambda_2 \Vert\Theta\Vert_2^2`$ |

### Mathematical Formulations

#### 1. NGCF Message Propagation & Layer Aggregation

The message propagation at layer $l$ aggregates neighbor representations with first-order and second-order feature interactions:

```math
\mathbf{e}_u^{(l)} = \text{LeakyReLU}\left(\mathbf{W}_1^{(l)}\left(\mathbf{e}_u^{(l-1)} + \sum_{i \in \mathcal{N}_u} \tilde{A}_{ui}\mathbf{e}_i^{(l-1)}\right) + \mathbf{b}_1^{(l)} + \mathbf{W}_2^{(l)}\left(\mathbf{e}_u^{(l-1)} \odot \sum_{i \in \mathcal{N}_u} \tilde{A}_{ui}\mathbf{e}_i^{(l-1)}\right) + \mathbf{b}_2^{(l)}\right)
```

The representations from all $L+1$ propagation layers are normalized and concatenated to form the final embedding:

```math
\mathbf{e}_u^* = [\mathbf{e}_u^{(0)} \,\|\, \mathbf{e}_u^{(1)} \,\|\, \dots \,\|\, \mathbf{e}_u^{(L)}]
```

#### 2. LightGCN Linear Propagation & Readout

LightGCN removes feature transformation matrices and nonlinear activations, performing linear neighborhood aggregation:

```math
\mathbf{E}^{(l)} = \tilde{\mathbf{A}}\mathbf{E}^{(l-1)}, \qquad \tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
```

The final embedding combines all layer representations via a uniform average:

```math
\mathbf{E}^* = \frac{1}{L+1}\sum_{l=0}^{L} \mathbf{E}^{(l)}
```

#### 3. SGL Multi-Task Objective (BPR + InfoNCE Contrastive SSL)

The complete joint loss function combines the primary recommendation task (BPR loss), the self-supervised graph contrastive task (InfoNCE loss), and L2 regularization:

```math
\mathcal{L} = \mathcal{L}_{\text{BPR}} + \lambda_1 \mathcal{L}_{\text{ssl}} + \lambda_2 \|\mathbf{E}^{(0)}\|^2
```

The primary recommendation objective uses the Bayesian Personalized Ranking (BPR) loss:

```math
\mathcal{L}_{\text{BPR}} = \sum_{(u, i, j) \in \mathcal{O}} -\log \sigma(\hat{y}_{ui} - \hat{y}_{uj}), \qquad \hat{y}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i
```

The auxiliary self-supervised learning loss is the sum of user-side and item-side contrastive objectives:

```math
\mathcal{L}_{\text{ssl}} = \mathcal{L}_{\text{ssl}}^{\text{user}} + \mathcal{L}_{\text{ssl}}^{\text{item}}
```

For the user side, given two augmented graph views $s_1(\mathcal{G})$ and $s_2(\mathcal{G})$ generated via Node Dropout, Edge Dropout, or Random Walk, the InfoNCE objective maximizes the agreement between positive pairs of the same user against all other users as negatives:

```math
\mathcal{L}_{\text{ssl}}^{\text{user}} = \sum_{u \in \mathcal{U}} -\log \frac{\exp(\mathbf{z}_u^{(1)\top} \mathbf{z}_u^{(2)} / \tau)}{\sum_{v \in \mathcal{U}} \exp(\mathbf{z}_u^{(1)\top} \mathbf{z}_v^{(2)} / \tau)}
```

Similarly, the item-side contrastive loss is defined as:

```math
\mathcal{L}_{\text{ssl}}^{\text{item}} = \sum_{i \in \mathcal{I}} -\log \frac{\exp(\mathbf{z}_i^{(1)\top} \mathbf{z}_i^{(2)} / \tau)}{\sum_{j \in \mathcal{I}} \exp(\mathbf{z}_i^{(1)\top} \mathbf{z}_j^{(2)} / \tau)}
```

where $\mathbf{z}_u^{(1)}, \mathbf{z}_u^{(2)}$ (and $\mathbf{z}_i^{(1)}, \mathbf{z}_i^{(2)}$) are the L2-normalized representations under the two augmented graph views, $\tau$ is the InfoNCE temperature hyperparameter (`ssl_temp`), and $\lambda_1, \lambda_2$ are the loss weights (`ssl_reg` and `reg`).

---

## 3. Unified Data & Evaluation Pipeline

All notebooks in `script/` share the exact same data pipeline (`SGLDataset`, `PairwiseSampler`, and `Evaluator`):

1. **Single-File Dataset Loading (`yelp2018.train`)**:
   - Automatically downloads `yelp2018.train` from the official [`wujcan/SGL-Torch`](https://github.com/wujcan/SGL-Torch/tree/master/dataset/yelp2018) repository (`31,668` users, `38,048` items, `1,237,259` interactions in `yelp2018.train`).
   - Also supports `dataset="synthetic"` to generate a power-law synthetic dataset for quick CPU smoke tests.
2. **Per-User `7 : 1 : 2` (`train : val : test`) Split**:
   - Each user's interaction history in `yelp2018.train` is deduplicated, deterministically shuffled (`seed=2021`), and partitioned with `split_ratios = (0.7, 0.1, 0.2)`:
     - **70% Training (`train_user_dict`)**: Used exclusively to build the sparse normalized adjacency matrix $\tilde{\mathbf{A}}$, sample SGL subgraphs, and draw BPR training triplets $(u, i, j)$ via `PairwiseSampler` (preventing any data leakage).
     - **10% Validation (`val_user_dict`)**: Evaluated every `verbose` epochs via `val_evaluator` (masking the 70% training items) for model selection (`best_state_dict`) and early stopping (`stop_cnt`).
     - **20% Testing (`test_user_dict`)**: Evaluated once at the end of training on the restored best validation checkpoint via `test_evaluator` (masking the combined 80% `train + val` items in `train_val_user_dict`).
3. **Evaluation Metrics**:
   - **All-Ranking Top-$K$ Protocol ($K = 20$)**: Ranks all items in $\mathcal{I}$ for each user (with observed positives masked to $-\infty$) and reports **`Precision@20`**, **`Recall@20`**, and **`NDCG@20`**.
   - **Long-Tail Item Group Decomposition (Paper Section 4.3.1, Figure 4)**: Partitions all items into 10 groups (`G1` = lowest-degree long-tail items $\rightarrow$ `G10` = highest-degree popular head items) such that each group accounts for an equal $10\%$ share of total training interactions, and measures each group's contribution to overall `Recall@20`.

---

## 4. Repository Structure

```text
SGL-implementation/
├── script/
│   ├── SGL_Colab.ipynb          # End-to-end SGL (ND / ED / RW) Colab notebook (7:1:2 split)
│   ├── LightGCN_Colab.ipynb     # End-to-end LightGCN baseline Colab notebook (7:1:2 split)
│   ├── NGCF_Colab.ipynb         # End-to-end NGCF baseline Colab notebook (7:1:2 split)
│   └── SGL_Colab_doh.ipynb      # Alternative self-contained SGL-ED notebook
├── sgl/
│   └── sgl_model_doh.py         # Standalone SGL-ED recommender module (SGLRecommender)
└── README.md                    # Project documentation
```

### Key Files
- **[`script/SGL_Colab.ipynb`](./script/SGL_Colab.ipynb)**: Complete self-contained notebook implementing `SGLDataset`, `PairwiseSampler`, `LightGCN_Encoder`, `SGLModel` (`ND`, `ED`, `RW`), `Evaluator`, training/validation loop, and 3-panel visualization (Training Losses, Validation Recall/NDCG curves, and Test Long-Tail Group Recall@20).
- **[`script/LightGCN_Colab.ipynb`](./script/LightGCN_Colab.ipynb)**: Implements `LightGCN` following [`gusye1234/LightGCN-PyTorch`](https://github.com/gusye1234/LightGCN-PyTorch) plugged into the exact same preprocessing and evaluation pipeline as `SGL_Colab.ipynb`.
- **[`script/NGCF_Colab.ipynb`](./script/NGCF_Colab.ipynb)**: Implements `NGCF` following [`xiangwang1223/neural_graph_collaborative_filtering`](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) with the SGL paper's baseline configuration (`embed_size=64`, `layer_size=[64, 64, 64]`, `lr=0.001`, `batch_size=2048`), plugged into the same preprocessing and evaluation pipeline.
- **[`sgl/sgl_model_doh.py`](./sgl/sgl_model_doh.py)** & **[`script/SGL_Colab_doh.ipynb`](./script/SGL_Colab_doh.ipynb)**: Self-contained `SGLRecommender` module (`fit()` / `predict()` API) for the Edge Dropout (`SGL-ED`) variant.

---

## 5. Getting Started

### Running in Google Colab (Recommended)
1. Upload any of the notebooks in [`script/`](./script/) (`SGL_Colab.ipynb`, `LightGCN_Colab.ipynb`, or `NGCF_Colab.ipynb`) to **Google Colab**.
2. Select a GPU runtime (**Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**).
3. Run all cells (**Runtime $\rightarrow$ Run all**). The notebook will automatically download `yelp2018.train`, perform the `7:1:2` per-user split, train the model, validate per epoch, and output final test metrics and plots.

### Running Locally
Ensure you have Python 3.9+ and install the required dependencies:
```bash
pip install torch numpy scipy matplotlib jupyter
```
Then open and execute the notebooks in VS Code or Jupyter Lab:
```bash
jupyter lab script/
```

---

## 6. Default Hyperparameters (SGL Paper Setup)

All three comparative notebooks use consistent core hyperparameters matching **Section 4.1.2** of the SGL paper:

| Parameter | SGL (`SGLConfig`) | LightGCN (`LightGCNConfig`) | NGCF (`NGCFConfig`) | Description |
| :--- | :---: | :---: | :---: | :--- |
| `dataset` | `"yelp2018"` | `"yelp2018"` | `"yelp2018"` | Loads `yelp2018.train` |
| `split_ratios` | `(0.7, 0.1, 0.2)` | `(0.7, 0.1, 0.2)` | `(0.7, 0.1, 0.2)` | Per-user `train : val : test` split |
| `embed_size` / `latent_dim_rec` | `64` | `64` | `64` | Embedding dimension $d$ |
| `n_layers` / `layer_size` | `3` | `3` | `[64, 64, 64]` | Number of GNN propagation layers $L$ |
| `lr` | `0.001` | `0.001` | `0.001` | Adam optimizer learning rate |
| `batch_size` | `2048` | `2048` | `2048` | BPR mini-batch size |
| `reg` / `decay` | `1e-4` | `1e-4` | `1e-4` | $L_2$ regularization coefficient $\lambda_2$ |
| `aug_type` | `"ED"` (`"ND"`, `"RW"`) | — | — | SGL graph augmentation operator |
| `ssl_reg` ($\lambda_1$) | `0.1` | — | — | SGL contrastive loss weight |
| `ssl_temp` ($\tau$) | `0.2` | — | — | InfoNCE softmax temperature |
| `ssl_ratio` ($\rho$) | `0.1` | — | — | SGL edge/node drop probability |
| `node_dropout` / `mess_dropout` | — | — | `0.1` / `[0.1, 0.1, 0.1]` | NGCF Laplacian & message dropout |
| `top_k` | `[20]` | `[20]` | `[20]` | Ranking cutoff $K$ for `Precision`, `Recall`, `NDCG` |

---

## 7. References

1. **SGL**: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. *"Self-Supervised Graph Learning for Recommendation."* **SIGIR 2021**. [[Paper]](https://arxiv.org/abs/2010.10783) [[Official Code: wujcan/SGL-Torch]](https://github.com/wujcan/SGL-Torch)
2. **LightGCN**: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, YongDong Zhang, and Meng Wang. *"LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation."* **SIGIR 2020**. [[Paper]](https://arxiv.org/abs/2002.02126) [[Official Code: gusye1234/LightGCN-PyTorch]](https://github.com/gusye1234/LightGCN-PyTorch)
3. **NGCF**: Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. *"Neural Graph Collaborative Filtering."* **SIGIR 2019**. [[Paper]](https://arxiv.org/abs/1905.08108) [[Official Code: xiangwang1223/neural_graph_collaborative_filtering]](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
