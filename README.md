# MambaIDS

> First application of Mamba Selective State Space Model to IIoT DDoS intrusion detection — a lightweight, edge-deployable binary classifier combining Mamba SSM, CNN projection, and CBAM attention with cross-dataset transfer learning.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Transfer Learning Pipeline](#transfer-learning-pipeline)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)
- [Ablation Study](#ablation-study)
- [Inference Benchmarks](#inference-benchmarks)
- [Citation](#citation)

---

## Overview

Distributed Denial-of-Service (DDoS) attacks on Industrial IoT (IIoT) networks are increasingly common and difficult to detect in real time on constrained edge hardware. Existing approaches rely on Transformers (O(L²) complexity) or GRUs (no content-selectivity), making them impractical at the edge.

**MambaIDS** solves this by using a Mamba Selective State Space Model as the temporal backbone — achieving linear-time sequence modelling with content-selective state updates. The model dynamically decides which network flow features to propagate, making it uniquely suited to bursty, high-volume DDoS traffic where most packets are noise and only a minority signal an active attack.

| Property            | Transformer | GRU    | MambaIDS     |
|---------------------|:-----------:|:------:|:------------:|
| Time complexity     | O(L²)       | O(L)   | O(L)         |
| Long-range memory   | Yes         | No     | Yes          |
| Content-selectivity | No          | No     | Yes          |
| Edge (CPU) speed    | Slow        | Medium | Fast         |
| Prior IIoT DDoS use | Yes         | Yes    | First        |

---

## Key Results

Evaluated over 10 random seeds with 95% confidence intervals and Shapiro-Wilk normality tests:

| Dataset        | F1 (mean ± 95% CI) | AUC (mean ± 95% CI) | FNR (mean ± 95% CI) |
|----------------|:------------------:|:-------------------:|:-------------------:|
| CIC-DDoS2019   | 0.9428 ± 0.0113    | 0.9847 ± 0.0049     | 0.0095 ± 0.0117     |
| Edge-IIoTset   | 0.9602 ± 0.0233    | 0.9972 ± 0.0018     | 0.0629 ± 0.0381     |
| CICIoT23       | 0.9900 ± 0.0007    | 0.9923 ± 0.0007     | 0.0089 ± 0.0097     |

With optimal threshold tuning (Opt_F1): 0.9663, 0.9669, 0.9916 respectively.

---

## Architecture

```
Input (F features)
  └─► ProjectionLayer  (F → PROJ_DIM=39)
        └─► Reshape    (Batch, SEQ_LEN=16, D_MODEL=64)
              └─► MambaBlock x 2       [selective SSM, O(L)]
                    └─► CBAM           [channel + spatial attention]
                          └─► GlobalAvgPool + concat(proj_features)
                                └─► MLP Classifier
                                      └─► Sigmoid  →  0 (benign) / 1 (attack)
```

**Components:**

- **ProjectionLayer** — a 1D CNN that maps arbitrary-width feature vectors to a fixed `PROJ_DIM=39` latent space, enabling transfer across datasets with different input dimensions.
- **MambaBlock** — replaces Transformer and GRU with a selective SSM that maintains linear complexity while capturing long-range temporal dependencies in network flow sequences.
- **CBAM** — Convolutional Block Attention Module applied after Mamba to focus channel and spatial attention on bursty attack patterns.
- **MLP Classifier** — a two-layer head with dropout that concatenates global-pooled Mamba output with the raw projection features before producing a binary output.

Model sizes: CIC-DDoS2019 pre-trained (137,578 params), Edge-IIoTset fine-tuned (92,755 params), CICIoT23 fine-tuned (92,911 params).

---

## Transfer Learning Pipeline

| Stage      | Dataset        | Input Features | Projected To | Trainable Layers              |
|------------|----------------|:--------------:|:------------:|-------------------------------|
| Pre-train  | CIC-DDoS2019   | 78             | 39           | All (full model)              |
| Fine-tune  | Edge-IIoTset   | 33             | 39           | ProjectionLayer + MLP only    |
| Fine-tune  | CICIoT23       | 47             | 39           | ProjectionLayer + MLP only    |

The Mamba blocks and CBAM are frozen after pre-training. Only the input projection and MLP classifier head are updated during fine-tuning, dramatically reducing the number of trainable parameters and preventing catastrophic forgetting.

---

## Datasets

| Dataset        | Samples       | Features       | Attack Types                                                         |
|----------------|:-------------:|:--------------:|----------------------------------------------------------------------|
| CIC-DDoS2019   | 431,371       | 78             | DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SYN, TFTP, UDP, UDPLag       |
| Edge-IIoTset   | 2,377,001     | 33 (selected)  | Multi-attack IIoT/IoT realistic traffic                              |
| CICIoT23       | 7,845,673     | 47             | Modern IoT attack traffic (pre-split train/val/test)                 |

All datasets are sourced from KaggleHub. Update `DATASET_PATHS` in Cell 2 of the notebook to point to your local copies.

---

## Installation

```bash
git clone https://github.com/your-username/MambaIDS.git
cd MambaIDS
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.0.0
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
pyarrow        # for .parquet support
fastparquet    # optional alternative parquet engine
```

No GPU is required. The notebook is fully designed for CPU-only execution.

---

## Usage

### Running the full pipeline

Open `MambaIDS_notebook.ipynb` in Jupyter and run cells in order:

```bash
jupyter notebook MambaIDS_notebook.ipynb
```

### Cell-by-cell summary

| Cell  | Purpose                                                              |
|-------|----------------------------------------------------------------------|
| 0     | Introduction and project summary                                     |
| 1     | Environment setup, global hyperparameters, reproducibility seeding   |
| 2     | Dataset path configuration — update paths here                       |
| 3     | Data loading with chunked ingestion for large CSVs and parquet files |
| 4     | Feature engineering, RobustScaler normalisation, train/val/test split|
| 5     | MambaIDS model definition (ProjectionLayer, MambaBlock, CBAM, MLP)  |
| 6     | Single-run training: pre-train on CIC-DDoS2019, fine-tune on others |
| 7     | Multi-seed statistical evaluation (10 seeds, 95% CI, Shapiro-Wilk)  |
| 8     | Classical ML baselines (Random Forest, XGBoost, Logistic Regression)|
| 9     | Ablation study: Mamba vs GRU vs Transformer vs CNN-only              |
| 10    | Inference latency and throughput benchmark (CPU-only)                |
| 11    | Publication figures: confusion matrices, ROC/PR curves, loss curves  |
| 12    | LaTeX-ready result tables and reproducibility metadata export        |

### Loading a saved model for inference

```python
import torch
from pathlib import Path

model = MambaIDS(n_features=78, proj_dim=39, d_model=64, d_state=16,
                 d_conv=4, expand=2, n_layers=2, dropout=0.0)
model.load_state_dict(torch.load('outputs/mambids_pretrained_cic.pth', map_location='cpu'))
model.eval()

with torch.no_grad():
    logits = model(x_tensor)
    probs  = torch.sigmoid(logits).squeeze(1)
    preds  = (probs >= 0.5).int()
```

---

## Project Structure

```
MambaIDS/
├── MambaIDS_notebook.ipynb   # Main notebook (all cells)
├── outputs/                  # Generated automatically at runtime
│   ├── mambids_pretrained_cic.pth
│   ├── mambids_edge_iot.pth
│   ├── mambids_ciciot23.pth
│   ├── multiseed_cic_ddos.csv
│   ├── multiseed_edge_iot.csv
│   ├── multiseed_ciciot23.csv
│   ├── ablation_results.csv
│   ├── latency_benchmark.csv
│   └── repro_metadata.json
├── requirements.txt
└── README.md
```

---

## Hyperparameters

| Parameter    | Value  | Description                              |
|--------------|:------:|------------------------------------------|
| PROJ_DIM     | 39     | Shared projection dimension              |
| SEQ_LEN      | 16     | Sliding window length                    |
| D_MODEL      | 64     | Mamba model dimension                    |
| D_STATE      | 16     | SSM state dimension                      |
| D_CONV       | 4      | Mamba depthwise conv kernel size         |
| EXPAND       | 2      | Mamba expansion factor                   |
| N_LAYERS     | 2      | Number of stacked Mamba blocks           |
| DROPOUT      | 0.25   | Dropout rate in MLP head                 |
| BATCH_TRAIN  | 256    | Training batch size                      |
| BATCH_INFER  | 512    | Inference batch size                     |
| EPOCHS_PRE   | 25     | Pre-training epochs (CIC-DDoS2019)       |
| EPOCHS_FT    | 20     | Fine-tuning epochs                       |
| LR_PRE       | 1e-3   | Pre-training learning rate               |
| LR_FT        | 3e-4   | Fine-tuning learning rate                |
| PATIENCE     | 7      | Early stopping patience (val F1)         |

Scheduler: OneCycleLR. Optimiser: AdamW with weight decay 1e-4. Loss: BCEWithLogitsLoss with class-weighted positive weight.

---

## Ablation Study

Conducted on CICIoT23 over 3 seeds. The full MambaIDS consistently outperforms all reduced variants:

| Variant                      | Acc    | F1     | FPR    | FNR    |
|------------------------------|:------:|:------:|:------:|:------:|
| Full MambaIDS (proposed)     | best   | best   | lowest | lowest |
| MambaIDS without CBAM        | lower  | lower  | higher | higher |
| MambaIDS with GRU backbone   | lower  | lower  | higher | higher |
| MambaIDS with Transformer    | lower  | lower  | higher | higher |
| CNN projection only          | lowest | lowest | highest| highest|

Full numeric results are saved to `outputs/ablation_results.csv` after running Cell 9.

---

## Inference Benchmarks

Measured on CPU-only hardware (16 threads):

| Dataset        | Avg latency (ms) | P99 latency (ms) | Throughput (pkt/s, single) | Throughput (pkt/s, batch) |
|----------------|:----------------:|:----------------:|:--------------------------:|:-------------------------:|
| CIC-DDoS2019   | 2.30             | 3.09             | 435                        | 1,639                     |
| Edge-IIoTset   | 2.92             | 4.40             | 342                        | 1,588                     |
| CICIoT23       | 3.38             | 7.64             | 296                        | 1,506                     |

Batch inference is recommended for production edge deployments. Results saved to `outputs/latency_benchmark.csv`.

---

## Reproducibility

All experiments use seeds 42–51 (10 seeds). Global seed is applied to Python `random`, NumPy, and PyTorch before each run. Full reproducibility metadata (library versions, hyperparameters, device) is exported to `outputs/repro_metadata.json` at the end of Cell 12.

To exactly reproduce results, ensure:

```python
SEED = 42          # or any seed in 42–51
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

---

## Citation

If you use MambaIDS in your research, please cite:

```bibtex
@misc{mambids2025,
  title   = {MambaIDS: Selective State Space Model for Real-Time DDoS Detection in IIoT Edge Networks},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/your-username/MambaIDS}
}
```

---

## License

MIT License. See `LICENSE` for details.
