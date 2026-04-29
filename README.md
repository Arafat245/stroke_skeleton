# Stroke Gait Analysis — Tangent Vectors vs Raw Skeletons

This repo studies stroke gait analysis with two skeleton
representations:

- **Tangent vectors** on aligned manifold trajectories
- **Raw skeleton / marker coordinates**

Two prediction tasks are evaluated:

- **Regression**: `POMA` score
- **Classification**: 3-class `LesionLeft`

The current dataset has:

- `155` subjects
- subject-level `30`-fold CV from [Tangent_Vector/val_test.py](./Tangent_Vector/val_test.py)
- tangent tensors of shape `(32, 3, 200, 155)`
- raw subject data stored as `6` gait cycles per subject, each
  `(100, 96)` = `32 markers x 3`

## Repository Layout

```text
stroke_riemann/
├── README.md
├── official_compare/                official-style graph model adapters
│   ├── common.py
│   ├── sparse_stgcn_runner.py
│   └── results/
├── Tangent_Vector/                  aligned tangent-vector pipeline
│   ├── README.md
│   ├── ES-VAE_*                     tangent ES-VAE notebooks
│   ├── PCA_full_aligned.ipynb
│   └── baselines/                   TCN / LSTM / Transformer / STGCN
├── Raw_Skeleton/                    raw-marker pipeline
│   ├── README.md
│   ├── TCN_regclf_raw.ipynb
│   ├── LSTM_regclf_raw.ipynb
│   ├── Transformer_regclf_raw.ipynb
│   ├── STGCN.ipynb
│   └── PCA_full_raw_unaligned.ipynb
├── aligned_data/                    tangent vectors and aligned curves
├── data/                            processed raw regression subjects
├── data_clf/                        processed raw classification subjects
└── labels_data/                     POMA, subject ids, and lesion labels
```

## Tangent vs Raw Summary

### Regression

| Representation | Best Method | MAE | RMSE | R2 | Pearson r |
| --- | --- | ---: | ---: | ---: | ---: |
| Tangent vector | ES-VAE Geodesic | **1.246** | **2.818** | **0.740** | **0.862** |
| Raw skeleton | LSTM raw | 1.493 | 3.307 | 0.642 | 0.806 |

### Classification

| Representation | Best Method | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Tangent vector | ES-VAE Geodesic | **0.916** | **0.825** | **0.856** | **0.805** |
| Raw skeleton | TCN raw | 0.858 | 0.637 | 0.699 | 0.616 |

Tangent vectors remain the strongest representation overall in this
repo, especially for the imbalanced lesion classification task.

## Same-Model Tangent vs Raw

Where the same family is available on both sides:

| Model | Tangent Regression MAE | Raw Regression MAE | Tangent Clf Macro F1 | Raw Clf Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| TCN | **1.743** | 2.661 | **0.752** | 0.637 |
| LSTM | 1.699 | **1.493** | **0.699** | 0.561 |
| Transformer | **1.602** | 2.388 | 0.626 | 0.626 |
| STGCN | **2.081** | 2.176 | **0.476** | 0.392 |
| Sparse-ST-GCN (official adaptation) | **2.497** | 3.486 | **0.278** | 0.218 |

The tangent representation wins on most matched comparisons. The one
clear raw-side exception is LSTM regression.

## Official Graph Model Adaptation

The repo now includes an official-style Sparse-ST-GCN port in
[official_compare/sparse_stgcn_runner.py](./official_compare/sparse_stgcn_runner.py).

Commands:

```bash
python official_compare/sparse_stgcn_runner.py --representation tangent --task regression --epochs 20 --batch-size 32 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation tangent --task classification --epochs 20 --batch-size 32 --device cuda:0
python official_compare/sparse_stgcn_runner.py --representation raw --task regression --epochs 20 --batch-size 32 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation raw --task classification --epochs 20 --batch-size 32 --device cuda:0
```

Current Sparse-ST-GCN result:

| Representation | Task | Headline Result |
| --- | --- | --- |
| Tangent | Regression | MAE `2.497`, RMSE `5.066`, R2 `0.160`, Pearson `0.497` |
| Tangent | Classification | Accuracy `0.716`, Macro-F1 `0.278` |
| Raw | Regression | MAE `3.486`, RMSE `5.098`, R2 `0.149`, Pearson `0.402` |
| Raw | Classification | Accuracy `0.239`, Macro-F1 `0.218` |

The code transfer is successful, but this straight Sparse-ST-GCN port is
not competitive with the older local stroke baselines in its current
form.

## Readmes

- [Tangent_Vector/README.md](./Tangent_Vector/README.md)
- [Raw_Skeleton/README.md](./Raw_Skeleton/README.md)

These contain task-specific commands, saved result sources, and the
full comparison tables for each representation.
