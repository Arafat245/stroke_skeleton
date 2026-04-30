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

Pooled out-of-fold metrics, mean `(95% CI)`.

| Representation | Best Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Tangent vector | ES-VAE Geodesic | **1.25 (0.94, 1.54)** | **2.82 (2.29, 3.21)** | **0.74 (0.66, 0.82)** | **0.86 (0.82, 0.91)** |
| Raw skeleton | LSTM raw | 1.49 (1.12, 1.86) | 3.31 (2.70, 3.79) | 0.64 (0.52, 0.74) | 0.81 (0.74, 0.87) |

### Classification

| Representation | Best Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Tangent vector | ES-VAE Geodesic | **0.92 (0.89, 0.95)** | **0.83 (0.75, 0.90)** | **0.86 (0.79, 0.93)** | **0.80 (0.73, 0.89)** |
| Raw skeleton | TCN raw | 0.86 (0.81, 0.90) | 0.64 (0.53, 0.72) | 0.70 (0.53, 0.87) | 0.62 (0.54, 0.69) |

Tangent vectors remain the strongest representation overall in this
repo, especially for the imbalanced lesion classification task.

## Same-Model Tangent vs Raw

Where the same family is available on both sides:

| Model | Tangent Regression MAE (95% CI) | Raw Regression MAE (95% CI) | Tangent Clf Macro F1 (95% CI) | Raw Clf Macro F1 (95% CI) |
| --- | --- | --- | --- | --- |
| TCN | **1.74 (1.40, 2.08)** | 2.66 (2.33, 2.98) | **0.75 (0.66, 0.83)** | 0.64 (0.53, 0.72) |
| LSTM | 1.70 (1.37, 2.01) | **1.49 (1.12, 1.86)** | **0.70 (0.60, 0.79)** | 0.56 (0.48, 0.63) |
| Transformer | **1.60 (1.26, 1.94)** | 2.39 (2.00, 2.77) | 0.63 (0.51, 0.71) | 0.63 (0.54, 0.70) |
| STGCN | **2.08 (1.73, 2.40)** | 2.18 (1.80, 2.57) | **0.48 (0.42, 0.52)** | 0.39 (0.33, 0.44) |
| Sparse-ST-GCN (official adaptation) | **2.50 (1.95, 2.99)** | 3.49 (3.05, 3.90) | **0.28 (0.27, 0.29)** | 0.22 (0.17, 0.26) |

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

Current Sparse-ST-GCN result, mean `(95% CI)`:

| Representation | Task | Headline Result |
| --- | --- | --- |
| Tangent | Regression | MAE `2.50 (1.95, 2.99)`, RMSE `5.07 (4.20, 5.73)`, R2 `0.16 (-0.03, 0.33)`, Pearson `0.50 (0.33, 0.63)` |
| Tangent | Classification | Accuracy `0.72 (0.66, 0.77)`, Macro-F1 `0.28 (0.27, 0.29)` |
| Raw | Regression | MAE `3.49 (3.05, 3.90)`, RMSE `5.10 (4.35, 5.67)`, R2 `0.15 (0.04, 0.25)`, Pearson `0.40 (0.31, 0.51)` |
| Raw | Classification | Accuracy `0.24 (0.19, 0.29)`, Macro-F1 `0.22 (0.17, 0.26)` |

The code transfer is successful, but this straight Sparse-ST-GCN port is
not competitive with the older local stroke baselines in its current
form.

## Readmes

- [Tangent_Vector/README.md](./Tangent_Vector/README.md)
- [Raw_Skeleton/README.md](./Raw_Skeleton/README.md)

These contain task-specific commands, saved result sources, and the
full comparison tables for each representation.
