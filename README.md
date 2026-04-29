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
| Tangent vector | ES-VAE Geodesic | **1.246 (0.941, 1.537)** | **2.818 (2.292, 3.212)** | **0.740 (0.659, 0.817)** | **0.862 (0.823, 0.906)** |
| Raw skeleton | LSTM raw | 1.493 (1.120, 1.857) | 3.307 (2.699, 3.789) | 0.642 (0.523, 0.744) | 0.806 (0.744, 0.870) |

### Classification

| Representation | Best Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Tangent vector | ES-VAE Geodesic | **0.916 (0.885, 0.949)** | **0.825 (0.754, 0.897)** | **0.856 (0.790, 0.931)** | **0.805 (0.728, 0.886)** |
| Raw skeleton | TCN raw | 0.858 (0.814, 0.902) | 0.637 (0.533, 0.719) | 0.699 (0.529, 0.865) | 0.616 (0.537, 0.689) |

Tangent vectors remain the strongest representation overall in this
repo, especially for the imbalanced lesion classification task.

## Same-Model Tangent vs Raw

Where the same family is available on both sides:

| Model | Tangent Regression MAE (95% CI) | Raw Regression MAE (95% CI) | Tangent Clf Macro F1 (95% CI) | Raw Clf Macro F1 (95% CI) |
| --- | --- | --- | --- | --- |
| TCN | **1.743 (1.402, 2.082)** | 2.661 (2.334, 2.983) | **0.752 (0.660, 0.834)** | 0.637 (0.533, 0.719) |
| LSTM | 1.699 (1.368, 2.014) | **1.493 (1.120, 1.857)** | **0.699 (0.598, 0.787)** | 0.561 (0.479, 0.631) |
| Transformer | **1.602 (1.257, 1.936)** | 2.388 (2.004, 2.771) | 0.626 (0.513, 0.711) | 0.626 (0.535, 0.704) |
| STGCN | **2.081 (1.733, 2.403)** | 2.176 (1.800, 2.565) | **0.476 (0.421, 0.523)** | 0.392 (0.331, 0.442) |
| Sparse-ST-GCN (official adaptation) | **2.497 (1.947, 2.992)** | 3.486 (3.048, 3.902) | **0.278 (0.266, 0.290)** | 0.218 (0.166, 0.263) |

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
| Tangent | Regression | MAE `2.497 (1.947, 2.992)`, RMSE `5.066 (4.200, 5.731)`, R2 `0.160 (-0.027, 0.328)`, Pearson `0.497 (0.331, 0.627)` |
| Tangent | Classification | Accuracy `0.716 (0.663, 0.771)`, Macro-F1 `0.278 (0.266, 0.290)` |
| Raw | Regression | MAE `3.486 (3.048, 3.902)`, RMSE `5.098 (4.353, 5.671)`, R2 `0.149 (0.035, 0.246)`, Pearson `0.402 (0.307, 0.508)` |
| Raw | Classification | Accuracy `0.239 (0.187, 0.287)`, Macro-F1 `0.218 (0.166, 0.263)` |

The code transfer is successful, but this straight Sparse-ST-GCN port is
not competitive with the older local stroke baselines in its current
form.

## Readmes

- [Tangent_Vector/README.md](./Tangent_Vector/README.md)
- [Raw_Skeleton/README.md](./Raw_Skeleton/README.md)

These contain task-specific commands, saved result sources, and the
full comparison tables for each representation.
