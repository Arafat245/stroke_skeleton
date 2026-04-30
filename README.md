# Stroke Gait Analysis — Tangent-Vector vs Raw-Skeleton

A controlled comparison of **aligned tangent-vector gait
representations** against **raw 3D marker coordinates** for stroke gait
prediction. We evaluate two tasks on the same `155` subjects under the
same subject-level `30`-fold cross-validation:

- **Regression**: `POMA`
- **Classification**: 3-class `LesionLeft`

## What's the manifold prior?

Each subject's gait trajectory can be represented either as:

- **Raw skeleton / marker coordinates** from consecutive gait cycles
- **Aligned tangent vectors** derived from the stroke gait manifold
  pipeline

The hypothesis is the same as in the activity-recognition repo: the
aligned tangent representation removes nuisance variation and gives the
downstream model cleaner geometry than raw coordinates.

## Dataset

| Item | Value |
|---|---|
| Subjects | `155` |
| Tangent tensor | `(32, 3, 200, 155)` |
| Raw subject data | `6` gait cycles per subject |
| Raw gait shape | `(100, 96)` = `32 markers x 3` |
| CV protocol | subject-level `30`-fold CV from `Tangent_Vector/val_test.py` |
| Regression target | `POMA` |
| Classification target | 3-class `LesionLeft` (`LesionRight`, `LesionLeft`, `Healthy`) |

## Repository layout

```text
stroke_riemann/
├── README.md
├── official_compare/             adapted official Hyper-GCN / Sparse-ST-GCN runners
│   ├── common.py                 shared stroke loaders, graph utils, metrics
│   ├── hypergcn_runner.py        Hyper-GCN runner for raw / tangent, regression / classification
│   ├── sparse_stgcn_runner.py    Sparse-ST-GCN runner for raw / tangent, regression / classification
│   └── results/                  saved JSON summaries for graph-model runs
├── Tangent_Vector/               tangent-vector pipeline
│   ├── README.md
│   ├── ES-VAE_*                  tangent ES-VAE notebooks
│   ├── PCA_full_aligned.ipynb    PCA + KNN tangent baseline
│   └── baselines/                TCN / LSTM / Transformer / STGCN
├── Raw_Skeleton/                 raw-marker pipeline
│   ├── README.md
│   ├── PCA_full_raw_unaligned.ipynb
│   ├── VAE_full_raw_unaligned.ipynb
│   ├── vae_knn_raw_matched.py    matched no-alignment raw VAE + k-NN runner
│   ├── TCN_regclf_raw.py
│   └── data_utils_load.py
├── aligned_data/                 tangent vectors and aligned curves
├── data/                         processed raw regression subjects
├── data_clf/                     processed raw classification subjects
└── labels_data/                  POMA, participant ids, lesion labels
```

## Pipeline

1. **Load / align** the stroke gait data into either tangent-vector form
   (`aligned_data/tangent_vecs200.pkl`) or raw gait-cycle windows.
2. **Train** tangent and raw baselines under the same `30` subject-level
   folds.
3. **Evaluate** pooled out-of-fold regression and classification
   metrics with bootstrap `95%` confidence intervals.
4. **Compare** matched model families where only the input
   representation changes.

## Headline results

Pooled out-of-fold metrics, **mean (95% CI)** from subject-level
bootstrap.

### Regression (30-fold subject CV)

| Input Representation | Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
|---|---|---|---|---|---|
| Tangent Vector | **ES-VAE + k-NN (proposed)** | **1.25 (0.94, 1.54)** | **2.82 (2.29, 3.21)** | **0.74 (0.66, 0.82)** | **0.86 (0.82, 0.91)** |
|  | PCA + k-NN | 1.31 (0.99, 1.62) | 3.03 (2.39, 3.48) | 0.70 (0.59, 0.80) | 0.84 (0.78, 0.90) |
|  | Sparse-ST-GCN | 1.50 (1.12, 1.85) | 3.36 (2.56, 3.88) | 0.63 (0.50, 0.76) | 0.80 (0.72, 0.87) |
|  | Transformer | 1.60 (1.26, 1.94) | 3.23 (2.52, 3.73) | 0.66 (0.58, 0.75) | 0.81 (0.77, 0.87) |
|  | LSTM | 1.70 (1.37, 2.01) | 3.22 (2.58, 3.69) | 0.66 (0.57, 0.75) | 0.81 (0.76, 0.87) |
|  | TCN | 1.74 (1.40, 2.08) | 3.39 (2.79, 3.86) | 0.62 (0.47, 0.75) | 0.83 (0.77, 0.89) |
|  | Hyper-GCN | 1.79 (1.35, 2.20) | 3.86 (3.09, 4.43) | 0.51 (0.36, 0.65) | 0.74 (0.66, 0.82) |
|  | STGCN | 2.08 (1.73, 2.40) | 3.51 (2.93, 3.99) | 0.60 (0.46, 0.71) | 0.77 (0.69, 0.84) |
| Raw Skeleton | **LSTM** | **1.49 (1.12, 1.86)** | **3.31 (2.70, 3.79)** | **0.64 (0.52, 0.74)** | **0.81 (0.74, 0.87)** |
|  | Hyper-GCN | 1.61 (1.21, 1.98) | 3.57 (2.90, 4.07) | 0.58 (0.48, 0.68) | 0.77 (0.71, 0.83) |
|  | Sparse-ST-GCN | 1.70 (1.26, 2.10) | 3.83 (3.02, 4.38) | 0.52 (0.37, 0.67) | 0.74 (0.67, 0.83) |
|  | STGCN | 2.18 (1.80, 2.57) | 3.88 (3.08, 4.50) | 0.51 (0.34, 0.66) | 0.72 (0.64, 0.82) |
|  | Transformer | 2.39 (2.00, 2.77) | 3.93 (3.19, 4.50) | 0.50 (0.29, 0.66) | 0.76 (0.69, 0.84) |
|  | TCN | 2.66 (2.33, 2.98) | 3.83 (3.21, 4.30) | 0.52 (0.30, 0.67) | 0.79 (0.73, 0.85) |
|  | Vanilla VAE + k-NN | 2.72 (2.33, 3.14) | 4.33 (3.77, 4.83) | 0.39 (0.26, 0.48) | 0.62 (0.52, 0.71) |
|  | PCA + k-NN | 2.87 (2.48, 3.27) | 4.46 (3.91, 4.95) | 0.35 (0.23, 0.44) | 0.59 (0.49, 0.68) |

### Classification (30-fold subject CV)

| Input Representation | Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
|---|---|---|---|---|---|
| Tangent Vector | **ES-VAE + k-NN (proposed)** | **0.92 (0.89, 0.95)** | **0.83 (0.75, 0.90)** | **0.86 (0.79, 0.93)** | **0.80 (0.73, 0.89)** |
|  | PCA + k-NN | 0.90 (0.86, 0.93) | 0.79 (0.70, 0.87) | 0.83 (0.75, 0.92) | 0.76 (0.68, 0.85) |
|  | TCN | 0.90 (0.86, 0.94) | 0.75 (0.66, 0.83) | 0.78 (0.68, 0.89) | 0.74 (0.66, 0.83) |
|  | LSTM | 0.87 (0.83, 0.91) | 0.70 (0.60, 0.79) | 0.81 (0.69, 0.91) | 0.67 (0.59, 0.74) |
|  | Transformer | 0.84 (0.80, 0.88) | 0.63 (0.51, 0.71) | 0.76 (0.51, 0.87) | 0.60 (0.52, 0.67) |
|  | Hyper-GCN | 0.83 (0.78, 0.87) | 0.56 (0.50, 0.62) | 0.56 (0.49, 0.63) | 0.57 (0.51, 0.63) |
|  | Sparse-ST-GCN | 0.79 (0.72, 0.86) | 0.53 (0.43, 0.62) | 0.73 (0.62, 0.85) | 0.49 (0.44, 0.55) |
|  | STGCN | 0.80 (0.75, 0.85) | 0.48 (0.42, 0.52) | 0.50 (0.44, 0.56) | 0.48 (0.43, 0.53) |
| Raw Skeleton | **TCN** | **0.86 (0.81, 0.90)** | **0.64 (0.53, 0.72)** | **0.70 (0.53, 0.87)** | 0.62 (0.54, 0.69) |
|  | Transformer | 0.81 (0.77, 0.86) | 0.63 (0.54, 0.70) | 0.63 (0.54, 0.72) | **0.63 (0.54, 0.72)** |
|  | LSTM | 0.81 (0.77, 0.86) | 0.56 (0.48, 0.63) | 0.57 (0.47, 0.67) | 0.56 (0.49, 0.62) |
|  | PCA + k-NN | 0.74 (0.69, 0.80) | 0.55 (0.46, 0.62) | 0.55 (0.46, 0.64) | 0.55 (0.47, 0.63) |
|  | Sparse-ST-GCN | 0.73 (0.65, 0.79) | 0.48 (0.40, 0.55) | 0.75 (0.65, 0.83) | 0.49 (0.43, 0.56) |
|  | Hyper-GCN | 0.76 (0.71, 0.81) | 0.47 (0.41, 0.54) | 0.48 (0.41, 0.55) | 0.47 (0.40, 0.54) |
|  | STGCN | 0.75 (0.70, 0.81) | 0.39 (0.33, 0.44) | 0.48 (0.38, 0.59) | 0.40 (0.36, 0.44) |
|  | Vanilla VAE + k-NN | 0.81 (0.76, 0.86) | 0.61 (0.51, 0.69) | 0.68 (0.51, 0.85) | 0.58 (0.51, 0.66) |

### Tangent - Raw gaps under subject CV

`Δ MAE = Raw - Tangent`, so a positive value means tangent is better.
`Δ Macro F1 = Tangent - Raw`, so a positive value means tangent is
better.

| Method pair | Tangent MAE | Raw MAE | Δ MAE | Tangent Macro F1 | Raw Macro F1 | Δ Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| ES-VAE / Vanilla VAE + k-NN | 1.25 | 2.72 | **+1.47** | 0.83 | 0.61 | **+0.22** |
| PCA + k-NN | 1.31 | 2.87 | **+1.56** | 0.79 | 0.55 | **+0.24** |
| Sparse-ST-GCN | 1.50 | 1.70 | +0.20 | 0.53 | 0.48 | +0.05 |
| TCN | 1.74 | 2.66 | +0.92 | 0.75 | 0.64 | +0.11 |
| LSTM | 1.70 | 1.49 | -0.21 | 0.70 | 0.56 | +0.14 |
| Transformer | 1.60 | 2.39 | +0.79 | 0.63 | 0.63 | +0.00 |
| STGCN | 2.08 | 2.18 | +0.10 | 0.48 | 0.39 | +0.09 |
| Hyper-GCN | 1.79 | 1.61 | -0.18 | 0.56 | 0.47 | +0.09 |

### Key findings

1. **Tangent ES-VAE is the best overall model on both tasks.** It is
   the strongest regressor (`MAE 1.25`, `R2 0.74`) and the strongest
   classifier (`Macro F1 0.83`).
2. **`PCA + k-NN` is the fair classical comparator** to ES-VAE on both
   tangent and raw inputs. On this dataset, the tangent PCA baseline is
   much stronger than the raw PCA baseline for both tasks.
3. **Tangent wins the classification comparison for every matched model
   family.** The gains are largest for `PCA + k-NN`, `Vanilla VAE /
   ES-VAE`, and the recurrent baselines.
4. **Regression is still somewhat mixed outside the VAE family.**
   Tangent wins for most families, but raw `LSTM` and raw `Hyper-GCN`
   outperform their tangent counterparts.
5. **The adapted official graph baselines become much more usable after
   light task-specific tuning.** `Sparse-ST-GCN` improves sharply on
   classification once the learning rate, balancing, smoothing, and
   warmup are adjusted, while still remaining below the top ES-VAE /
   PCA / TCN tangent baselines.

## Reproduce

See the folder-specific READMEs for commands and saved result files:

- [Tangent_Vector/README.md](./Tangent_Vector/README.md)
- [Raw_Skeleton/README.md](./Raw_Skeleton/README.md)

The adapted graph baselines live in `official_compare/` and write JSON
summaries into `official_compare/results/`.
