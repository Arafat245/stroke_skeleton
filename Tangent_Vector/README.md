# Tangent-Vector Stroke Prediction

This folder contains the aligned **tangent-vector** pipeline for the
stroke gait dataset. It uses the same `155` subjects and the same `30`
subject-level folds as `../Raw_Skeleton/`, but feeds models the aligned
stroke tangent representation instead of raw marker coordinates.

The two evaluated tasks are:

- **Regression**: `POMA`
- **Classification**: 3-class `LesionLeft`

## Inputs

All tangent experiments read from the aligned repo artifacts:

| File | Shape | Purpose |
|---|---|---|
| `../aligned_data/tangent_vecs200.pkl` | `(32, 3, 200, 155)` | Tangent-vector input |
| `../labels_data/y_poma.txt` | `155` labels | Regression target |
| `../labels_data/pids.txt` | `155` ids | Subject-level CV |
| `../labels_data/demo_data.csv` | metadata table | 3-class lesion labels |

## Cross-validation protocol

All tangent experiments use the same subject-level `30`-fold split
helper in `val_test.py`.

Reported `95%` confidence intervals are subject-level bootstrap
intervals over pooled out-of-fold predictions.

## Files

- `ES-VAE_Reg_Final_(Geodesic_Loss).ipynb` — tangent ES-VAE regression
- `ES-VAE_Clf_Final_(Geodesic_Loss).ipynb` — tangent ES-VAE
  classification
- `PCA_full_aligned.ipynb` — tangent PCA baseline; the README tables
  report the `PCA + k-NN` rows from this notebook
- `baselines/TCN_regclf_tangent.py` — tangent TCN
- `baselines/sequence_regclf_tangent.py` — tangent LSTM / Transformer /
  STGCN
- `val_test.py` — shared subject-fold construction

The adapted graph-model runners are shared from `../official_compare/`.

## How to run

From `Tangent_Vector/`:

```bash
jupyter notebook ES-VAE_Reg_Final_(Geodesic_Loss).ipynb
jupyter notebook ES-VAE_Clf_Final_(Geodesic_Loss).ipynb
jupyter notebook PCA_full_aligned.ipynb
python baselines/TCN_regclf_tangent.py --normalize-input --n-folds 30
python baselines/sequence_regclf_tangent.py --model lstm --normalize-input --n-folds 30
python baselines/sequence_regclf_tangent.py --model transformer --normalize-input --n-folds 30
python baselines/sequence_regclf_tangent.py --model stgcn --normalize-input --n-folds 30
```

From repo root, for the adapted graph baselines:

```bash
python official_compare/hypergcn_runner.py --representation tangent --task regression --epochs 20 --batch-size 64 --reg-calibration linear --output-name hypergcn_tangent_regression_tuned.json
python official_compare/hypergcn_runner.py --representation tangent --task classification --epochs 20 --batch-size 64
python official_compare/sparse_stgcn_runner.py --representation tangent --task regression --epochs 30 --batch-size 32 --lr 0.01 --warmup 5 --reg-balance-mode inverse --reg-calibration linear --output-name sparse_tangent_regression_tuned.json
python official_compare/sparse_stgcn_runner.py --representation tangent --task classification --epochs 20 --batch-size 32
```

## Headline comparison (subject CV, 30 folds)

Pooled out-of-fold metrics, **mean (95% CI)**.

### Regression

| Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
|---|---|---|---|---|
| **ES-VAE + k-NN (proposed)** | **1.25 (0.94, 1.54)** | **2.82 (2.29, 3.21)** | **0.74 (0.66, 0.82)** | **0.86 (0.82, 0.91)** |
| PCA + k-NN | 1.31 (0.99, 1.62) | 3.03 (2.39, 3.48) | 0.70 (0.59, 0.80) | 0.84 (0.78, 0.90) |
| Sparse-ST-GCN | 1.50 (1.12, 1.85) | 3.36 (2.56, 3.88) | 0.63 (0.50, 0.76) | 0.80 (0.72, 0.87) |
| Transformer | 1.60 (1.26, 1.94) | 3.23 (2.52, 3.73) | 0.66 (0.58, 0.75) | 0.81 (0.77, 0.87) |
| LSTM | 1.70 (1.37, 2.01) | 3.22 (2.58, 3.69) | 0.66 (0.57, 0.75) | 0.81 (0.76, 0.87) |
| TCN | 1.74 (1.40, 2.08) | 3.39 (2.79, 3.86) | 0.62 (0.47, 0.75) | 0.83 (0.77, 0.89) |
| Hyper-GCN | 1.79 (1.35, 2.20) | 3.86 (3.09, 4.43) | 0.51 (0.36, 0.65) | 0.74 (0.66, 0.82) |
| STGCN | 2.08 (1.73, 2.40) | 3.51 (2.93, 3.99) | 0.60 (0.46, 0.71) | 0.77 (0.69, 0.84) |

### Classification

| Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
|---|---|---|---|---|
| **ES-VAE + k-NN (proposed)** | **0.92 (0.89, 0.95)** | **0.83 (0.75, 0.90)** | **0.86 (0.79, 0.93)** | **0.80 (0.73, 0.89)** |
| PCA + k-NN | 0.90 (0.86, 0.93) | 0.79 (0.70, 0.87) | 0.83 (0.75, 0.92) | 0.76 (0.68, 0.85) |
| TCN | 0.90 (0.86, 0.94) | 0.75 (0.66, 0.83) | 0.78 (0.68, 0.89) | 0.74 (0.66, 0.83) |
| LSTM | 0.87 (0.83, 0.91) | 0.70 (0.60, 0.79) | 0.81 (0.69, 0.91) | 0.67 (0.59, 0.74) |
| Transformer | 0.84 (0.80, 0.88) | 0.63 (0.51, 0.71) | 0.76 (0.51, 0.87) | 0.60 (0.52, 0.67) |
| Hyper-GCN | 0.83 (0.78, 0.87) | 0.56 (0.50, 0.62) | 0.56 (0.49, 0.63) | 0.57 (0.51, 0.63) |
| STGCN | 0.80 (0.75, 0.85) | 0.48 (0.42, 0.52) | 0.50 (0.44, 0.56) | 0.48 (0.43, 0.53) |
| Sparse-ST-GCN | 0.72 (0.66, 0.77) | 0.28 (0.27, 0.29) | 0.24 (0.22, 0.26) | 0.33 (0.33, 0.33) |

## Tangent vs Raw — same-method comparison

| Method | Tangent MAE | Raw MAE | Δ MAE (Raw - Tangent) | Tangent Macro F1 | Raw Macro F1 | Δ Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| ES-VAE / Vanilla VAE + k-NN | 1.25 | 2.72 | **+1.47** | 0.83 | 0.61 | **+0.22** |
| PCA + k-NN | 1.31 | 2.87 | **+1.56** | 0.79 | 0.55 | **+0.24** |
| Sparse-ST-GCN | 1.50 | 1.70 | +0.20 | 0.28 | 0.22 | +0.06 |
| TCN | 1.74 | 2.66 | +0.92 | 0.75 | 0.64 | +0.11 |
| LSTM | 1.70 | 1.49 | -0.21 | 0.70 | 0.56 | +0.14 |
| Transformer | 1.60 | 2.39 | +0.79 | 0.63 | 0.63 | +0.00 |
| STGCN | 2.08 | 2.18 | +0.10 | 0.48 | 0.39 | +0.09 |
| Hyper-GCN | 1.79 | 1.61 | -0.18 | 0.56 | 0.47 | +0.09 |

## Notes

- The tangent tables intentionally report `PCA + k-NN` rather than the
  best arbitrary classical classifier, so the comparison with
  `ES-VAE + k-NN` is fair.
- `Sparse-ST-GCN` is the stronger imported tangent regressor after the
  light regression-only tuning used here.
- `ES-VAE + k-NN` remains the strongest tangent-side model overall on
  both tasks.
