# Raw-Skeleton Stroke Prediction

This folder is the official implementation of the **raw-skeleton**
pipeline for the stroke gait dataset. It mirrors `../Tangent_Vector/`
on the same `155` subjects and the same `30` subject-level folds, but
feeds models the original `32 × 3` marker coordinates instead of
aligned tangent vectors.

The two evaluated tasks are:

- **Regression**: `POMA`
- **Classification**: 3-class `LesionLeft`

## Requirements

To install requirements:

```setup
pip install -r ../requirements.txt
```

Python `3.10`+, PyTorch `2.x`, scikit-learn, NumPy, pandas, SciPy,
Jupyter. A CUDA GPU is recommended for the deep baselines.

Inputs (read from the repo root):

- `../data/processed_loaded.pt` — regression subjects
- `../data_clf/processed_loaded.pt` — classification subjects

Each subject contributes:

- `6` gait cycles
- each gait cycle stored as `(100, 96)` = `32 markers × 3`
- subject-level regression and classification labels

All raw experiments use the same subject-level `30`-fold split helper
as the tangent pipeline: `../Tangent_Vector/val_test.py`.

## Training

To train the raw-skeleton models in the paper, from `Raw_Skeleton/`:

```train
jupyter notebook PCA_full_raw_unaligned.ipynb
jupyter notebook VAE_full_raw_unaligned.ipynb
jupyter notebook LSTM_regclf_raw.ipynb
jupyter notebook Transformer_regclf_raw.ipynb
jupyter notebook STGCN.ipynb
python TCN_regclf_raw.py
python vae_knn_raw_matched.py --device cuda:0
```

For the adapted graph baselines (run from repo root):

```train
python official_compare/hypergcn_runner.py --representation raw --task regression --epochs 20 --batch-size 64 --reg-calibration linear --output-name hypergcn_raw_regression_tuned.json
python official_compare/hypergcn_runner.py --representation raw --task classification --epochs 20 --batch-size 64
python official_compare/sparse_stgcn_runner.py --representation raw --task regression --epochs 30 --batch-size 32 --lr 0.01 --warmup 5 --reg-balance-mode inverse --reg-calibration linear --output-name sparse_raw_regression_tuned.json
python official_compare/sparse_stgcn_runner.py --representation raw --task classification --epochs 30 --patience 10 --batch-size 32 --lr 0.01 --label-smoothing 0.0 --no-clf-balancing --warmup 5 --output-name sparse_stgcn_raw_classification_tuned.json
```

## Evaluation

Evaluation is integrated with training: each runner performs the full
subject-level `30`-fold cross-validation and reports pooled
out-of-fold metrics with bootstrap `95%` confidence intervals. The
graph-baseline runners write JSON summaries into
`../official_compare/results/`. Re-running the commands above
re-evaluates the corresponding model.

## Pre-trained Models

Trained checkpoints are not redistributed. All numbers can be
reproduced end-to-end from the training commands above using the
shared `30`-fold subject split in `../Tangent_Vector/val_test.py`.

## Results

Pooled out-of-fold metrics, **mean (95% CI)**.

### Regression

| Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
|---|---|---|---|---|
| **LSTM** | **1.49 (1.12, 1.86)** | **3.31 (2.70, 3.79)** | **0.64 (0.52, 0.74)** | **0.81 (0.74, 0.87)** |
| Hyper-GCN | 1.61 (1.21, 1.98) | 3.57 (2.90, 4.07) | 0.58 (0.48, 0.68) | 0.77 (0.71, 0.83) |
| Sparse-ST-GCN | 1.70 (1.26, 2.10) | 3.83 (3.02, 4.38) | 0.52 (0.37, 0.67) | 0.74 (0.67, 0.83) |
| STGCN | 2.18 (1.80, 2.57) | 3.88 (3.08, 4.50) | 0.51 (0.34, 0.66) | 0.72 (0.64, 0.82) |
| Transformer | 2.39 (2.00, 2.77) | 3.93 (3.19, 4.50) | 0.50 (0.29, 0.66) | 0.76 (0.69, 0.84) |
| TCN | 2.66 (2.33, 2.98) | 3.83 (3.21, 4.30) | 0.52 (0.30, 0.67) | 0.79 (0.73, 0.85) |
| Vanilla VAE + k-NN | 2.72 (2.33, 3.14) | 4.33 (3.77, 4.83) | 0.39 (0.26, 0.48) | 0.62 (0.52, 0.71) |
| PCA + k-NN | 2.87 (2.48, 3.27) | 4.46 (3.91, 4.95) | 0.35 (0.23, 0.44) | 0.59 (0.49, 0.68) |

### Classification

| Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
|---|---|---|---|---|
| **TCN** | **0.86 (0.81, 0.90)** | **0.64 (0.53, 0.72)** | **0.70 (0.53, 0.87)** | 0.62 (0.54, 0.69) |
| Transformer | 0.81 (0.77, 0.86) | 0.63 (0.54, 0.70) | 0.63 (0.54, 0.72) | **0.63 (0.54, 0.72)** |
| LSTM | 0.81 (0.77, 0.86) | 0.56 (0.48, 0.63) | 0.57 (0.47, 0.67) | 0.56 (0.49, 0.62) |
| PCA + k-NN | 0.74 (0.69, 0.80) | 0.55 (0.46, 0.62) | 0.55 (0.46, 0.64) | 0.55 (0.47, 0.63) |
| Sparse-ST-GCN | 0.73 (0.65, 0.79) | 0.48 (0.40, 0.55) | 0.75 (0.65, 0.83) | 0.49 (0.43, 0.56) |
| Hyper-GCN | 0.76 (0.71, 0.81) | 0.47 (0.41, 0.54) | 0.48 (0.41, 0.55) | 0.47 (0.40, 0.54) |
| STGCN | 0.75 (0.70, 0.81) | 0.39 (0.33, 0.44) | 0.48 (0.38, 0.59) | 0.40 (0.36, 0.44) |
| Vanilla VAE + k-NN | 0.81 (0.76, 0.86) | 0.61 (0.51, 0.69) | 0.68 (0.51, 0.85) | 0.58 (0.51, 0.66) |

### Tangent vs Raw — same-method comparison

| Method | Tangent MAE | Raw MAE | Δ MAE (Raw - Tangent) | Tangent Macro F1 | Raw Macro F1 | Δ Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| ES-VAE / Vanilla VAE + k-NN | 1.25 | 2.72 | **+1.47** | 0.83 | 0.61 | **+0.22** |
| PCA + k-NN | 1.31 | 2.87 | **+1.56** | 0.79 | 0.55 | **+0.24** |
| Sparse-ST-GCN | 1.50 | 1.70 | +0.20 | 0.53 | 0.48 | +0.05 |
| TCN | 1.74 | 2.66 | +0.92 | 0.75 | 0.64 | +0.11 |
| LSTM | 1.70 | 1.49 | -0.21 | 0.70 | 0.56 | +0.14 |
| Transformer | 1.60 | 2.39 | +0.79 | 0.63 | 0.63 | +0.00 |
| STGCN | 2.08 | 2.18 | +0.10 | 0.48 | 0.39 | +0.09 |
| Hyper-GCN | 1.79 | 1.61 | -0.18 | 0.56 | 0.47 | +0.09 |

### Notes

- The raw tables intentionally report `PCA + k-NN` rather than the
  best arbitrary classical model, so the comparison with
  `ES-VAE + k-NN` is fair.
- The raw `Vanilla VAE + k-NN` row comes from
  `VAE_full_raw_unaligned.ipynb` and is included in both regression
  and classification tables.
- After light task-specific tuning, `Sparse-ST-GCN` becomes a much
  more reasonable raw classification baseline, improving from a
  collapsed `0.22` Macro F1 run to `0.48`.

## Files

- `TCN_regclf_raw.py` / `TCN_regclf_raw.ipynb` — raw TCN
- `LSTM_regclf_raw.ipynb` — raw LSTM baseline
- `Transformer_regclf_raw.ipynb` — raw Transformer baseline
- `STGCN.ipynb` — raw STGCN baseline
- `PCA_full_raw_unaligned.ipynb` — raw `PCA + k-NN`
- `VAE_full_raw_unaligned.ipynb` — raw `Vanilla VAE + k-NN`
- `vae_knn_raw_matched.py` — auxiliary raw VAE runner matched to the
  no-alignment ablation setup
- `data_utils_load.py` — subject loading, train-fold standardization,
  gait batching
- `val_test.py` — shared subject-fold construction

## Contributing

Released under the **MIT License** for academic and research use.
Contributions welcome via pull request; please file an issue first for
substantial changes. New methods should follow the same subject-level
`30`-fold protocol so results remain comparable.
