# Raw Skeleton Experiments

This folder contains the raw-marker stroke gait experiments. The raw
pipeline keeps the original `32 x 3` marker coordinates and builds
subject-level windows from consecutive gait cycles, then evaluates the
same two downstream tasks used on the tangent side:

- `POMA` regression
- 3-class `LesionLeft` classification

The current raw-side results come from a mix of saved notebooks and two
official-style adaptations in `../official_compare/`.

## Inputs

The raw scripts use the processed gait-cycle tensors saved under the repo
root:

- `data/processed_loaded.pt` for `POMA` regression
- `data_clf/processed_loaded.pt` for classification

Each subject entry stores:

- `6` gait cycles
- each gait cycle as `(100, 96)` = `100` frames of `32 markers x 3`
- subject-level labels

The classification task uses the same 3-class label already present in
the saved baselines:

- `0 = LesionRight`
- `1 = LesionLeft`
- `2 = Healthy`

## Main Files

- `TCN_regclf_raw.ipynb` and `TCN_regclf_raw.py`
  - Raw TCN regression and classification.
- `LSTM_regclf_raw.ipynb`
  - Raw LSTM regression and classification.
- `Transformer_regclf_raw.ipynb`
  - Raw Transformer regression and classification.
- `STGCN.ipynb`
  - Raw STGCN regression and classification.
- `PCA_full_raw_unaligned.ipynb`
  - Classical raw baseline on flattened gait data.
- `VAE_full_raw_unaligned.ipynb`
  - Raw VAE latent baseline with KNN downstream evaluation.
- `data_utils_load.py`
  - Subject split loading, train-fold standardization, and gait batching.
- `val_test.py`
  - Shared `30`-fold subject-level split helper.

## Official Adaptation

Run from repo root:

```bash
python official_compare/hypergcn_runner.py --representation raw --task regression --epochs 20 --batch-size 64 --device cuda:1
python official_compare/hypergcn_runner.py --representation raw --task classification --epochs 20 --batch-size 64 --device cuda:0
python official_compare/sparse_stgcn_runner.py --representation raw --task regression --epochs 20 --batch-size 32 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation raw --task classification --epochs 20 --batch-size 32 --device cuda:0
```

Outputs:

- `../official_compare/results/hypergcn_raw_regression.json`
- `../official_compare/results/hypergcn_raw_classification.json`
- `../official_compare/results/sparse_stgcn_raw_regression.json`
- `../official_compare/results/sparse_stgcn_raw_classification.json`

These runners adapt the official Hyper-GCN and Sparse-ST-GCN backbones
to the local `32`-marker graph and the same `30`-fold subject CV used
by the rest of the stroke repo.

## Regression Comparison

Pooled test metrics, mean `(95% CI)`.

| Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
| --- | --- | --- | --- | --- |
| LSTM raw | **1.49 (1.12, 1.86)** | **3.31 (2.70, 3.79)** | **0.64 (0.52, 0.74)** | **0.81 (0.74, 0.87)** |
| PCA raw (best classical, XGBoost) | 2.14 (1.72, 2.55) | 4.04 (3.35, 4.56) | 0.47 (0.30, 0.61) | 0.69 (0.61, 0.78) |
| STGCN raw | 2.18 (1.80, 2.57) | 3.88 (3.08, 4.50) | 0.51 (0.34, 0.66) | 0.72 (0.64, 0.82) |
| Transformer raw | 2.39 (2.00, 2.77) | 3.93 (3.19, 4.50) | 0.50 (0.29, 0.66) | 0.76 (0.69, 0.84) |
| TCN raw | 2.66 (2.33, 2.98) | 3.83 (3.21, 4.30) | 0.52 (0.30, 0.67) | 0.79 (0.73, 0.85) |
| VAE raw + KNN | 2.72 (2.33, 3.14) | 4.33 (3.77, 4.83) | 0.39 (0.26, 0.48) | 0.62 (0.52, 0.71) |
| Hyper-GCN (official adaptation) | 3.06 (2.34, 3.72) | 6.51 (5.45, 7.31) | -0.39 (-0.84, -0.06) | 0.37 (0.23, 0.51) |
| Sparse-ST-GCN (official adaptation) | 3.49 (3.05, 3.90) | 5.10 (4.35, 5.67) | 0.15 (0.04, 0.25) | 0.40 (0.31, 0.51) |

## Classification Comparison

Pooled test metrics, mean `(95% CI)`. Macro-F1 is the main score
because the three classes are imbalanced.

| Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
| --- | --- | --- | --- | --- |
| TCN raw | **0.86 (0.81, 0.90)** | **0.64 (0.53, 0.72)** | **0.70 (0.53, 0.87)** | 0.62 (0.54, 0.69) |
| Transformer raw | 0.81 (0.77, 0.86) | 0.63 (0.54, 0.70) | 0.63 (0.54, 0.72) | **0.63 (0.54, 0.72)** |
| PCA raw (best classical, XGBoost) | 0.82 (0.78, 0.87) | 0.58 (0.49, 0.66) | 0.69 (0.50, 0.88) | 0.55 (0.48, 0.62) |
| LSTM raw | 0.81 (0.77, 0.86) | 0.56 (0.48, 0.63) | 0.57 (0.47, 0.67) | 0.56 (0.49, 0.62) |
| Hyper-GCN (official adaptation) | 0.76 (0.71, 0.81) | 0.47 (0.41, 0.54) | 0.48 (0.41, 0.55) | 0.47 (0.40, 0.54) |
| STGCN raw | 0.75 (0.70, 0.81) | 0.39 (0.33, 0.44) | 0.48 (0.38, 0.59) | 0.40 (0.36, 0.44) |
| Sparse-ST-GCN (official adaptation) | 0.24 (0.19, 0.29) | 0.22 (0.17, 0.26) | 0.41 (0.36, 0.46) | 0.38 (0.30, 0.45) |

## Notes

- The raw baselines aggregate predictions at the subject level from
  multiple gait windows per subject.
- The saved notebook outputs show that raw LSTM is the strongest
  regression baseline, while raw TCN is the strongest classification
  baseline in this repo.
- Hyper-GCN is the stronger imported raw baseline, particularly for
  classification, but it still does not beat the best local raw models.
- The official Sparse-ST-GCN adaptation runs successfully on the raw
  stroke graph, but it does not outperform the older local baselines on
  either task in its current straight port.
