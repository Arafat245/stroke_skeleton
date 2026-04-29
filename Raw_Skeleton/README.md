# Raw Skeleton Experiments

This folder contains the raw-marker stroke gait experiments. The raw
pipeline keeps the original `32 x 3` marker coordinates and builds
subject-level windows from consecutive gait cycles, then evaluates the
same two downstream tasks used on the tangent side:

- `POMA` regression
- 3-class `LesionLeft` classification

The current raw-side results come from a mix of saved notebooks and one
new official-style adaptation in `../official_compare/`.

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
python official_compare/sparse_stgcn_runner.py --representation raw --task regression --epochs 20 --batch-size 32 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation raw --task classification --epochs 20 --batch-size 32 --device cuda:0
```

Outputs:

- `../official_compare/results/sparse_stgcn_raw_regression.json`
- `../official_compare/results/sparse_stgcn_raw_classification.json`

This runner adapts the official Sparse-ST-GCN backbone to the local
`32`-marker graph and the same `30`-fold subject CV used by the rest of
the stroke repo.

## Regression Comparison

Pooled test metrics, mean `(95% CI)`.

| Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
| --- | --- | --- | --- | --- |
| LSTM raw | **1.493 (1.120, 1.857)** | **3.307 (2.699, 3.789)** | **0.642 (0.523, 0.744)** | **0.806 (0.744, 0.870)** |
| PCA raw (best classical, XGBoost) | 2.138 (1.721, 2.549) | 4.039 (3.347, 4.563) | 0.466 (0.295, 0.606) | 0.694 (0.610, 0.781) |
| STGCN raw | 2.176 (1.800, 2.565) | 3.884 (3.084, 4.499) | 0.506 (0.337, 0.660) | 0.723 (0.635, 0.817) |
| Transformer raw | 2.388 (2.004, 2.771) | 3.925 (3.189, 4.496) | 0.495 (0.294, 0.660) | 0.764 (0.688, 0.844) |
| TCN raw | 2.661 (2.334, 2.983) | 3.825 (3.212, 4.302) | 0.521 (0.301, 0.674) | 0.786 (0.729, 0.849) |
| VAE raw + KNN | 2.720 (2.328, 3.138) | 4.331 (3.767, 4.829) | 0.386 (0.257, 0.480) | 0.624 (0.515, 0.707) |
| Sparse-ST-GCN (official adaptation) | 3.486 (3.048, 3.902) | 5.098 (4.353, 5.671) | 0.149 (0.035, 0.246) | 0.402 (0.307, 0.508) |

## Classification Comparison

Pooled test metrics, mean `(95% CI)`. Macro-F1 is the main score
because the three classes are imbalanced.

| Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
| --- | --- | --- | --- | --- |
| TCN raw | **0.858 (0.814, 0.902)** | **0.637 (0.533, 0.719)** | **0.699 (0.529, 0.865)** | 0.616 (0.537, 0.689) |
| Transformer raw | 0.813 (0.765, 0.861) | 0.626 (0.535, 0.704) | 0.625 (0.536, 0.716) | **0.628 (0.538, 0.715)** |
| PCA raw (best classical, XGBoost) | 0.819 (0.776, 0.866) | 0.582 (0.486, 0.663) | 0.691 (0.502, 0.883) | 0.553 (0.481, 0.620) |
| LSTM raw | 0.813 (0.765, 0.859) | 0.561 (0.479, 0.631) | 0.574 (0.471, 0.674) | 0.558 (0.486, 0.624) |
| STGCN raw | 0.755 (0.701, 0.806) | 0.392 (0.331, 0.442) | 0.477 (0.380, 0.589) | 0.400 (0.357, 0.438) |
| Sparse-ST-GCN (official adaptation) | 0.239 (0.187, 0.287) | 0.218 (0.166, 0.263) | 0.410 (0.356, 0.464) | 0.381 (0.299, 0.453) |

## Notes

- The raw baselines aggregate predictions at the subject level from
  multiple gait windows per subject.
- The saved notebook outputs show that raw LSTM is the strongest
  regression baseline, while raw TCN is the strongest classification
  baseline in this repo.
- The official Sparse-ST-GCN adaptation runs successfully on the raw
  stroke graph, but it does not outperform the older local baselines on
  either task in its current straight port.
