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

Test metrics pooled across folds.

| Method | MAE | RMSE | R2 | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| LSTM raw | **1.493** | **3.307** | **0.642** | **0.806** |
| PCA raw (best classical, XGBoost) | 2.138 | 4.039 | 0.466 | 0.694 |
| STGCN raw | 2.176 | 3.884 | 0.506 | 0.723 |
| Transformer raw | 2.388 | 3.925 | 0.495 | 0.764 |
| TCN raw | 2.661 | 3.825 | 0.521 | 0.786 |
| VAE raw + KNN | 2.720 | 4.331 | 0.386 | 0.624 |
| Sparse-ST-GCN (official adaptation) | 3.486 | 5.098 | 0.149 | 0.402 |

## Classification Comparison

Test metrics pooled across folds. Macro-F1 is the main score because the
three classes are imbalanced.

| Method | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| --- | ---: | ---: | ---: | ---: |
| TCN raw | **0.858** | **0.637** | **0.699** | 0.616 |
| Transformer raw | 0.813 | 0.626 | 0.625 | **0.628** |
| PCA raw (best classical, XGBoost) | 0.820 | 0.580 | 0.690 | 0.550 |
| LSTM raw | 0.813 | 0.561 | 0.574 | 0.558 |
| STGCN raw | 0.755 | 0.392 | 0.477 | 0.400 |
| Sparse-ST-GCN (official adaptation) | 0.239 | 0.218 | 0.410 | 0.381 |

## Notes

- The raw baselines aggregate predictions at the subject level from
  multiple gait windows per subject.
- The saved notebook outputs show that raw LSTM is the strongest
  regression baseline, while raw TCN is the strongest classification
  baseline in this repo.
- The official Sparse-ST-GCN adaptation runs successfully on the raw
  stroke graph, but it does not outperform the older local baselines on
  either task in its current straight port.
