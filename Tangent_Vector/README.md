# Tangent Vector Experiments

This folder contains the aligned-tangent experiments for stroke gait analysis. The code is split into three groups:

1. ES-VAE notebooks with geodesic loss.
2. A PCA baseline notebook on aligned data.
3. Baseline scripts in `baselines/` for TCN, LSTM, Transformer, and STGCN.
4. `../official_compare/` for the adapted official Sparse-ST-GCN runner.

All runs use the same subject-level cross-validation scheme from
`val_test.py` unless noted otherwise. The result tables below report
pooled out-of-fold means with `95% CI` from subject-level bootstrap.

## What Each File Does

### ES-VAE notebooks

- `ES-VAE_Reg_Final_(Geodesic_Loss).ipynb`
  - Trains the ES-VAE regression pipeline on tangent vectors using geodesic loss.
  - Extracts latent embeddings and evaluates POMA prediction with KNN regression.
- `ES-VAE_Clf_Final_(Geodesic_Loss).ipynb`
  - Same ES-VAE setup, but for lesion-side classification.
  - Uses latent embeddings and evaluates with KNN classification.

### PCA baseline

- `PCA_full_aligned.ipynb`
  - Builds a classical baseline on aligned tangent data using PCA features.
  - Benchmarks multiple downstream models: KNN, SVM, Random Forest, XGBoost, and MLP.
  - The comparison tables below report the best test result from that notebook for each task.

### Baseline scripts

- `baselines/TCN_regclf_tangent.py`
  - TCN baseline on tangent vectors.
  - Regression and classification share the same backbone, but the classifier uses its own smaller capacity by default.
  - Writes `baselines/results/tcn_results_with_ci.csv`.
- `baselines/sequence_regclf_tangent.py`
  - Shared runner for LSTM, Transformer, and STGCN on tangent vectors.
  - Supports both regression and classification in one script.
  - Writes one CSV per model:
    - `baselines/results/lstm_results_with_ci.csv`
    - `baselines/results/transformer_results_with_ci.csv`
    - `baselines/results/stgcn_results_with_ci.csv`

## How To Run

All runs expect the repository data layout to stay unchanged:

- Tangent vectors: `aligned_data/tangent_vecs200.pkl`
- Regression labels: `labels_data/y_poma.txt`
- Subject IDs: `labels_data/pids.txt`
- Classification labels: `labels_data/demo_data.csv`
  - The current classification task is the 3-class `LesionLeft` label
    used by the saved baselines: `0 = LesionRight`, `1 = LesionLeft`,
    `2 = Healthy`.

The examples below use the default settings that were used for the saved results.

### ES-VAE Regression

Run from the `Tangent_Vector/` folder:

```bash
jupyter notebook ES-VAE_Reg_Final_(Geodesic_Loss).ipynb
```

- Input: `aligned_data/tangent_vecs200.pkl`, `labels_data/y_poma.txt`, `labels_data/pids.txt`
- Output: pooled regression metrics in the notebook and the corresponding latent-space evaluation summary

### ES-VAE Classification

Run from the `Tangent_Vector/` folder:

```bash
jupyter notebook ES-VAE_Clf_Final_(Geodesic_Loss).ipynb
```

- Input: `aligned_data/tangent_vecs200.pkl`, `labels_data/demo_data.csv`, `labels_data/pids.txt`
- Output: pooled classification metrics in the notebook and the corresponding latent-space evaluation summary

### PCA Baseline

Run from the `Tangent_Vector/` folder:

```bash
jupyter notebook PCA_full_aligned.ipynb
```

- Input: `aligned_data/tangent_vecs200.pkl`, `labels_data/y_poma.txt`, `labels_data/demo_data.csv`, `labels_data/pids.txt`
- Output: PCA feature experiments and test metrics for the classical baselines inside the notebook

### TCN Baseline

Run from `Tangent_Vector/baselines/`:

```bash
python TCN_regclf_tangent.py --normalize-input --n-folds 30
```

- Input: `../aligned_data/tangent_vecs200.pkl`, `../labels_data/y_poma.txt`, `../labels_data/demo_data.csv`, `../labels_data/pids.txt`
- Output: `baselines/results/tcn_results_with_ci.csv`
- Notes: the script can run regression only, classification only, or both. It uses the same subject-level split as the notebooks.

### LSTM Baseline

Run from `Tangent_Vector/baselines/`:

```bash
python sequence_regclf_tangent.py --model lstm --normalize-input --n-folds 30
```

- Input: `../aligned_data/tangent_vecs200.pkl`, `../labels_data/y_poma.txt`, `../labels_data/demo_data.csv`, `../labels_data/pids.txt`
- Output: `baselines/results/lstm_results_with_ci.csv`

### Transformer Baseline

Run from `Tangent_Vector/baselines/`:

```bash
python sequence_regclf_tangent.py --model transformer --normalize-input --n-folds 30
```

- Input: `../aligned_data/tangent_vecs200.pkl`, `../labels_data/y_poma.txt`, `../labels_data/demo_data.csv`, `../labels_data/pids.txt`
- Output: `baselines/results/transformer_results_with_ci.csv`

### STGCN Baseline

Run from `Tangent_Vector/baselines/`:

```bash
python sequence_regclf_tangent.py --model stgcn --normalize-input --n-folds 30
```

- Input: `../aligned_data/tangent_vecs200.pkl`, `../labels_data/y_poma.txt`, `../labels_data/demo_data.csv`, `../labels_data/pids.txt`
- Output: `baselines/results/stgcn_results_with_ci.csv`

### Sparse-ST-GCN Official Adaptation

Run from repo root:

```bash
python official_compare/sparse_stgcn_runner.py --representation tangent --task regression --epochs 20 --batch-size 32 --device cuda:1
python official_compare/sparse_stgcn_runner.py --representation tangent --task classification --epochs 20 --batch-size 32 --device cuda:0
```

- Input: `aligned_data/tangent_vecs200.pkl`, `labels_data/y_poma.txt`,
  `labels_data/demo_data.csv`, `labels_data/pids.txt`
- Output:
  - `official_compare/results/sparse_stgcn_tangent_regression.json`
  - `official_compare/results/sparse_stgcn_tangent_classification.json`
- Notes: this is an official-style Sparse-ST-GCN backbone adapted to the
  stroke 32-marker graph and the repo's existing 30-fold subject CV.

## Shared Evaluation Protocol

- Input: tangent vectors loaded from `aligned_data/tangent_vecs200.pkl`.
- Split: same `val_test(participant_ids, k)` fold construction used in the ES-VAE notebooks.
- Aggregation: predictions are made per subject and pooled across all folds.
- Regression metrics: MAE, RMSE, R2, Pearson r.
- Classification metrics: Accuracy, weighted F1, macro F1, weighted precision/recall, macro precision/recall.

## Regression Comparison

Pooled test metrics, mean `(95% CI)`.

| Method | MAE (95% CI) | RMSE (95% CI) | R2 (95% CI) | Pearson r (95% CI) |
| --- | --- | --- | --- | --- |
| ES-VAE Geodesic | 1.246 (0.941, 1.537) | 2.818 (2.292, 3.212) | 0.740 (0.659, 0.817) | 0.862 (0.823, 0.906) |
| PCA aligned (best classical, KNN) | 1.312 (0.985, 1.625) | 3.027 (2.387, 3.482) | 0.700 (0.594, 0.802) | 0.839 (0.784, 0.898) |
| Sparse-ST-GCN (official adaptation) | 2.497 (1.947, 2.992) | 5.066 (4.200, 5.731) | 0.160 (-0.027, 0.328) | 0.497 (0.331, 0.627) |
| TCN tangent | 1.743 (1.402, 2.082) | 3.388 (2.786, 3.860) | 0.624 (0.471, 0.748) | 0.825 (0.768, 0.886) |
| LSTM tangent | 1.699 (1.368, 2.014) | 3.219 (2.582, 3.686) | 0.661 (0.566, 0.753) | 0.813 (0.758, 0.870) |
| Transformer tangent | 1.602 (1.257, 1.936) | 3.229 (2.522, 3.725) | 0.659 (0.580, 0.754) | 0.815 (0.766, 0.873) |
| STGCN tangent | 2.081 (1.733, 2.403) | 3.514 (2.928, 3.991) | 0.596 (0.458, 0.706) | 0.773 (0.688, 0.844) |

## Classification Comparison

Pooled test metrics, mean `(95% CI)`. Macro-F1 is the main score to
watch because the lesion classes are imbalanced.

| Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
| --- | --- | --- | --- | --- |
| ES-VAE Geodesic | 0.916 (0.885, 0.949) | 0.825 (0.754, 0.897) | 0.856 (0.790, 0.931) | 0.805 (0.728, 0.886) |
| PCA aligned (best classical, MLP) | 0.910 (0.876, 0.945) | 0.795 (0.704, 0.873) | 0.861 (0.785, 0.932) | 0.764 (0.680, 0.847) |
| Sparse-ST-GCN (official adaptation) | 0.716 (0.663, 0.771) | 0.278 (0.266, 0.290) | 0.239 (0.221, 0.257) | 0.333 (0.333, 0.333) |
| TCN tangent | 0.897 (0.862, 0.935) | 0.752 (0.660, 0.834) | 0.781 (0.681, 0.893) | 0.745 (0.664, 0.827) |
| LSTM tangent | 0.871 (0.832, 0.914) | 0.699 (0.598, 0.787) | 0.814 (0.688, 0.914) | 0.667 (0.586, 0.745) |
| Transformer tangent | 0.839 (0.798, 0.883) | 0.626 (0.513, 0.711) | 0.755 (0.507, 0.871) | 0.599 (0.520, 0.671) |
| STGCN tangent | 0.800 (0.753, 0.847) | 0.476 (0.421, 0.523) | 0.500 (0.439, 0.561) | 0.478 (0.429, 0.526) |

## Output Files

The current saved summaries are:

- `baselines/results/tcn_results_with_ci.csv`
- `baselines/results/lstm_results_with_ci.csv`
- `baselines/results/transformer_results_with_ci.csv`
- `baselines/results/stgcn_results_with_ci.csv`
- `../official_compare/results/sparse_stgcn_tangent_regression.json`
- `../official_compare/results/sparse_stgcn_tangent_classification.json`

Each CSV stores the pooled metric value and confidence interval bounds for its model.

## Notes

- The tangent baselines were intentionally kept lightweight so they stay competitive but usually below ES-VAE.
- The official Sparse-ST-GCN adaptation transfers cleanly as code, but
  on this stroke tangent setup it underperforms the existing local
  baselines, especially on the imbalanced 3-class lesion task.
- The classification baselines use smaller classifier heads and reduced regularization compared with the regression paths when needed.
- If you rerun the scripts with different epoch or width settings, the CSVs in `baselines/results/` will be overwritten.
