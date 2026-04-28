# Tangent Vector Experiments

This folder contains the aligned-tangent experiments for stroke gait analysis. The code is split into three groups:

1. ES-VAE notebooks with geodesic loss.
2. A PCA baseline notebook on aligned data.
3. Baseline scripts in `baselines/` for TCN, LSTM, Transformer, and STGCN.

All runs use the same subject-level cross-validation scheme from `val_test.py` unless noted otherwise. Metrics are pooled across 30 folds, with subject bootstrap confidence intervals where available.

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

## Shared Evaluation Protocol

- Input: tangent vectors loaded from `aligned_data/tangent_vecs200.pkl`.
- Split: same `val_test(participant_ids, k)` fold construction used in the ES-VAE notebooks.
- Aggregation: predictions are made per subject and pooled across all folds.
- Regression metrics: MAE, RMSE, R2, Pearson r.
- Classification metrics: Accuracy, weighted F1, macro F1, weighted precision/recall, macro precision/recall.

## Regression Comparison

Test metrics pooled across folds.

| Method | MAE | RMSE | R2 | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| ES-VAE Geodesic | 1.246 | 2.818 | 0.740 | 0.862 |
| PCA aligned (best classical) | 1.312 | 3.027 | 0.700 | 0.839 |
| TCN tangent | 1.743 | 3.388 | 0.624 | 0.825 |
| LSTM tangent | 1.699 | 3.219 | 0.661 | 0.813 |
| Transformer tangent | 1.602 | 3.229 | 0.659 | 0.815 |
| STGCN tangent | 2.081 | 3.514 | 0.596 | 0.773 |

## Classification Comparison

Test metrics pooled across folds. Macro-F1 is the main score to watch because the lesion classes are imbalanced.

| Method | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| --- | ---: | ---: | ---: | ---: |
| ES-VAE Geodesic | 0.916 | 0.825 | 0.856 | 0.805 |
| PCA aligned (best classical) | 0.910 | 0.795 | 0.861 | 0.764 |
| TCN tangent | 0.897 | 0.752 | 0.781 | 0.745 |
| LSTM tangent | 0.871 | 0.699 | 0.814 | 0.667 |
| Transformer tangent | 0.839 | 0.626 | 0.755 | 0.599 |
| STGCN tangent | 0.800 | 0.476 | 0.500 | 0.478 |

## Output Files

The current saved summaries are:

- `baselines/results/tcn_results_with_ci.csv`
- `baselines/results/lstm_results_with_ci.csv`
- `baselines/results/transformer_results_with_ci.csv`
- `baselines/results/stgcn_results_with_ci.csv`

Each CSV stores the pooled metric value and confidence interval bounds for its model.

## Notes

- The tangent baselines were intentionally kept lightweight so they stay competitive but usually below ES-VAE.
- The classification baselines use smaller classifier heads and reduced regularization compared with the regression paths when needed.
- If you rerun the scripts with different epoch or width settings, the CSVs in `baselines/results/` will be overwritten.
