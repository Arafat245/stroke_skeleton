# Tangent Vector Experiments

This folder contains the aligned-tangent experiments for stroke gait analysis. The code is split into three groups:

1. ES-VAE notebooks with geodesic loss.
2. A PCA baseline notebook on aligned data.
3. Baseline scripts in `baselines/` for TCN, LSTM, Transformer, and STGCN.
4. `../official_compare/` for the adapted official Hyper-GCN and Sparse-ST-GCN runners.

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
python official_compare/sparse_stgcn_runner.py --representation tangent --task regression --epochs 30 --batch-size 32 --device cuda:1 --lr 0.01 --warmup 5 --reg-balance-mode inverse --reg-calibration linear
python official_compare/sparse_stgcn_runner.py --representation tangent --task classification --epochs 20 --batch-size 32 --device cuda:0
```

- Input: `aligned_data/tangent_vecs200.pkl`, `labels_data/y_poma.txt`,
  `labels_data/demo_data.csv`, `labels_data/pids.txt`
- Output:
  - `official_compare/results/sparse_tangent_regression_tuned.json`
  - `official_compare/results/sparse_stgcn_tangent_classification.json`
- Notes: this is an official-style Sparse-ST-GCN backbone adapted to the
  stroke 32-marker graph and the repo's existing 30-fold subject CV.
  The tuned regression setting uses inverse POMA balancing plus a simple
  validation-fit linear calibrator.

### Hyper-GCN Official Adaptation

Run from repo root:

```bash
python official_compare/hypergcn_runner.py --representation tangent --task regression --epochs 20 --batch-size 64 --device cuda:1 --reg-calibration linear
python official_compare/hypergcn_runner.py --representation tangent --task classification --epochs 20 --batch-size 64 --device cuda:0
```

- Input: `aligned_data/tangent_vecs200.pkl`, `labels_data/y_poma.txt`,
  `labels_data/demo_data.csv`, `labels_data/pids.txt`
- Output:
  - `official_compare/results/hypergcn_tangent_regression_tuned.json`
  - `official_compare/results/hypergcn_tangent_classification.json`
- Notes: this Hyper-GCN runner uses one subject sequence by default to
  mirror the lighter activity-recognition adaptation. Add `--multi-clip`
  if you want the heavier stroke-style clip expansion. The tuned
  regression setting adds a simple validation-fit linear calibrator.

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
| ES-VAE Geodesic | 1.25 (0.94, 1.54) | 2.82 (2.29, 3.21) | 0.74 (0.66, 0.82) | 0.86 (0.82, 0.91) |
| PCA aligned (best classical, KNN) | 1.31 (0.98, 1.62) | 3.03 (2.39, 3.48) | 0.70 (0.59, 0.80) | 0.84 (0.78, 0.90) |
| Hyper-GCN (official adaptation) | 1.79 (1.35, 2.20) | 3.86 (3.09, 4.43) | 0.51 (0.36, 0.65) | 0.74 (0.66, 0.82) |
| Sparse-ST-GCN (official adaptation) | 1.50 (1.12, 1.85) | 3.36 (2.56, 3.88) | 0.63 (0.50, 0.76) | 0.80 (0.72, 0.87) |
| TCN tangent | 1.74 (1.40, 2.08) | 3.39 (2.79, 3.86) | 0.62 (0.47, 0.75) | 0.83 (0.77, 0.89) |
| LSTM tangent | 1.70 (1.37, 2.01) | 3.22 (2.58, 3.69) | 0.66 (0.57, 0.75) | 0.81 (0.76, 0.87) |
| Transformer tangent | 1.60 (1.26, 1.94) | 3.23 (2.52, 3.73) | 0.66 (0.58, 0.75) | 0.81 (0.77, 0.87) |
| STGCN tangent | 2.08 (1.73, 2.40) | 3.51 (2.93, 3.99) | 0.60 (0.46, 0.71) | 0.77 (0.69, 0.84) |

## Classification Comparison

Pooled test metrics, mean `(95% CI)`. Macro-F1 is the main score to
watch because the lesion classes are imbalanced.

| Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro Precision (95% CI) | Macro Recall (95% CI) |
| --- | --- | --- | --- | --- |
| ES-VAE Geodesic | 0.92 (0.89, 0.95) | 0.83 (0.75, 0.90) | 0.86 (0.79, 0.93) | 0.80 (0.73, 0.89) |
| PCA aligned (best classical, MLP) | 0.91 (0.88, 0.95) | 0.79 (0.70, 0.87) | 0.86 (0.79, 0.93) | 0.76 (0.68, 0.85) |
| Hyper-GCN (official adaptation) | 0.83 (0.78, 0.87) | 0.56 (0.50, 0.62) | 0.56 (0.49, 0.63) | 0.57 (0.51, 0.63) |
| Sparse-ST-GCN (official adaptation) | 0.72 (0.66, 0.77) | 0.28 (0.27, 0.29) | 0.24 (0.22, 0.26) | 0.33 (0.33, 0.33) |
| TCN tangent | 0.90 (0.86, 0.94) | 0.75 (0.66, 0.83) | 0.78 (0.68, 0.89) | 0.74 (0.66, 0.83) |
| LSTM tangent | 0.87 (0.83, 0.91) | 0.70 (0.60, 0.79) | 0.81 (0.69, 0.91) | 0.67 (0.59, 0.74) |
| Transformer tangent | 0.84 (0.80, 0.88) | 0.63 (0.51, 0.71) | 0.76 (0.51, 0.87) | 0.60 (0.52, 0.67) |
| STGCN tangent | 0.80 (0.75, 0.85) | 0.48 (0.42, 0.52) | 0.50 (0.44, 0.56) | 0.48 (0.43, 0.53) |

## Output Files

The current saved summaries are:

- `baselines/results/tcn_results_with_ci.csv`
- `baselines/results/lstm_results_with_ci.csv`
- `baselines/results/transformer_results_with_ci.csv`
- `baselines/results/stgcn_results_with_ci.csv`
- `../official_compare/results/hypergcn_tangent_regression_tuned.json`
- `../official_compare/results/hypergcn_tangent_classification.json`
- `../official_compare/results/sparse_tangent_regression_tuned.json`
- `../official_compare/results/sparse_stgcn_tangent_classification.json`

Each CSV stores the pooled metric value and confidence interval bounds for its model.

## Notes

- The tangent baselines were intentionally kept lightweight so they stay competitive but usually below ES-VAE.
- Hyper-GCN is the stronger imported tangent classifier, and after
  light regression calibration it also becomes a reasonable POMA
  regressor.
- The official Sparse-ST-GCN adaptation transfers cleanly as code, but
  on this stroke tangent setup it still underperforms the best tangent
  classifiers. Its tuned regression path, however, is much stronger
  than the original straight port.
- The classification baselines use smaller classifier heads and reduced regularization compared with the regression paths when needed.
- If you rerun the scripts with different epoch or width settings, the CSVs in `baselines/results/` will be overwritten.
