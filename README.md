# Stroke Gait Analysis — Tangent-Vector vs Raw-Skeleton

This repository is the official implementation of *Stroke Gait Analysis
— Tangent-Vector vs Raw-Skeleton*, a controlled comparison of
**aligned tangent-vector gait representations** against **raw 3D marker
coordinates** for stroke gait prediction. We evaluate two tasks on the
same `155` subjects under the same subject-level `30`-fold
cross-validation:

- **Regression**: `POMA`
- **Classification**: 3-class `LesionLeft`

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Tested with Python `3.10`, PyTorch `2.x`, scikit-learn, NumPy, pandas,
SciPy, and Jupyter. A CUDA-capable GPU is recommended for the
deep-learning baselines but not required for the PCA / VAE + k-NN
pipelines.

Datasets are expected at:

- `aligned_data/tangent_vecs200.pkl` — tangent vectors `(32, 3, 200, 155)`
- `data/processed_loaded.pt` — raw regression subjects
- `data_clf/processed_loaded.pt` — raw classification subjects
- `labels_data/` — `POMA`, participant ids, lesion labels

## Training

To train and evaluate all models reported in the paper, run the
representation-specific commands in the sub-folders:

```train
# Tangent-vector pipeline (proposed)
cd Tangent_Vector
python baselines/TCN_regclf_tangent.py --normalize-input --n-folds 30
python baselines/sequence_regclf_tangent.py --model lstm --normalize-input --n-folds 30
python baselines/sequence_regclf_tangent.py --model transformer --normalize-input --n-folds 30
python baselines/sequence_regclf_tangent.py --model stgcn --normalize-input --n-folds 30
jupyter notebook ES-VAE_Reg_Final_\(Geodesic_Loss\).ipynb
jupyter notebook ES-VAE_Clf_Final_\(Geodesic_Loss\).ipynb
jupyter notebook PCA_full_aligned.ipynb

# Raw-skeleton pipeline
cd ../Raw_Skeleton
python TCN_regclf_raw.py
python vae_knn_raw_matched.py --device cuda:0
jupyter notebook PCA_full_raw_unaligned.ipynb
jupyter notebook VAE_full_raw_unaligned.ipynb
jupyter notebook LSTM_regclf_raw.ipynb
jupyter notebook Transformer_regclf_raw.ipynb
jupyter notebook STGCN.ipynb

# Adapted official graph baselines (run from repo root)
cd ..
python official_compare/hypergcn_runner.py --representation tangent --task regression --epochs 20 --batch-size 64 --reg-calibration linear
python official_compare/hypergcn_runner.py --representation raw --task classification --epochs 20 --batch-size 64
python official_compare/sparse_stgcn_runner.py --representation tangent --task regression --epochs 30 --batch-size 32 --lr 0.01 --warmup 5 --reg-balance-mode inverse --reg-calibration linear
python official_compare/sparse_stgcn_runner.py --representation raw --task classification --epochs 30 --patience 10 --batch-size 32 --lr 0.01 --label-smoothing 0.0 --no-clf-balancing --warmup 5
```

All experiments use the same subject-level `30`-fold split helper from
`Tangent_Vector/val_test.py` so results are directly comparable.

## Evaluation

Evaluation is integrated with training: each runner / notebook performs
subject-level `30`-fold cross-validation and writes pooled out-of-fold
predictions plus bootstrap `95%` confidence intervals. Saved JSON
summaries from the adapted graph baselines are written to
`official_compare/results/`. To re-aggregate metrics from the saved
JSON outputs:

```eval
python official_compare/hypergcn_runner.py --representation tangent --task regression --epochs 20 --batch-size 64 --reg-calibration linear --output-name hypergcn_tangent_regression_tuned.json
python official_compare/sparse_stgcn_runner.py --representation tangent --task classification --epochs 30 --patience 10 --batch-size 32 --lr 0.01 --label-smoothing 0.0 --clf-balance-mode inverse --warmup 5 --output-name sparse_stgcn_tangent_classification_tuned.json
```

## Pre-trained Models

Trained model checkpoints are not redistributed in this repository. All
reported numbers can be reproduced end-to-end from the commands above
on the included data using fixed seeds and the shared
`30`-fold subject split in `Tangent_Vector/val_test.py`.

## Results

Pooled out-of-fold metrics, **mean (95% CI)** from subject-level
bootstrap.

### Regression — POMA (30-fold subject CV)

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

### Classification — 3-class LesionLeft (30-fold subject CV)

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

### Tangent vs Raw — same-method comparison

`Δ MAE = Raw − Tangent` (positive ⇒ tangent better).
`Δ Macro F1 = Tangent − Raw` (positive ⇒ tangent better).

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

1. **Tangent ES-VAE is the best overall model on both tasks** — strongest
   regressor (`MAE 1.25`, `R2 0.74`) and strongest classifier
   (`Macro F1 0.83`).
2. **`PCA + k-NN` is the fair classical comparator** to ES-VAE; the
   tangent variant beats the raw variant on both tasks.
3. **Tangent wins the classification comparison for every matched
   model family**, with the largest gains for `PCA + k-NN`,
   `Vanilla VAE / ES-VAE`, and the recurrent baselines.
4. **Regression is mixed outside the VAE family** — tangent wins for
   most families, but raw `LSTM` and raw `Hyper-GCN` outperform their
   tangent counterparts.
5. **The adapted graph baselines become much more usable after light
   task-specific tuning** — `Sparse-ST-GCN` improves sharply on
   classification, while still trailing the top tangent baselines.

## Repository layout

```text
stroke_riemann/
├── README.md
├── official_compare/             adapted official Hyper-GCN / Sparse-ST-GCN runners
├── Tangent_Vector/               tangent-vector pipeline (see Tangent_Vector/README.md)
├── Raw_Skeleton/                 raw-marker pipeline (see Raw_Skeleton/README.md)
├── aligned_data/                 tangent vectors and aligned curves
├── data/                         processed raw regression subjects
├── data_clf/                     processed raw classification subjects
└── labels_data/                  POMA, participant ids, lesion labels
```

See the folder-specific READMEs for per-pipeline commands and details:

- [Tangent_Vector/README.md](./Tangent_Vector/README.md)
- [Raw_Skeleton/README.md](./Raw_Skeleton/README.md)

## Contributing

This code is released under the **MIT License** for academic and
research use. Contributions are welcome via pull requests; please open
an issue first to discuss substantial changes. When reporting new
results, please follow the same subject-level `30`-fold protocol so
that numbers stay directly comparable to those above.
