#!/usr/bin/env python
"""Matched raw-skeleton VAE + k-NN baseline for stroke_riemann.

This script is the raw-side counterpart to the no-alignment ablation row:
  - raw lab-frame coordinates from `../csv_r`
  - temporal resampling to T=200 only
  - no translation / scale / rotation / TSRVF alignment
  - simple VAE encoder + downstream k-NN / k-NN regressor

It is intended to replace the older `VAE_full_raw_unaligned.ipynb` numbers
when a strictly matched raw-vs-tangent comparison is desired.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from ci import subject_bootstrap_ci
from ci_class import subject_bootstrap_ci_class
from val_test import val_test


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "csv_r"
DEMO_CSV = REPO_ROOT / "labels_data" / "demo_data.csv"

SEED = 42


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resample_curve_euclidean(curve: np.ndarray, target_len: int) -> np.ndarray:
    src_len = curve.shape[-1]
    if src_len == target_len:
        return np.array(curve, dtype=np.float32, copy=True)
    src_t = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = curve.reshape(-1, src_len)
    flat_out = np.vstack([np.interp(dst_t, src_t, row).astype(np.float32) for row in flat])
    return flat_out.reshape(curve.shape[0], curve.shape[1], target_len)


def parse_raw_curve(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    gait_cycles = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    n_rows = gait_cycles.shape[0]
    return gait_cycles.reshape(n_rows, 32, 3).transpose(1, 2, 0).astype(np.float32)


def load_raw_dataset(tslen: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(RAW_DIR.glob("ID*_*.csv"), key=lambda p: int(p.name.split("_")[0][2:]))
    participant_ids = np.array([re.search(r"ID(\d+)_", p.name).group(1) for p in files], dtype=str)
    y_poma = np.array([int(p.name.split("_")[1].split(".")[0]) for p in files], dtype=np.float32)
    curves = [resample_curve_euclidean(parse_raw_curve(path), target_len=tslen) for path in files]
    x = np.stack(curves, axis=0).astype(np.float32)
    return x, participant_ids, y_poma


def load_lesion_labels(participant_ids: np.ndarray) -> np.ndarray:
    demo_df = pd.read_csv(DEMO_CSV)
    id_to_lesion = dict(zip(demo_df["s"].astype(int), demo_df["LesionLeft"]))
    return np.array([int(id_to_lesion[int(pid)]) for pid in participant_ids], dtype=np.int64)


class StrokeVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 38, hidden: int = 128, decoder_hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden, bias=False)
        self.mu_head = nn.Linear(hidden, latent_dim, bias=False)
        self.lv_head = nn.Linear(hidden, latent_dim)
        self.dec1 = nn.Linear(latent_dim, decoder_hidden, bias=False)
        self.dec2 = nn.Linear(decoder_hidden, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.enc1(x))
        h = self.dropout(h)
        return self.mu_head(h), self.lv_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decode(z)
        return x_hat, mu, lv


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor, beta: float = 2 ** (-3)) -> torch.Tensor:
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()
    kl = -0.5 * torch.sum(1.0 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
    return recon + beta * kl


def train_vae(train_x: np.ndarray, device: torch.device, epochs: int, seed: int) -> StrokeVAE:
    set_deterministic(seed)
    x = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    model = StrokeVAE(x.shape[1]).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        x_hat, mu, lv = model(x)
        loss = vae_loss(x, x_hat, mu, lv)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def encode_latents(model: StrokeVAE, x: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    mu, _ = model.encode(tensor)
    return mu.detach().cpu().numpy()


def evaluate_regression(x_flat: np.ndarray, participant_ids: np.ndarray, y_poma: np.ndarray, device: torch.device, epochs: int) -> dict:
    pooled = {"targets": [], "preds": [], "subjects": []}
    for fold in range(30):
        val_pids, test_pids = val_test(participant_ids, fold)
        val_pids = set(val_pids)
        test_pids = set(test_pids)
        train_pids = set(participant_ids) - val_pids - test_pids

        train_idx = np.array([i for i, pid in enumerate(participant_ids) if pid in train_pids], dtype=np.int64)
        test_idx = np.array([i for i, pid in enumerate(participant_ids) if pid in test_pids], dtype=np.int64)

        x_train = x_flat[train_idx].astype(np.float32)
        x_test = x_flat[test_idx].astype(np.float32)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

        model = train_vae(x_train, device=device, epochs=epochs, seed=SEED + fold)
        z_train = encode_latents(model, x_train, device)
        z_test = encode_latents(model, x_test, device)

        knr = KNeighborsRegressor()
        knr.fit(z_train, y_poma[train_idx])
        preds = knr.predict(z_test)

        pooled["targets"].extend(y_poma[test_idx].tolist())
        pooled["preds"].extend(preds.tolist())
        pooled["subjects"].extend(participant_ids[test_idx].tolist())

    targets = np.asarray(pooled["targets"], dtype=np.float32)
    preds = np.asarray(pooled["preds"], dtype=np.float32)
    subjects = np.asarray(pooled["subjects"], dtype=str)
    ci = subject_bootstrap_ci(targets, preds, subjects)
    return {
        "metrics": {
            "MAE": {"mean": float(mean_absolute_error(targets, preds)), "ci_low": float(ci["MAE"]["ci"][0]), "ci_high": float(ci["MAE"]["ci"][1])},
            "RMSE": {"mean": float(np.sqrt(mean_squared_error(targets, preds))), "ci_low": float(ci["RMSE"]["ci"][0]), "ci_high": float(ci["RMSE"]["ci"][1])},
            "R2": {"mean": float(r2_score(targets, preds)), "ci_low": float(ci["R2"]["ci"][0]), "ci_high": float(ci["R2"]["ci"][1])},
            "Pearson r": {"mean": float(np.corrcoef(targets, preds)[0, 1]), "ci_low": float(ci["Pearson r"]["ci"][0]), "ci_high": float(ci["Pearson r"]["ci"][1])},
        },
        "pooled": pooled,
        "config": {"epochs": epochs, "latent_dim": 38, "hidden": 128, "decoder_hidden": 16, "dropout": 0.10, "beta_kl": 2 ** (-3), "knn": "default"},
    }


def evaluate_classification(x_flat: np.ndarray, participant_ids: np.ndarray, y_lesion: np.ndarray, device: torch.device, epochs: int) -> dict:
    pooled = {"targets": [], "preds": [], "subjects": []}
    for fold in range(30):
        val_pids, test_pids = val_test(participant_ids, fold)
        val_pids = set(val_pids)
        test_pids = set(test_pids)
        train_pids = set(participant_ids) - val_pids - test_pids

        train_idx = np.array([i for i, pid in enumerate(participant_ids) if pid in train_pids], dtype=np.int64)
        test_idx = np.array([i for i, pid in enumerate(participant_ids) if pid in test_pids], dtype=np.int64)

        x_train = x_flat[train_idx].astype(np.float32)
        x_test = x_flat[test_idx].astype(np.float32)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

        model = train_vae(x_train, device=device, epochs=epochs, seed=SEED + fold)
        z_train = encode_latents(model, x_train, device)
        z_test = encode_latents(model, x_test, device)

        knn = KNeighborsClassifier()
        knn.fit(z_train, y_lesion[train_idx])
        preds = knn.predict(z_test)

        pooled["targets"].extend(y_lesion[test_idx].tolist())
        pooled["preds"].extend(preds.tolist())
        pooled["subjects"].extend(participant_ids[test_idx].tolist())

    targets = np.asarray(pooled["targets"], dtype=np.int64)
    preds = np.asarray(pooled["preds"], dtype=np.int64)
    subjects = np.asarray(pooled["subjects"], dtype=str)
    ci = subject_bootstrap_ci_class(targets, preds, subjects)
    return {
        "metrics": {
            "Accuracy": {"mean": float(accuracy_score(targets, preds)), "ci_low": float(ci["Accuracy"]["ci"][0]), "ci_high": float(ci["Accuracy"]["ci"][1])},
            "Macro F1": {"mean": float(f1_score(targets, preds, average="macro", zero_division=0)), "ci_low": float(ci["F1 (macro)"]["ci"][0]), "ci_high": float(ci["F1 (macro)"]["ci"][1])},
            "Macro Precision": {"mean": float(precision_score(targets, preds, average="macro", zero_division=0)), "ci_low": float(ci["Precision (macro)"]["ci"][0]), "ci_high": float(ci["Precision (macro)"]["ci"][1])},
            "Macro Recall": {"mean": float(recall_score(targets, preds, average="macro", zero_division=0)), "ci_low": float(ci["Recall (macro)"]["ci"][0]), "ci_high": float(ci["Recall (macro)"]["ci"][1])},
        },
        "pooled": pooled,
        "config": {"epochs": epochs, "latent_dim": 38, "hidden": 128, "decoder_hidden": 16, "dropout": 0.10, "beta_kl": 2 ** (-3), "knn": "default"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_matched")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda:0" or torch.cuda.is_available()) else "cpu")
    x, participant_ids, y_poma = load_raw_dataset(tslen=200)
    x_flat = x.reshape(x.shape[0], -1).astype(np.float32)
    y_lesion = load_lesion_labels(participant_ids)

    reg = evaluate_regression(x_flat, participant_ids, y_poma, device=device, epochs=args.epochs)
    clf = evaluate_classification(x_flat, participant_ids, y_lesion, device=device, epochs=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reg_path = args.output_dir / "vae_knn_raw_matched_regression.json"
    clf_path = args.output_dir / "vae_knn_raw_matched_classification.json"
    reg_path.write_text(json.dumps(reg, indent=2))
    clf_path.write_text(json.dumps(clf, indent=2))

    print(f"Wrote {reg_path}")
    print(json.dumps(reg["metrics"], indent=2))
    print(f"Wrote {clf_path}")
    print(json.dumps(clf["metrics"], indent=2))


if __name__ == "__main__":
    main()
