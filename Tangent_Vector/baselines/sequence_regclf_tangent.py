#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TV_DIR = Path(__file__).resolve().parents[1]
if str(TV_DIR) not in sys.path:
    sys.path.insert(0, str(TV_DIR))

from ci import subject_bootstrap_ci
from ci_class import subject_bootstrap_ci_class
from val_test import val_test
from TCN_regclf_tangent import (
    build_input_standardizer,
    load_dataset,
    sliding_starts,
    split_fold,
    set_deterministic,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LSTM / Transformer / STGCN baselines on tangent vectors."
    )
    parser.add_argument("--model", choices=["lstm", "transformer", "stgcn"], required=True)
    parser.add_argument("--tslen", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-reg", type=int, default=8)
    parser.add_argument("--epochs-clf", type=int, default=6)
    parser.add_argument("--lr-reg", type=float, default=5e-4)
    parser.add_argument("--lr-clf", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--no-clf-balancing", action="store_true")
    parser.add_argument("--use-val-early-stopping", action="store_true")
    parser.add_argument("--n-folds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )

    parser.add_argument("--lstm-reg-hidden", type=int, default=64)
    parser.add_argument("--lstm-clf-hidden", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-dropout", type=float, default=0.15)
    parser.add_argument("--lstm-bidir", action="store_true")

    parser.add_argument("--tr-reg-d-model", type=int, default=24)
    parser.add_argument("--tr-clf-d-model", type=int, default=12)
    parser.add_argument("--tr-reg-d-ff", type=int, default=64)
    parser.add_argument("--tr-clf-d-ff", type=int, default=32)
    parser.add_argument("--tr-reg-heads", type=int, default=2)
    parser.add_argument("--tr-clf-heads", type=int, default=2)
    parser.add_argument("--tr-reg-layers", type=int, default=1)
    parser.add_argument("--tr-clf-layers", type=int, default=1)
    parser.add_argument("--tr-reg-dropout", type=float, default=0.10)
    parser.add_argument("--tr-clf-dropout", type=float, default=0.20)

    parser.add_argument("--stgcn-reg-channels", type=str, default="32,64")
    parser.add_argument("--stgcn-clf-channels", type=str, default="16,32")
    parser.add_argument("--stgcn-reg-dropout", type=float, default=0.20)
    parser.add_argument("--stgcn-clf-dropout", type=float, default=0.30)
    parser.add_argument("--stgcn-kernel", type=int, default=7)
    return parser.parse_args()


def get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    def __init__(self, windows, targets):
        self.x = torch.from_numpy(windows.astype(np.float32))
        if np.issubdtype(targets.dtype, np.integer):
            self.y = torch.from_numpy(targets.astype(np.int64))
        else:
            self.y = torch.from_numpy(targets.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_subject_windows(x_subjects, starts, window_size):
    windows = []
    owners = []
    for subject_idx, x in enumerate(x_subjects):
        for start in starts:
            w = x[:, start : start + window_size].T  # (W, C)
            windows.append(w)
            owners.append(subject_idx)
    return np.stack(windows).astype(np.float32), np.asarray(owners, dtype=np.int64)


def build_regression_loader(x_subjects, y_subjects, starts, window_size, batch_size, seed):
    windows, owners = make_subject_windows(x_subjects, starts, window_size)
    targets = y_subjects[owners]
    dataset = WindowDataset(windows, targets)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def build_classification_loader(
    x_subjects, y_subjects, starts, window_size, batch_size, seed, use_balancing=True
):
    windows, owners = make_subject_windows(x_subjects, starts, window_size)
    targets = y_subjects[owners]
    dataset = WindowDataset(windows, targets)
    if use_balancing:
        class_counts = np.bincount(targets, minlength=int(targets.max()) + 1).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / np.sqrt(class_counts)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def parse_channel_str(channel_str):
    return [int(x) for x in channel_str.split(",") if x.strip()]


def safe_subject_bootstrap_ci(targets, preds, subjects):
    try:
        return subject_bootstrap_ci(targets, preds, subjects)
    except Exception:
        return {
            "MAE": {"mean": mean_absolute_error(targets, preds), "ci": np.array([np.nan, np.nan])},
            "RMSE": {"mean": np.sqrt(mean_squared_error(targets, preds)), "ci": np.array([np.nan, np.nan])},
            "R2": {"mean": r2_score(targets, preds), "ci": np.array([np.nan, np.nan])},
            "Pearson r": {"mean": np.corrcoef(targets, preds)[0, 1] if len(np.unique(targets)) > 1 else np.nan, "ci": np.array([np.nan, np.nan])},
        }


def safe_subject_bootstrap_ci_class(targets, preds, subjects):
    try:
        return subject_bootstrap_ci_class(targets, preds, subjects)
    except Exception:
        return {
            "Accuracy": {"mean": accuracy_score(targets, preds), "ci": np.array([np.nan, np.nan])},
            "F1 (weighted)": {"mean": f1_score(targets, preds, average="weighted", zero_division=0), "ci": np.array([np.nan, np.nan])},
            "F1 (macro)": {"mean": f1_score(targets, preds, average="macro", zero_division=0), "ci": np.array([np.nan, np.nan])},
            "Precision (weighted)": {"mean": precision_score(targets, preds, average="weighted", zero_division=0), "ci": np.array([np.nan, np.nan])},
            "Precision (macro)": {"mean": precision_score(targets, preds, average="macro", zero_division=0), "ci": np.array([np.nan, np.nan])},
            "Recall (weighted)": {"mean": recall_score(targets, preds, average="weighted", zero_division=0), "ci": np.array([np.nan, np.nan])},
            "Recall (macro)": {"mean": recall_score(targets, preds, average="macro", zero_division=0), "ci": np.array([np.nan, np.nan])},
        }


def parse_summary_bounds(value):
    if isinstance(value, str):
        vals = np.fromstring(value.strip().strip("[]"), sep=" ")
        if len(vals) == 2:
            return float(vals[0]), float(vals[1])
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return np.nan, np.nan


def save_model_summary(output_dir, model_name, reg_results_df=None, reg_ci_df=None, clf_results_df=None, clf_ci_df=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{model_name}_results_with_ci.csv"
    display_name = "STGCN" if model_name == "stgcn" else model_name.upper()

    rows = []
    if reg_results_df is not None and reg_ci_df is not None:
        reg_row = reg_results_df.loc[display_name]
        ci_mean = reg_ci_df.loc["mean"] if "mean" in reg_ci_df.index else pd.Series(dtype=float)
        ci_row = reg_ci_df.loc["ci"] if "ci" in reg_ci_df.index else pd.Series(dtype=object)
        for metric, value in reg_row.items():
            low, high = parse_summary_bounds(ci_row.get(metric)) if metric in ci_row.index else (np.nan, np.nan)
            rows.append(
                {
                    "task": "regression",
                    "metric": metric,
                    "value": float(value),
                    "ci_mean": float(ci_mean.get(metric, np.nan)) if metric in ci_mean.index else np.nan,
                    "ci_lower": low,
                    "ci_upper": high,
                }
            )
    if clf_results_df is not None and clf_ci_df is not None:
        clf_row = clf_results_df.loc[display_name]
        ci_mean = clf_ci_df.loc["mean"] if "mean" in clf_ci_df.index else pd.Series(dtype=float)
        ci_row = clf_ci_df.loc["ci"] if "ci" in clf_ci_df.index else pd.Series(dtype=object)
        for metric, value in clf_row.items():
            low, high = parse_summary_bounds(ci_row.get(metric)) if metric in ci_row.index else (np.nan, np.nan)
            rows.append(
                {
                    "task": "classification",
                    "metric": metric,
                    "value": float(value),
                    "ci_mean": float(ci_mean.get(metric, np.nan)) if metric in ci_mean.index else np.nan,
                    "ci_lower": low,
                    "ci_upper": high,
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    return summary_path, summary_df


class LSTMBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        feat = self.drop(self.act(self.proj(feat)))
        return feat


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.backbone = LSTMBackbone(input_dim, hidden_dim, num_layers, bidirectional, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout, num_classes):
        super().__init__()
        self.backbone = LSTMBackbone(input_dim, hidden_dim, num_layers, bidirectional, dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, n_heads, n_layers, dropout):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        z = self.embed(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.proj(z)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, n_heads, n_layers, dropout):
        super().__init__()
        self.backbone = TransformerBackbone(input_dim, d_model, d_ff, n_heads, n_layers, dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, n_heads, n_layers, dropout, num_classes):
        super().__init__()
        self.backbone = TransformerBackbone(input_dim, d_model, d_ff, n_heads, n_layers, dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_default_skeleton_adjacency(num_nodes=32):
    A = torch.eye(num_nodes)
    edges = [(4, 5), (5, 6), (6, 7), (4, 6), (5, 7)]
    edges += [(0, 4), (1, 4), (2, 4), (3, 4)]
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, A):
        x = torch.einsum("ij,btjc->btic", A, x)
        return self.linear(x)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.bn(self.conv(x))
        return out.permute(0, 2, 3, 1)


class STBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal = TemporalConvBlock(out_channels, kernel_size)
        self.temporal_bn = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x, A):
        res = x
        out = self.spatial(x, A)
        out = out.permute(0, 3, 1, 2)
        out = F.relu(self.spatial_bn(out))
        out = out.permute(0, 2, 3, 1)
        out = self.temporal(out)
        out = out.permute(0, 3, 1, 2)
        out = self.temporal_bn(out)
        out = out.permute(0, 2, 3, 1)
        if self.downsample is not None:
            res = self.downsample(res)
        return F.relu(out + res)


class STGCNBackbone(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, kernel_size, dropout):
        super().__init__()
        self.register_buffer("A", build_default_skeleton_adjacency(num_nodes))
        self.input_embed = nn.Linear(in_channels, hidden_channels[0])
        self.blocks = nn.ModuleList(
            [
                STBlock(hidden_channels[i], hidden_channels[i + 1], kernel_size=kernel_size)
                for i in range(len(hidden_channels) - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_channels[-1], hidden_channels[-1])

    def forward(self, x):
        # x: (B, T, 96) -> (B, T, 32, 3)
        b, t, c = x.shape
        x = x.view(b, t, 32, 3)
        x = self.input_embed(x)
        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=(1, 2))
        x = F.gelu(x)
        x = self.dropout(x)
        return self.proj(x)


class STGCNRegressor(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dropout):
        super().__init__()
        self.backbone = STGCNBackbone(32, 3, hidden_channels, kernel_size, dropout)
        self.head = nn.Linear(hidden_channels[-1], 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class STGCNClassifier(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dropout, num_classes):
        super().__init__()
        self.backbone = STGCNBackbone(32, 3, hidden_channels, kernel_size, dropout)
        self.head = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def window_to_tensor(x_subjects, starts, window_size, device, model_name):
    windows, _ = make_subject_windows(x_subjects, starts, window_size)
    if model_name == "stgcn":
        return torch.from_numpy(windows.astype(np.float32)).to(device)
    return torch.from_numpy(windows.astype(np.float32)).to(device)


def predict_subject_regression(model, x_subjects, scaler, starts, window_size, batch_size, device, model_name, y_mean, y_std):
    model.eval()
    x_subjects = scaler.transform(x_subjects)
    preds = []
    with torch.no_grad():
        for x in x_subjects:
            windows, _ = make_subject_windows([x], starts, window_size)
            window_preds = []
            for i in range(0, len(windows), batch_size):
                xb = torch.from_numpy(windows[i : i + batch_size]).to(device)
                preds_batch = model(xb).cpu().numpy()
                window_preds.append(preds_batch)
            window_preds = np.concatenate(window_preds)
            preds.append(float(np.median(window_preds) * y_std + y_mean))
    return np.clip(np.asarray(preds, dtype=np.float32), 6.0, 28.0)


def predict_subject_classification(model, x_subjects, scaler, starts, window_size, batch_size, device, model_name):
    model.eval()
    x_subjects = scaler.transform(x_subjects)
    preds = []
    with torch.no_grad():
        for x in x_subjects:
            windows, _ = make_subject_windows([x], starts, window_size)
            prob_chunks = []
            for i in range(0, len(windows), batch_size):
                xb = torch.from_numpy(windows[i : i + batch_size]).to(device)
                logits = model(xb)
                prob_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs = np.concatenate(prob_chunks, axis=0).mean(axis=0)
            preds.append(int(np.argmax(probs)))
    return np.asarray(preds, dtype=np.int64)


def regression_metrics(targets, preds):
    return {
        "MAE": mean_absolute_error(targets, preds),
        "RMSE": np.sqrt(mean_squared_error(targets, preds)),
        "R2": r2_score(targets, preds),
    }


def classification_metrics(targets, preds):
    return {
        "Accuracy": accuracy_score(targets, preds),
        "F1 (weighted)": f1_score(targets, preds, average="weighted", zero_division=0),
        "F1 (macro)": f1_score(targets, preds, average="macro", zero_division=0),
        "Precision (weighted)": precision_score(targets, preds, average="weighted", zero_division=0),
        "Precision (macro)": precision_score(targets, preds, average="macro", zero_division=0),
        "Recall (weighted)": recall_score(targets, preds, average="weighted", zero_division=0),
        "Recall (macro)": recall_score(targets, preds, average="macro", zero_division=0),
    }


def build_model(args, task, input_dim):
    if args.model == "lstm":
        hidden = args.lstm_reg_hidden if task == "regression" else args.lstm_clf_hidden
        dropout = args.lstm_dropout
        num_layers = args.lstm_layers
        bidir = args.lstm_bidir
        if task == "regression":
            return LSTMRegressor(input_dim, hidden, num_layers, bidir, dropout)
        return LSTMClassifier(input_dim, hidden, num_layers, bidir, dropout, num_classes=3)

    if args.model == "transformer":
        d_model = args.tr_reg_d_model if task == "regression" else args.tr_clf_d_model
        d_ff = args.tr_reg_d_ff if task == "regression" else args.tr_clf_d_ff
        n_heads = args.tr_reg_heads if task == "regression" else args.tr_clf_heads
        n_layers = args.tr_reg_layers if task == "regression" else args.tr_clf_layers
        dropout = args.tr_reg_dropout if task == "regression" else args.tr_clf_dropout
        if task == "regression":
            return TransformerRegressor(input_dim, d_model, d_ff, n_heads, n_layers, dropout)
        return TransformerClassifier(input_dim, d_model, d_ff, n_heads, n_layers, dropout, num_classes=3)

    hidden_channels = (
        parse_channel_str(args.stgcn_reg_channels)
        if task == "regression"
        else parse_channel_str(args.stgcn_clf_channels)
    )
    dropout = args.stgcn_reg_dropout if task == "regression" else args.stgcn_clf_dropout
    if task == "regression":
        return STGCNRegressor(hidden_channels, args.stgcn_kernel, dropout)
    return STGCNClassifier(hidden_channels, args.stgcn_kernel, dropout, num_classes=3)


def train_regression_fold(args, x_train, y_train, x_val, y_val, fold_seed, device, input_dim):
    set_deterministic(fold_seed)
    scaler = build_input_standardizer(args.normalize_input).fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    starts = sliding_starts(x_train.shape[-1], args.window_size, args.window_stride)
    train_loader = build_regression_loader(
        x_train_scaled, y_train, starts, args.window_size, args.batch_size, fold_seed
    )
    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))
    model = build_model(args, "regression", input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_reg, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_reg, 1))
    criterion = nn.SmoothL1Loss(beta=1.0)

    for epoch in range(args.epochs_reg):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = ((y_batch.to(device) - y_mean) / y_std).float()
            if args.input_noise_std > 0:
                x_batch = x_batch + args.input_noise_std * torch.randn_like(x_batch)
            if args.model == "stgcn":
                preds = model(x_batch)
            else:
                preds = model(x_batch)
            loss = criterion(preds, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return model, scaler, starts, y_mean, y_std


def train_classification_fold(args, x_train, y_train, x_val, y_val, fold_seed, device, input_dim):
    set_deterministic(fold_seed)
    scaler = build_input_standardizer(args.normalize_input).fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    starts = sliding_starts(x_train.shape[-1], args.window_size, args.window_stride)
    train_loader = build_classification_loader(
        x_train_scaled,
        y_train,
        starts,
        args.window_size,
        args.batch_size,
        fold_seed,
        use_balancing=not args.no_clf_balancing,
    )
    model = build_model(args, "classification", input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_clf, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs_clf, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for epoch in range(args.epochs_clf):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()
            if args.input_noise_std > 0:
                x_batch = x_batch + args.input_noise_std * torch.randn_like(x_batch)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return model, scaler, starts


def run_regression_cv(args, x_all, y_poma, participant_ids, device):
    print(f"\n[Regression] {args.model.upper()}")
    model_name = args.model.upper() if args.model != "stgcn" else "STGCN"
    all_results_val = {model_name: {"targets": [], "preds": []}}
    all_results_test = {model_name: {"targets": [], "preds": [], "subjects": []}}
    for fold_idx in tqdm(range(args.n_folds), total=args.n_folds, desc=f"{model_name} regression"):
        train_idx, val_idx, test_idx = split_fold(participant_ids, fold_idx)
        model, scaler, starts, y_mean, y_std = train_regression_fold(
            args,
            x_all[train_idx],
            y_poma[train_idx],
            x_all[val_idx],
            y_poma[val_idx],
            args.seed + fold_idx,
            device,
            x_all.shape[1],
        )
        val_preds = predict_subject_regression(
            model, x_all[val_idx], scaler, starts, args.window_size, args.batch_size, device, args.model, y_mean, y_std
        )
        test_preds = predict_subject_regression(
            model, x_all[test_idx], scaler, starts, args.window_size, args.batch_size, device, args.model, y_mean, y_std
        )
        all_results_val[model_name]["targets"].extend(y_poma[val_idx].tolist())
        all_results_val[model_name]["preds"].extend(val_preds.tolist())
        all_results_test[model_name]["targets"].extend(y_poma[test_idx].tolist())
        all_results_test[model_name]["preds"].extend(test_preds.tolist())
        all_results_test[model_name]["subjects"].extend(participant_ids[test_idx].tolist())
        fold_metrics = regression_metrics(y_poma[val_idx], val_preds)
        print(
            f"Fold {fold_idx + 1:02d} | {model_name} | Validation: "
            f"MAE={fold_metrics['MAE']:.3f}, RMSE={fold_metrics['RMSE']:.3f}, R2={fold_metrics['R2']:.3f}"
        )

    results_test_df = pd.DataFrame(
        {
            model_name: {
                **regression_metrics(np.array(all_results_test[model_name]["targets"]), np.array(all_results_test[model_name]["preds"])),
                "Pearson r": np.corrcoef(
                    np.array(all_results_test[model_name]["targets"]),
                    np.array(all_results_test[model_name]["preds"]),
                )[0, 1],
                "Pearson p": np.nan,
            }
        }
    ).T
    ci_df = pd.DataFrame(
        safe_subject_bootstrap_ci(
            all_results_test[model_name]["targets"],
            all_results_test[model_name]["preds"],
            all_results_test[model_name]["subjects"],
        )
    )
    print("\n=== Validation Performance (across all folds) ===")
    print(pd.DataFrame({model_name: {
        **regression_metrics(np.array(all_results_val[model_name]["targets"]), np.array(all_results_val[model_name]["preds"])),
        "Pearson r": np.corrcoef(np.array(all_results_val[model_name]["targets"]), np.array(all_results_val[model_name]["preds"]))[0,1],
        "Pearson p": np.nan,
    }}).T)
    print("\n=== Test Performance (across all folds) ===")
    print(results_test_df)
    print(ci_df)
    return results_test_df, ci_df


def run_classification_cv(args, x_all, y_lesion, participant_ids, device):
    print(f"\n[Classification] {args.model.upper()}")
    model_name = args.model.upper() if args.model != "stgcn" else "STGCN"
    all_results_val = {model_name: {"targets": [], "preds": []}}
    all_results_test = {model_name: {"targets": [], "preds": [], "subjects": []}}
    for fold_idx in tqdm(range(args.n_folds), total=args.n_folds, desc=f"{model_name} classification"):
        train_idx, val_idx, test_idx = split_fold(participant_ids, fold_idx)
        model, scaler, starts = train_classification_fold(
            args,
            x_all[train_idx],
            y_lesion[train_idx],
            x_all[val_idx],
            y_lesion[val_idx],
            args.seed + 1000 + fold_idx,
            device,
            x_all.shape[1],
        )
        val_preds = predict_subject_classification(
            model, x_all[val_idx], scaler, starts, args.window_size, args.batch_size, device, args.model
        )
        test_preds = predict_subject_classification(
            model, x_all[test_idx], scaler, starts, args.window_size, args.batch_size, device, args.model
        )
        all_results_val[model_name]["targets"].extend(y_lesion[val_idx].tolist())
        all_results_val[model_name]["preds"].extend(val_preds.tolist())
        all_results_test[model_name]["targets"].extend(y_lesion[test_idx].tolist())
        all_results_test[model_name]["preds"].extend(test_preds.tolist())
        all_results_test[model_name]["subjects"].extend(participant_ids[test_idx].tolist())
        acc_val = accuracy_score(y_lesion[val_idx], val_preds)
        f1_val = f1_score(y_lesion[val_idx], val_preds, average="macro", zero_division=0)
        print(f"Fold {fold_idx + 1:02d} | {model_name} | Validation: Accuracy={acc_val:.3f}, F1-macro={f1_val:.3f}")

    results_test_df = pd.DataFrame(
        {
            model_name: classification_metrics(
                np.array(all_results_test[model_name]["targets"]),
                np.array(all_results_test[model_name]["preds"]),
            )
        }
    ).T
    ci_df = pd.DataFrame(
        safe_subject_bootstrap_ci_class(
            all_results_test[model_name]["targets"],
            all_results_test[model_name]["preds"],
            all_results_test[model_name]["subjects"],
        )
    )
    print("\n=== Validation Performance (across all folds) ===")
    print(pd.DataFrame({model_name: classification_metrics(
        np.array(all_results_val[model_name]["targets"]), np.array(all_results_val[model_name]["preds"])
    )}).T)
    print("\n=== Test Performance (across all folds) ===")
    print(results_test_df)
    print(ci_df)
    return results_test_df, ci_df


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = get_device(args.device)
    x_all, y_poma, y_lesion, participant_ids = load_dataset(args.tslen)

    print(f"Device: {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(torch.cuda.get_device_name(idx))
    print(f"Tangent data shape: {x_all.shape}")
    print(f"Input normalization: {'on' if args.normalize_input else 'off'}")
    print(f"Classification balancing: {'off' if args.no_clf_balancing else 'on'}")
    print(f"Model: {args.model}")

    t0 = time.time()
    reg_results_df, reg_ci_df = run_regression_cv(args, x_all, y_poma, participant_ids, device)
    clf_results_df, clf_ci_df = run_classification_cv(args, x_all, y_lesion, participant_ids, device)
    summary_path, summary_df = save_model_summary(
        Path(args.output_dir),
        args.model,
        reg_results_df=reg_results_df,
        reg_ci_df=reg_ci_df,
        clf_results_df=clf_results_df,
        clf_ci_df=clf_ci_df,
    )
    print(f"\nSaved summary: {summary_path}")
    print(summary_df)
    print(f"\nTotal runtime: {(time.time() - t0) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
