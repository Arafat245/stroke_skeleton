#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import pickle
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
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TV_DIR = Path(__file__).resolve().parents[1]
if str(TV_DIR) not in sys.path:
    sys.path.insert(0, str(TV_DIR))

from ci import subject_bootstrap_ci
from ci_class import subject_bootstrap_ci_class
from print_results import print_results_clf, print_results_regression
from val_test import val_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="TCN baseline on tangent vectors for regression and classification."
    )
    parser.add_argument("--tslen", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--channels", type=str, default="64,64,96")
    parser.add_argument("--clf-channels", type=str, default="32,32")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--clf-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--clf-dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-reg", type=int, default=140)
    parser.add_argument("--epochs-clf", type=int, default=160)
    parser.add_argument("--patience-reg", type=int, default=30)
    parser.add_argument("--patience-clf", type=int, default=35)
    parser.add_argument("--lr-reg", type=float, default=5e-4)
    parser.add_argument("--lr-clf", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--no-clf-balancing", action="store_true")
    parser.add_argument("--use-val-early-stopping", action="store_true")
    parser.add_argument("--n-folds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_dataset(tslen):
    participant_ids = np.loadtxt(ROOT / "labels_data" / "pids.txt").astype(int)
    y_poma = np.loadtxt(ROOT / "labels_data" / "y_poma.txt").astype(np.float32)

    demo_df = pd.read_csv(ROOT / "labels_data" / "demo_data.csv")
    id_to_lesion = dict(
        zip(demo_df["s"].astype(int).tolist(), demo_df["LesionLeft"].astype(int).tolist())
    )
    y_lesion = np.array([id_to_lesion[int(pid)] for pid in participant_ids], dtype=np.int64)

    tangent_path = ROOT / "aligned_data" / f"tangent_vecs{tslen}.pkl"
    with open(tangent_path, "rb") as handle:
        tangent_vec_all = np.asarray(pickle.load(handle), dtype=np.float32)

    x_all = tangent_vec_all.reshape(-1, tslen, tangent_vec_all.shape[-1]).transpose(2, 0, 1)
    return x_all, y_poma, y_lesion, participant_ids


class ChannelStandardizer:
    def fit(self, x):
        self.mean = x.mean(axis=(0, 2), keepdims=True)
        self.std = x.std(axis=(0, 2), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)


class IdentityStandardizer:
    def fit(self, x):
        return self

    def transform(self, x):
        return x.astype(np.float32)


def build_input_standardizer(normalize_input):
    if normalize_input:
        return ChannelStandardizer()
    return IdentityStandardizer()


def sliding_starts(seq_len, window_size, stride):
    if window_size > seq_len:
        raise ValueError(f"window_size={window_size} is larger than seq_len={seq_len}")
    starts = list(range(0, seq_len - window_size + 1, stride))
    last_start = seq_len - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


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
            windows.append(x[:, start : start + window_size])
            owners.append(subject_idx)
    return np.stack(windows).astype(np.float32), np.asarray(owners, dtype=np.int64)


def build_regression_loader(x_subjects, y_subjects, starts, window_size, batch_size, seed):
    windows, owners = make_subject_windows(x_subjects, starts, window_size)
    targets = y_subjects[owners]
    dataset = WindowDataset(windows, targets)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_classification_loader(
    x_subjects, y_subjects, starts, window_size, batch_size, seed, use_weighted_sampler=True
):
    windows, owners = make_subject_windows(x_subjects, starts, window_size)
    targets = y_subjects[owners]
    dataset = WindowDataset(windows, targets)

    if use_weighted_sampler:
        class_counts = np.bincount(targets, minlength=int(targets.max()) + 1).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / np.sqrt(class_counts)
        sample_weights = class_weights[targets]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class TCNBackbone(nn.Module):
    def __init__(self, input_dim, channels, kernel_size, hidden_dim, dropout):
        super().__init__()
        blocks = []
        for level, out_channels in enumerate(channels):
            in_channels = input_dim if level == 0 else channels[level - 1]
            blocks.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1] * 2),
            nn.Linear(channels[-1] * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        feat = self.network(x)
        pooled_avg = feat.mean(dim=-1)
        pooled_max = feat.max(dim=-1).values
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)
        return self.head(pooled)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim, channels, kernel_size, hidden_dim, dropout):
        super().__init__()
        self.backbone = TCNBackbone(input_dim, channels, kernel_size, hidden_dim, dropout)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.reg_head(self.backbone(x)).squeeze(1)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels, kernel_size, hidden_dim, dropout, num_classes):
        super().__init__()
        self.backbone = TCNBackbone(input_dim, channels, kernel_size, hidden_dim, dropout)
        self.cls_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.cls_head(self.backbone(x))


def predict_regression(model, x_subjects, scaler, starts, window_size, batch_size, device, y_mean, y_std):
    model.eval()
    x_subjects = scaler.transform(x_subjects)
    preds = []

    with torch.no_grad():
        for x in x_subjects:
            windows = np.stack([x[:, start : start + window_size] for start in starts]).astype(np.float32)
            window_preds = []
            for batch_start in range(0, len(windows), batch_size):
                x_batch = torch.from_numpy(windows[batch_start : batch_start + batch_size]).to(device)
                window_preds.append(model(x_batch).cpu().numpy())
            window_preds = np.concatenate(window_preds)
            subject_pred = np.median(window_preds) * y_std + y_mean
            preds.append(subject_pred)

    return np.clip(np.asarray(preds, dtype=np.float32), 6.0, 28.0)


def predict_classification(model, x_subjects, scaler, starts, window_size, batch_size, device):
    model.eval()
    x_subjects = scaler.transform(x_subjects)
    preds = []

    with torch.no_grad():
        for x in x_subjects:
            windows = np.stack([x[:, start : start + window_size] for start in starts]).astype(np.float32)
            prob_chunks = []
            for batch_start in range(0, len(windows), batch_size):
                x_batch = torch.from_numpy(windows[batch_start : batch_start + batch_size]).to(device)
                logits = model(x_batch)
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
        "Precision (weighted)": precision_score(
            targets, preds, average="weighted", zero_division=0
        ),
        "Precision (macro)": precision_score(targets, preds, average="macro", zero_division=0),
        "Recall (weighted)": recall_score(targets, preds, average="weighted", zero_division=0),
        "Recall (macro)": recall_score(targets, preds, average="macro", zero_division=0),
    }


def train_regression_fold(args, x_train, y_train, x_val, y_val, fold_seed, device, input_dim):
    set_deterministic(fold_seed)
    scaler = build_input_standardizer(args.normalize_input).fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    starts = sliding_starts(x_train.shape[-1], args.window_size, args.window_stride)
    train_loader = build_regression_loader(
        x_train_scaled,
        y_train,
        starts,
        args.window_size,
        args.batch_size,
        fold_seed,
    )

    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))
    model = TCNRegressor(
        input_dim=input_dim,
        channels=[int(ch) for ch in args.channels.split(",")],
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr_reg, weight_decay=args.weight_decay
    )
    if args.use_val_early_stopping:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs_reg, 1)
        )
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_state = None
    best_val_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs_reg):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = ((y_batch.to(device) - y_mean) / y_std).float()

            if args.input_noise_std > 0:
                x_batch = x_batch + args.input_noise_std * torch.randn_like(x_batch)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if args.use_val_early_stopping:
            val_preds = predict_regression(
                model,
                x_val,
                scaler,
                starts,
                args.window_size,
                args.batch_size,
                device,
                y_mean,
                y_std,
            )
            val_mae = mean_absolute_error(y_val, val_preds)
            scheduler.step(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience_reg:
                    break
        else:
            scheduler.step()

    if args.use_val_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

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
        use_weighted_sampler=not args.no_clf_balancing,
    )

    class_weights = None
    if not args.no_clf_balancing:
        class_counts = np.bincount(y_train, minlength=int(y_train.max()) + 1).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = TCNClassifier(
        input_dim=input_dim,
        channels=[int(ch) for ch in args.clf_channels.split(",")],
        kernel_size=args.kernel_size,
        hidden_dim=args.clf_hidden_dim,
        dropout=args.clf_dropout,
        num_classes=int(y_train.max()) + 1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr_clf, weight_decay=args.weight_decay
    )
    if args.use_val_early_stopping:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-5
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs_clf, 1)
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    best_state = None
    best_val_f1 = -float("inf")
    best_val_acc = -float("inf")
    epochs_no_improve = 0

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

        if args.use_val_early_stopping:
            val_preds = predict_classification(
                model,
                x_val,
                scaler,
                starts,
                args.window_size,
                args.batch_size,
                device,
            )
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)
            scheduler.step(val_f1)

            improved = (val_f1 > best_val_f1) or (
                np.isclose(val_f1, best_val_f1) and val_acc > best_val_acc
            )
            if improved:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience_clf:
                    break
        else:
            scheduler.step()

    if args.use_val_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, starts


def split_fold(participant_ids, fold_idx):
    val_pids, test_pids = val_test(participant_ids, fold_idx)
    val_pids = set(int(pid) for pid in val_pids)
    test_pids = set(int(pid) for pid in test_pids)
    train_pids = set(int(pid) for pid in participant_ids) - val_pids - test_pids

    train_idx = np.array([i for i, pid in enumerate(participant_ids) if int(pid) in train_pids])
    val_idx = np.array([i for i, pid in enumerate(participant_ids) if int(pid) in val_pids])
    test_idx = np.array([i for i, pid in enumerate(participant_ids) if int(pid) in test_pids])
    return train_idx, val_idx, test_idx


def run_regression_cv(args, x_all, y_poma, participant_ids, device):
    print("\n[Regression] Running TCN CV")
    input_dim = x_all.shape[1]
    model_name = "TCN"
    all_results_val = {model_name: {"targets": [], "preds": []}}
    all_results_test = {model_name: {"targets": [], "preds": [], "subjects": []}}

    fold_iter = tqdm(range(args.n_folds), total=args.n_folds, desc="Regression folds")
    for fold_idx in fold_iter:
        train_idx, val_idx, test_idx = split_fold(participant_ids, fold_idx)
        fold_seed = args.seed + fold_idx

        model, scaler, starts, y_mean, y_std = train_regression_fold(
            args=args,
            x_train=x_all[train_idx],
            y_train=y_poma[train_idx],
            x_val=x_all[val_idx],
            y_val=y_poma[val_idx],
            fold_seed=fold_seed,
            device=device,
            input_dim=input_dim,
        )

        val_preds = predict_regression(
            model,
            x_all[val_idx],
            scaler,
            starts,
            args.window_size,
            args.batch_size,
            device,
            y_mean,
            y_std,
        )
        test_preds = predict_regression(
            model,
            x_all[test_idx],
            scaler,
            starts,
            args.window_size,
            args.batch_size,
            device,
            y_mean,
            y_std,
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

    results_test_df = print_results_regression(all_results_val, all_results_test, {model_name: None})
    ci_df = pd.DataFrame(
        subject_bootstrap_ci(
            all_results_test[model_name]["targets"],
            all_results_test[model_name]["preds"],
            all_results_test[model_name]["subjects"],
        )
    )
    print(results_test_df)
    print(ci_df)
    return results_test_df, ci_df, all_results_val, all_results_test


def run_classification_cv(args, x_all, y_lesion, participant_ids, device):
    print("\n[Classification] Running TCN CV")
    input_dim = x_all.shape[1]
    model_name = "TCN"
    all_results_val = {model_name: {"targets": [], "preds": []}}
    all_results_test = {model_name: {"targets": [], "preds": [], "subjects": []}}

    fold_iter = tqdm(range(args.n_folds), total=args.n_folds, desc="Classification folds")
    for fold_idx in fold_iter:
        train_idx, val_idx, test_idx = split_fold(participant_ids, fold_idx)
        fold_seed = args.seed + 1000 + fold_idx

        model, scaler, starts = train_classification_fold(
            args=args,
            x_train=x_all[train_idx],
            y_train=y_lesion[train_idx],
            x_val=x_all[val_idx],
            y_val=y_lesion[val_idx],
            fold_seed=fold_seed,
            device=device,
            input_dim=input_dim,
        )

        val_preds = predict_classification(
            model,
            x_all[val_idx],
            scaler,
            starts,
            args.window_size,
            args.batch_size,
            device,
        )
        test_preds = predict_classification(
            model,
            x_all[test_idx],
            scaler,
            starts,
            args.window_size,
            args.batch_size,
            device,
        )

        all_results_val[model_name]["targets"].extend(y_lesion[val_idx].tolist())
        all_results_val[model_name]["preds"].extend(val_preds.tolist())
        all_results_test[model_name]["targets"].extend(y_lesion[test_idx].tolist())
        all_results_test[model_name]["preds"].extend(test_preds.tolist())
        all_results_test[model_name]["subjects"].extend(participant_ids[test_idx].tolist())

        acc_val = accuracy_score(y_lesion[val_idx], val_preds)
        f1_val = f1_score(y_lesion[val_idx], val_preds, average="macro", zero_division=0)
        print(
            f"Fold {fold_idx + 1:02d} | {model_name} | Validation: "
            f"Accuracy={acc_val:.3f}, F1-macro={f1_val:.3f}"
        )

    results_test_df = print_results_clf(all_results_val, all_results_test, {model_name: None})
    ci_df = pd.DataFrame(
        subject_bootstrap_ci_class(
            all_results_test[model_name]["targets"],
            all_results_test[model_name]["preds"],
            all_results_test[model_name]["subjects"],
        )
    )
    print(results_test_df)
    print(ci_df)
    return results_test_df, ci_df, all_results_val, all_results_test


def _parse_ci_bounds(ci_value):
    if isinstance(ci_value, str):
        stripped = ci_value.strip().strip("[]")
        vals = np.fromstring(stripped, sep=" ")
        if vals.size == 2:
            return float(vals[0]), float(vals[1])
    if isinstance(ci_value, (list, tuple, np.ndarray)) and len(ci_value) == 2:
        return float(ci_value[0]), float(ci_value[1])
    return np.nan, np.nan


def build_summary_rows(task_name, results_df, ci_df):
    model_row = results_df.loc["TCN"]
    ci_mean_row = ci_df.loc["mean"] if "mean" in ci_df.index else pd.Series(dtype=float)
    ci_row = ci_df.loc["ci"] if "ci" in ci_df.index else pd.Series(dtype=object)

    rows = []
    for metric, value in model_row.items():
        ci_low, ci_high = _parse_ci_bounds(ci_row.get(metric)) if metric in ci_row.index else (np.nan, np.nan)
        ci_mean = ci_mean_row.get(metric, np.nan)
        rows.append(
            {
                "task": task_name,
                "metric": metric,
                "value": float(value),
                "ci_mean": float(ci_mean) if pd.notna(ci_mean) else np.nan,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }
        )
    return rows


def save_summary_csv(output_dir, reg_results_df=None, reg_ci_df=None, clf_results_df=None, clf_ci_df=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "tcn_results_with_ci.csv"
    rows = []
    if reg_results_df is not None and reg_ci_df is not None:
        rows.extend(build_summary_rows("regression", reg_results_df, reg_ci_df))
    if clf_results_df is not None and clf_ci_df is not None:
        rows.extend(build_summary_rows("classification", clf_results_df, clf_ci_df))

    updated_tasks = {row["task"] for row in rows}
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        if updated_tasks:
            existing_df = existing_df[~existing_df["task"].isin(updated_tasks)]
        summary_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
    else:
        summary_df = pd.DataFrame(rows)

    for old_csv in output_dir.glob("*.csv"):
        if old_csv != summary_path:
            old_csv.unlink()

    if "task" in summary_df.columns and "metric" in summary_df.columns:
        task_order = {"regression": 0, "classification": 1}
        summary_df = summary_df.sort_values(
            by=["task", "metric"],
            key=lambda col: col.map(task_order) if col.name == "task" else col,
        ).reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_path, summary_df


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)

    x_all, y_poma, y_lesion, participant_ids = load_dataset(args.tslen)
    print(f"Device: {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(torch.cuda.get_device_name(idx))
    print(f"Tangent data shape: {x_all.shape}")
    print(f"POMA range: {y_poma.min()} to {y_poma.max()}")
    print(f"Lesion class counts: {np.bincount(y_lesion)}")
    print(f"Input normalization: {'on' if args.normalize_input else 'off'}")
    print(f"Classification balancing: {'off' if args.no_clf_balancing else 'on'}")
    print(f"Classification width: channels={args.clf_channels}, hidden_dim={args.clf_hidden_dim}, dropout={args.clf_dropout}")
    print(f"Validation early stopping: {'on' if args.use_val_early_stopping else 'off'}")

    t0 = time.time()
    reg_results_df = reg_ci_df = None
    clf_results_df = clf_ci_df = None

    if not args.skip_regression:
        reg_results_df, reg_ci_df, _, _ = run_regression_cv(
            args, x_all, y_poma, participant_ids, device
        )

    if not args.skip_classification:
        clf_results_df, clf_ci_df, _, _ = run_classification_cv(
            args, x_all, y_lesion, participant_ids, device
        )

    summary_path, summary_df = save_summary_csv(
        output_dir,
        reg_results_df=reg_results_df,
        reg_ci_df=reg_ci_df,
        clf_results_df=clf_results_df,
        clf_ci_df=clf_ci_df,
    )
    print(f"\nSaved summary: {summary_path}")
    print(summary_df)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
