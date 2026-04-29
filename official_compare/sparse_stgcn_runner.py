from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from official_compare.common import (
    NUM_CHANNELS,
    POMA_MAX,
    POMA_MIN,
    RESULTS_DIR,
    RawStandardizer,
    TangentStandardizer,
    classification_metrics,
    evaluate_classification,
    evaluate_regression,
    load_raw_subjects,
    load_tangent_subjects,
    raw_subject_clips,
    regression_metrics,
    save_json,
    set_deterministic,
    sparse_adjacency,
    split_fold,
    tangent_subject_clips,
)


EPS = 1e-4


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        return torch.where(scores < threshold, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class SparseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        conv_sparsity: float = 0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sparsity = conv_sparsity
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.weight.is_mask = True
        self.weight_score = nn.Parameter(torch.empty_like(self.weight))
        self.weight_score.is_score = True
        self.weight_score.sparsity = conv_sparsity
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.weight_score, nonlinearity="relu")
        self.register_buffer("zeros", torch.zeros_like(self.weight_score))
        self.register_buffer("ones", torch.ones_like(self.weight_score))

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        subnet = GetSubnet.apply(self.weight_score, threshold, self.zeros, self.ones)
        return F.conv2d(
            x,
            self.weight * subnet,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class UnitGCNSparse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, sparse_ratio: float = 0.6):
        super().__init__()
        self.num_subsets = A.size(0)
        self.A = nn.Parameter(A.clone())
        self.conv = SparseConv2d(in_channels, out_channels * A.size(0), 1, sparse_ratio)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.conv(x, threshold).view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, self.A).contiguous()
        return self.act(self.bn(x))


class UnitTCNSparse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=9, stride=1, conv_sparsity=0.6):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = SparseConv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            conv_sparsity=conv_sparsity,
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.drop(self.bn(self.conv(x, threshold)))


class STGCNBlockSparse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1, residual: bool = True, sparse_ratio: float = 0.6):
        super().__init__()
        self.gcn = UnitGCNSparse(in_channels, out_channels, A, sparse_ratio=sparse_ratio)
        self.tcn = UnitTCNSparse(out_channels, out_channels, stride=stride, conv_sparsity=sparse_ratio)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x, threshold: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x, threshold: x
        else:
            self.residual = UnitTCNSparse(in_channels, out_channels, kernel_size=1, stride=stride, conv_sparsity=sparse_ratio)

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        res = self.residual(x, threshold) if isinstance(self.residual, nn.Module) else self.residual(x, threshold)
        return self.relu(self.tcn(self.gcn(x, threshold), threshold) + res)


class SparseSTGCNBackbone(nn.Module):
    def __init__(self, sparse_ratio: float = 0.6, warm_up: int = 10):
        super().__init__()
        A = sparse_adjacency()
        self.data_bn = nn.BatchNorm1d(NUM_CHANNELS * A.size(1))
        self.linear_sparsity = sparse_ratio
        self.warm_up = warm_up
        self.base_channels = 64
        modules = [STGCNBlockSparse(NUM_CHANNELS, 64, A.clone(), 1, residual=False, sparse_ratio=sparse_ratio)]
        inflate_times = 0
        current = 64
        for i in range(2, 11):
            stride = 2 if i in [5, 8] else 1
            if i in [5, 8]:
                inflate_times += 1
            out_channels = int(self.base_channels * 2 ** inflate_times + EPS)
            modules.append(STGCNBlockSparse(current, out_channels, A.clone(), stride=stride, sparse_ratio=sparse_ratio))
            current = out_channels
        self.gcn = nn.ModuleList(modules)
        self.out_channels = current

    def percentile(self, t: torch.Tensor, q: float) -> float:
        k = 1 + round(0.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()

    def get_threshold(self, sparsity: float) -> float:
        score_params = []
        for p in self.gcn.parameters():
            if hasattr(p, "is_score") and p.is_score and p.sparsity == self.linear_sparsity:
                score_params.append(p.detach().flatten())
        joined = torch.cat(score_params)
        return self.percentile(joined, sparsity * 100)

    def forward(self, x: torch.Tensor, current_epoch: int) -> torch.Tensor:
        sparsity = 0.0 if current_epoch < self.warm_up else self.linear_sparsity
        n, t, v, c = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        threshold = self.get_threshold(sparsity)
        for block in self.gcn:
            x = block(x, threshold)
        return x


class SparseSTGCNStroke(nn.Module):
    def __init__(self, task: str, num_classes: int = 3, warm_up: int = 10):
        super().__init__()
        self.task = task
        self.backbone = SparseSTGCNBackbone(warm_up=warm_up)
        out_dim = 1 if task == "regression" else num_classes
        self.head = nn.Linear(self.backbone.out_channels, out_dim)

    def forward(self, x: torch.Tensor, current_epoch: int) -> torch.Tensor:
        feat = self.backbone(x, current_epoch=current_epoch)
        pooled = feat.mean(dim=(2, 3))
        out = self.head(pooled)
        if self.task == "regression":
            return out.squeeze(1)
        return out


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.x = torch.from_numpy(windows.astype(np.float32))
        if np.issubdtype(targets.dtype, np.integer):
            self.y = torch.from_numpy(targets.astype(np.int64))
        else:
            self.y = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * epoch / total_epochs))


def make_train_windows(
    representation: str,
    sequences,
    labels: np.ndarray,
    standardizer,
    window_size: int,
    window_stride: int,
    gait_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_targets = []
    for seq, label in zip(sequences, labels):
        seq = standardizer.transform_sequence(seq)
        if representation == "tangent":
            clips = tangent_subject_clips(seq, window_size=window_size, stride=window_stride)
        else:
            clips = raw_subject_clips(seq, gait_window=gait_window)
        all_windows.append(clips)
        all_targets.append(np.full(clips.shape[0], label, dtype=labels.dtype))
    return np.concatenate(all_windows, axis=0), np.concatenate(all_targets, axis=0)


def build_classification_loader(
    windows: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    seed: int,
    use_balancing: bool,
) -> DataLoader:
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
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def build_regression_loader(windows: np.ndarray, targets: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    dataset = WindowDataset(windows, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def predict_subjects(
    model: SparseSTGCNStroke,
    representation: str,
    task: str,
    sequences,
    standardizer,
    device: torch.device,
    current_epoch: int,
    batch_size: int,
    window_size: int,
    window_stride: int,
    gait_window: int,
    y_mean: float | None = None,
    y_std: float | None = None,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in sequences:
            seq = standardizer.transform_sequence(seq)
            if representation == "tangent":
                clips = tangent_subject_clips(seq, window_size=window_size, stride=window_stride)
            else:
                clips = raw_subject_clips(seq, gait_window=gait_window)
            outputs = []
            for start in range(0, len(clips), batch_size):
                x_batch = torch.from_numpy(clips[start : start + batch_size]).to(device=device, dtype=torch.float32)
                outputs.append(model(x_batch, current_epoch=current_epoch).cpu().numpy())
            outputs = np.concatenate(outputs, axis=0)
            if task == "classification":
                probs = torch.softmax(torch.from_numpy(outputs), dim=1).numpy().mean(axis=0)
                preds.append(int(np.argmax(probs)))
            else:
                assert y_mean is not None and y_std is not None
                value = float(np.median(outputs) * y_std + y_mean)
                preds.append(float(np.clip(value, POMA_MIN, POMA_MAX)))
    dtype = np.int64 if task == "classification" else np.float32
    return np.asarray(preds, dtype=dtype)


def fit_standardizer(representation: str, sequences):
    if representation == "tangent":
        return TangentStandardizer().fit(np.asarray(sequences, dtype=np.float32))
    return RawStandardizer().fit(sequences)


def train_fold(
    args: argparse.Namespace,
    representation: str,
    task: str,
    train_sequences,
    train_labels: np.ndarray,
    val_sequences,
    val_labels: np.ndarray,
    fold_seed: int,
    device: torch.device,
) -> tuple[SparseSTGCNStroke, object, float | None, float | None]:
    set_deterministic(fold_seed)
    standardizer = fit_standardizer(representation, train_sequences)
    train_windows, train_targets = make_train_windows(
        representation,
        train_sequences,
        train_labels,
        standardizer,
        args.window_size,
        args.window_stride,
        args.gait_window,
    )
    model = SparseSTGCNStroke(
        task=task,
        num_classes=int(train_labels.max()) + 1 if task == "classification" else 3,
        warm_up=args.warmup if args.warmup is not None else max(1, round(args.epochs * 0.5)),
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    if task == "classification":
        loader = build_classification_loader(
            train_windows,
            train_targets.astype(np.int64),
            args.batch_size,
            fold_seed,
            use_balancing=not args.no_clf_balancing,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        y_mean = None
        y_std = None
        best_value = -np.inf
    else:
        loader = build_regression_loader(
            train_windows,
            train_targets.astype(np.float32),
            args.batch_size,
            fold_seed,
        )
        criterion = nn.SmoothL1Loss(beta=1.0)
        y_mean = float(train_labels.mean())
        y_std = float(max(train_labels.std(), 1e-6))
        best_value = np.inf

    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        optimizer.param_groups[0]["lr"] = cosine_lr(args.lr, epoch, max(args.epochs, 1))
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            if task == "classification":
                logits = model(x_batch, current_epoch=epoch + 1)
                loss = criterion(logits, y_batch.to(device).long())
            else:
                targets = ((y_batch.to(device) - y_mean) / y_std).float()
                preds = model(x_batch, current_epoch=epoch + 1)
                loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_preds = predict_subjects(
            model,
            representation,
            task,
            val_sequences,
            standardizer,
            device,
            current_epoch=epoch + 1,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            gait_window=args.gait_window,
            y_mean=y_mean,
            y_std=y_std,
        )
        if task == "classification":
            score = classification_metrics(val_labels.astype(np.int64), val_preds)["F1 (macro)"]
            improved = score > best_value + 1e-6
            current_value = score
        else:
            score = regression_metrics(val_labels.astype(np.float32), val_preds)["MAE"]
            improved = score < best_value - 1e-6
            current_value = score
        if improved:
            best_value = current_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, standardizer, y_mean, y_std


def run_cv(args: argparse.Namespace) -> Path:
    set_deterministic(args.seed)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.representation == "tangent":
        sequences, y_poma, y_lesion, participant_ids = load_tangent_subjects(tslen=args.tslen)
        labels = y_poma if args.task == "regression" else y_lesion
    else:
        sequences, labels, participant_ids = load_raw_subjects(args.task)

    all_targets = []
    all_preds = []
    all_subjects = []
    per_fold = []

    for fold_idx in range(args.n_folds):
        if args.max_folds is not None and fold_idx >= args.max_folds:
            break
        train_idx, val_idx, test_idx = split_fold(participant_ids, fold_idx)
        train_sequences = sequences[train_idx] if args.representation == "tangent" else [sequences[i] for i in train_idx]
        val_sequences = sequences[val_idx] if args.representation == "tangent" else [sequences[i] for i in val_idx]
        test_sequences = sequences[test_idx] if args.representation == "tangent" else [sequences[i] for i in test_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        model, standardizer, y_mean, y_std = train_fold(
            args,
            args.representation,
            args.task,
            train_sequences,
            train_labels,
            val_sequences,
            val_labels,
            args.seed + fold_idx,
            device,
        )
        test_preds = predict_subjects(
            model,
            args.representation,
            args.task,
            test_sequences,
            standardizer,
            device,
            current_epoch=args.epochs + 1,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            gait_window=args.gait_window,
            y_mean=y_mean,
            y_std=y_std,
        )
        all_targets.extend(test_labels.tolist())
        all_preds.extend(test_preds.tolist())
        all_subjects.extend(participant_ids[test_idx].tolist())
        if args.task == "classification":
            fold_metric = classification_metrics(test_labels.astype(np.int64), test_preds)["F1 (macro)"]
            print(f"[Sparse-ST-GCN][{args.representation}][classification] fold={fold_idx} f1={fold_metric:.4f}")
        else:
            fold_metric = regression_metrics(test_labels.astype(np.float32), test_preds)["MAE"]
            print(f"[Sparse-ST-GCN][{args.representation}][regression] fold={fold_idx} mae={fold_metric:.4f}")
        per_fold.append(
            {
                "fold": fold_idx,
                "val_subjects": participant_ids[val_idx].tolist(),
                "test_subjects": participant_ids[test_idx].tolist(),
            }
        )

    if args.task == "classification":
        payload = evaluate_classification(all_targets, all_preds, all_subjects)
    else:
        payload = evaluate_regression(all_targets, all_preds, all_subjects)
    payload["folds"] = per_fold
    payload["config"] = vars(args)
    out_path = RESULTS_DIR / f"sparse_stgcn_{args.representation}_{args.task}.json"
    save_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["raw", "tangent"], required=True)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--tslen", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--gait-window", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-clf-balancing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()
    out_path = run_cv(args)
    print(out_path)


if __name__ == "__main__":
    main()
