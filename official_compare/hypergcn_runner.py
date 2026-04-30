from __future__ import annotations

import argparse
import copy
import math
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
    NUM_NODES,
    POMA_MAX,
    POMA_MIN,
    RESULTS_DIR,
    RawStandardizer,
    TangentStandardizer,
    classification_metrics,
    evaluate_classification,
    evaluate_regression,
    hyper_adjacency,
    load_raw_subjects,
    load_tangent_subjects,
    raw_subject_clips,
    regression_metrics,
    save_json,
    set_deterministic,
    split_fold,
    tangent_subject_clips,
)


class DivergenceLoss(nn.Module):
    """Official Hyper-GCN divergence regularizer on virtual hyper-joints."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, hyper_joints: list[torch.Tensor]) -> torch.Tensor:
        vertex_num, _ = hyper_joints[0].size()
        loss = hyper_joints[0].new_tensor(0.0)
        for item in hyper_joints:
            norm = torch.norm(item, dim=-1, keepdim=True, p=2)
            norm = norm @ norm.T
            loss_i = item @ item.T
            loss_i = loss_i / (norm + 1e-8)
            loss_p = self.relu(loss_i)
            loss_p = (loss_p.sum() - vertex_num) / (vertex_num * (vertex_num - 1))
            loss = loss + loss_p
        return loss / len(hyper_joints)


def conv_init(conv: nn.Module) -> None:
    if getattr(conv, "weight", None) is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if getattr(conv, "bias", None) is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.Module, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if hasattr(module, "bias") and module.bias is not None and isinstance(module.bias, torch.Tensor):
            nn.init.constant_(module.bias, 0)
    elif "BatchNorm" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.normal_(1.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(0)


class HyperGC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vertex_nums: int,
        virtual_num: int,
        adjacency: np.ndarray,
        hyper: bool = True,
        num_subset: int = 8,
        rel_reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.virtual_num = virtual_num
        self.rel_reduction = rel_reduction
        self.num_subset = num_subset
        self.hyper = hyper
        self.mid_in_channels = in_channels // num_subset
        self.mid_out_channels = out_channels // num_subset

        if self.hyper:
            self.hidden_channels = max(1, self.mid_in_channels // rel_reduction)
            self.to_v = nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset)
            self.to_w = nn.Sequential(
                nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset),
                nn.LeakyReLU(),
                nn.Conv1d(num_subset * self.hidden_channels, num_subset, kernel_size=1),
                nn.Tanh(),
            )
            self.hyper_joint = nn.Parameter(torch.randn(virtual_num, in_channels))
            self.alpha = nn.Parameter(torch.ones(1))
            self.softmax = nn.Softmax(dim=-1)

        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)
        self.register_buffer("adjacency", torch.from_numpy(adjacency.astype(np.float32)))
        self.edge_importance = nn.Parameter(torch.ones_like(self.adjacency))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)
        bn_init(self.bn, 1e-6)
        if self.hyper:
            conv_init(self.to_v)
            conv_init(self.to_w[0])
            conv_init(self.to_w[2])
        conv_init(self.conv_d)

    def hyper_norm(self, incidence: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        diag_weights = torch.diag_embed(weights)
        norm_w = torch.norm(incidence, 1, dim=2, keepdim=True) + 1e-8
        diag_weights = diag_weights / norm_w
        incidence_weighted = incidence @ torch.diag_embed(weights)
        norm_v = torch.norm(incidence_weighted, 1, dim=3, keepdim=True) + 1e-8
        incidence_weighted = incidence_weighted / norm_v
        return incidence_weighted @ diag_weights @ incidence.transpose(3, 2)

    def a_norm(self, adjacency: torch.Tensor) -> torch.Tensor:
        degree = torch.norm(adjacency, 1, dim=2, keepdim=True) + 1e-8
        return adjacency / degree

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, c, t, v = x.size()
        h_x = self.hyper_joint
        h_x_t = h_x.t().unsqueeze(1)
        x = torch.cat([x, h_x_t.repeat(n, 1, t, 1)], dim=-1)
        v_total = v + self.virtual_num
        adjacency = self.a_norm(self.edge_importance * self.adjacency.to(device=x.device))

        if self.hyper:
            pooled = x.mean(2)
            value_proj = self.to_v(pooled)
            value_proj = value_proj.view(n, self.num_subset, self.hidden_channels, v_total).permute(0, 1, 3, 2).contiguous()
            distance = torch.cdist(value_proj, value_proj)
            incidence = torch.zeros_like(distance)
            topk = min(9, v_total)
            topk_vals, topk_indices = torch.topk(distance, topk, largest=False)
            topk_vals = self.softmax(-topk_vals)
            incidence = torch.scatter(incidence, 3, topk_indices, topk_vals)
            weights = self.to_w(pooled)
            hyper_adj = self.hyper_norm(incidence, weights)
            adjacency = adjacency + self.relu(self.alpha) * hyper_adj

        dense = self.conv_d(x)
        dense = dense.view(n, self.num_subset, self.mid_out_channels, t, v_total)
        y = torch.einsum("nkuv,nkctv->nkctu", adjacency, dense).contiguous()
        y = y.view(n, self.out_channels, t, v_total)
        x = x[..., : self.vertex_nums]
        y = y[..., : self.vertex_nums]
        if self.down is not None:
            residual = self.down(x)
        else:
            residual = x
        y = self.bn(y)
        y = self.relu(y + residual)
        return y, self.hyper_joint


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode="replicate",
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dilations: list[int] | None = None,
        residual: bool = True,
        residual_kernel_size: int = 1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]
        assert out_channels % (len(dilations) + 2) == 0
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
            kernels = kernel_size
        else:
            kernels = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernels, dilations):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
                )
            )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual is None else self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        if self.residual is None:
            return out
        return out + residual


class UnitTCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            padding_mode="replicate",
        )
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class TCNGCNUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_point: int,
        hyper_joints: int,
        adjacency: np.ndarray,
        stride: int = 1,
        residual: bool = True,
        kernel_size: int = 5,
        dilations: list[int] | None = None,
        hyper: bool = True,
    ):
        super().__init__()
        self.gcn1 = HyperGC(in_channels, out_channels, num_point, hyper_joints, adjacency, hyper=hyper)
        self.tcn1 = MultiScaleTemporalConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations if dilations is not None else [1, 2],
            residual=False,
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, h_x = self.gcn1(x)
        residual = x if self.residual is None else self.residual(x)
        y = self.relu(self.tcn1(y) + residual)
        return y, h_x


class HyperGCNBackbone(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_point: int = NUM_NODES,
        num_person: int = 1,
        in_channels: int = NUM_CHANNELS,
        hyper_joints: int = 3,
        drop_out: float = 0.0,
    ):
        super().__init__()
        adjacency = hyper_adjacency(hyper_joints=hyper_joints, nums=8)
        self.num_point = num_point
        self.num_person = num_person
        self.embedding_channels = 128

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, self.embedding_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, self.embedding_channels))
        self.tanh = nn.Tanh()

        self.l1 = TCNGCNUnit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, adjacency)
        self.l2 = TCNGCNUnit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, adjacency)
        self.l3 = TCNGCNUnit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, adjacency)
        self.l4 = TCNGCNUnit(self.embedding_channels, self.embedding_channels * 2, num_point, hyper_joints, adjacency, stride=2)
        self.l5 = TCNGCNUnit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, adjacency)
        self.l6 = TCNGCNUnit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, adjacency)
        self.l7 = TCNGCNUnit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, adjacency, stride=2)
        self.l8 = TCNGCNUnit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, adjacency)
        self.l9 = TCNGCNUnit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, adjacency)
        self.fc = nn.Linear(self.embedding_channels * 2, out_dim)
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / max(out_dim, 1)))
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
        bn_init(self.data_bn, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, : self.num_point]
        x = self.tanh(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(n, m * v * self.embedding_channels, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, self.embedding_channels, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, self.embedding_channels, t, v)

        x, h_x1 = self.l1(x)
        x1 = x
        x, h_x2 = self.l2(x)
        x, h_x3 = self.l3(x + x1)

        x, h_x4 = self.l4(x)
        x4 = x
        x, h_x5 = self.l5(x)
        x, h_x6 = self.l6(x + x4)

        x, h_x7 = self.l7(x)
        x7 = x
        x, h_x8 = self.l8(x)
        x, h_x9 = self.l9(x + x7)

        c_new = x.size(1)
        x = x.view(n, m, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x), [h_x1, h_x2, h_x3, h_x4, h_x5, h_x6, h_x7, h_x8, h_x9]


class HyperGCNStroke(nn.Module):
    def __init__(self, task: str, num_classes: int = 3, hyper_joints: int = 3):
        super().__init__()
        self.task = task
        out_dim = 1 if task == "regression" else num_classes
        self.backbone = HyperGCNBackbone(out_dim=out_dim, hyper_joints=hyper_joints)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = x.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()  # (N, C, T, V, M=1)
        logits, hyper_joints = self.backbone(x)
        if self.task == "regression":
            return logits.squeeze(1), hyper_joints
        return logits, hyper_joints


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


def fit_standardizer(representation: str, sequences):
    if representation == "tangent":
        return TangentStandardizer().fit(np.asarray(sequences, dtype=np.float32))
    return RawStandardizer().fit(sequences)


def subject_sequence(
    representation: str,
    seq: np.ndarray,
    window_size: int,
    gait_window: int,
) -> np.ndarray:
    if representation == "tangent":
        if seq.shape[0] < window_size:
            raise ValueError(f"Need at least {window_size} frames, got {seq.shape[0]}")
        return seq[None].astype(np.float32)

    num_gaits = seq.shape[0]
    if num_gaits < gait_window:
        raise ValueError(f"Need at least {gait_window} gaits, got {num_gaits}")
    start = max(0, (num_gaits - gait_window) // 2)
    clip = seq[start : start + gait_window].reshape(gait_window * seq.shape[1], NUM_NODES, NUM_CHANNELS)
    return clip[None].astype(np.float32)


def make_train_windows(
    representation: str,
    sequences,
    labels: np.ndarray,
    standardizer,
    window_size: int,
    window_stride: int,
    gait_window: int,
    multi_clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_targets = []
    for seq, label in zip(sequences, labels):
        seq = standardizer.transform_sequence(seq)
        if not multi_clip:
            clips = subject_sequence(representation, seq, window_size=window_size, gait_window=gait_window)
        elif representation == "tangent":
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
    model: HyperGCNStroke,
    representation: str,
    task: str,
    sequences,
    standardizer,
    device: torch.device,
    batch_size: int,
    window_size: int,
    window_stride: int,
    gait_window: int,
    multi_clip: bool,
    y_mean: float | None = None,
    y_std: float | None = None,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in sequences:
            seq = standardizer.transform_sequence(seq)
            if not multi_clip:
                clips = subject_sequence(representation, seq, window_size=window_size, gait_window=gait_window)
            elif representation == "tangent":
                clips = tangent_subject_clips(seq, window_size=window_size, stride=window_stride)
            else:
                clips = raw_subject_clips(seq, gait_window=gait_window)
            outputs = []
            for start in range(0, len(clips), batch_size):
                x_batch = torch.from_numpy(clips[start : start + batch_size]).to(device=device, dtype=torch.float32)
                output, _ = model(x_batch)
                outputs.append(output.cpu().numpy())
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
) -> tuple[HyperGCNStroke, object, float | None, float | None]:
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
        args.multi_clip,
    )
    model = HyperGCNStroke(
        task=task,
        num_classes=int(train_labels.max()) + 1 if task == "classification" else 3,
        hyper_joints=args.hyper_joints,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    divergence = DivergenceLoss().to(device)
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
                logits, hyper_joints = model(x_batch)
                loss = criterion(logits, y_batch.to(device).long())
            else:
                targets = ((y_batch.to(device) - y_mean) / y_std).float()
                preds, hyper_joints = model(x_batch)
                loss = criterion(preds, targets)
            loss = loss + args.divergence_weight * divergence(hyper_joints)
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
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            gait_window=args.gait_window,
            multi_clip=args.multi_clip,
            y_mean=y_mean,
            y_std=y_std,
        )
        if task == "classification":
            current_value = classification_metrics(val_labels.astype(np.int64), val_preds)["F1 (macro)"]
            improved = current_value > best_value + 1e-6
        else:
            current_value = regression_metrics(val_labels.astype(np.float32), val_preds)["MAE"]
            improved = current_value < best_value - 1e-6
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
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            gait_window=args.gait_window,
            multi_clip=args.multi_clip,
            y_mean=y_mean,
            y_std=y_std,
        )
        all_targets.extend(test_labels.tolist())
        all_preds.extend(test_preds.tolist())
        all_subjects.extend(participant_ids[test_idx].tolist())
        if args.task == "classification":
            fold_metric = classification_metrics(test_labels.astype(np.int64), test_preds)["F1 (macro)"]
            print(f"[Hyper-GCN][{args.representation}][classification] fold={fold_idx} f1={fold_metric:.4f}")
        else:
            fold_metric = regression_metrics(test_labels.astype(np.float32), test_preds)["MAE"]
            print(f"[Hyper-GCN][{args.representation}][regression] fold={fold_idx} mae={fold_metric:.4f}")
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
    out_path = RESULTS_DIR / f"hypergcn_{args.representation}_{args.task}.json"
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--divergence-weight", type=float, default=1.0)
    parser.add_argument("--hyper-joints", type=int, default=3)
    parser.add_argument("--no-clf-balancing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--multi-clip", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    out_path = run_cv(args)
    print(out_path)


if __name__ == "__main__":
    main()
