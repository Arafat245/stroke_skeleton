from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Tangent_Vector.ci import subject_bootstrap_ci
from Tangent_Vector.ci_class import subject_bootstrap_ci_class
from Tangent_Vector.val_test import val_test


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "official_compare" / "results"
NUM_NODES = 32
NUM_CHANNELS = 3
POMA_MIN = 6.0
POMA_MAX = 28.0


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stroke_edges() -> list[tuple[int, int]]:
    edges = [
        (4, 5),
        (5, 6),
        (6, 7),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ]
    edges.extend((i, i + 1) for i in range(NUM_NODES - 1))
    dedup = []
    seen = set()
    for edge in edges:
        if edge not in seen:
            dedup.append(edge)
            seen.add(edge)
    return dedup


def stroke_graph_parts() -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    self_link = [(i, i) for i in range(NUM_NODES)]
    inward = _stroke_edges()
    outward = [(j, i) for (i, j) in inward]
    return self_link, inward, outward


def edge2mat(link: list[tuple[int, int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1.0
    return A


def get_spatial_graph(num_node: int, self_link, inward, outward) -> np.ndarray:
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    return np.stack((I, In, Out)).astype(np.float32)


def get_spatial_graph_ensemble(num_node: int, self_link, inward, outward, nums: int) -> np.ndarray:
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A = I + In + Out
    return np.repeat(A[np.newaxis, :], nums, axis=0).astype(np.float32)


def get_virtual_graph_ensemble(
    num_node: int,
    self_link,
    inward,
    outward,
    virtual: int,
    nums: int,
) -> np.ndarray:
    I = np.pad(edge2mat(self_link, num_node), ((0, virtual), (0, virtual)))
    In = np.pad(edge2mat(inward, num_node), ((0, virtual), (0, virtual)))
    Out = np.pad(edge2mat(outward, num_node), ((0, virtual), (0, virtual)))
    for i in range(virtual):
        I[num_node + i, num_node + i] = 1.0
        In[:num_node, num_node + i] = 1.0
        Out[num_node + i, :num_node] = 1.0
    A = I + In + Out
    return np.repeat(A[np.newaxis, :], nums, axis=0).astype(np.float32)


def sparse_adjacency() -> torch.Tensor:
    self_link, inward, outward = stroke_graph_parts()
    return torch.tensor(get_spatial_graph(NUM_NODES, self_link, inward, outward), dtype=torch.float32)


def hyper_adjacency(hyper_joints: int = 3, nums: int = 8) -> np.ndarray:
    self_link, inward, outward = stroke_graph_parts()
    return get_virtual_graph_ensemble(NUM_NODES, self_link, inward, outward, hyper_joints, nums)


def sliding_starts(seq_len: int, window_size: int, stride: int) -> list[int]:
    if window_size > seq_len:
        raise ValueError(f"window_size={window_size} is larger than seq_len={seq_len}")
    starts = list(range(0, seq_len - window_size + 1, stride))
    last_start = seq_len - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def load_tangent_subjects(tslen: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    participant_ids = np.loadtxt(REPO_ROOT / "labels_data" / "pids.txt").astype(np.int64)
    y_poma = np.loadtxt(REPO_ROOT / "labels_data" / "y_poma.txt").astype(np.float32)
    demo_df = pd.read_csv(REPO_ROOT / "labels_data" / "demo_data.csv")
    id_to_lesion = dict(
        zip(demo_df["s"].astype(int).tolist(), demo_df["LesionLeft"].astype(int).tolist())
    )
    y_lesion = np.array([id_to_lesion[int(pid)] for pid in participant_ids], dtype=np.int64)
    with open(REPO_ROOT / "aligned_data" / f"tangent_vecs{tslen}.pkl", "rb") as handle:
        tangent = np.asarray(pickle.load(handle), dtype=np.float32)
    sequences = np.moveaxis(tangent, -1, 0).transpose(0, 3, 1, 2)  # (N, T, V, C)
    return sequences, y_poma, y_lesion, participant_ids


def load_raw_subjects(task: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if task == "regression":
        path = REPO_ROOT / "data" / "processed_loaded.pt"
    elif task == "classification":
        path = REPO_ROOT / "data_clf" / "processed_loaded.pt"
    else:
        raise ValueError(f"Unknown task: {task}")
    loaded = torch.load(path, weights_only=False)
    participant_ids = np.array(sorted((int(pid) for pid in loaded.keys())), dtype=np.int64)
    sequences: list[np.ndarray] = []
    labels = []
    for pid in participant_ids:
        task_dict = loaded[str(pid)]
        task_data = next(iter(task_dict.values()))
        gaits = [np.asarray(gait, dtype=np.float32) for gait in task_data["gaits"].values()]
        gait_array = np.stack(gaits, axis=0).reshape(len(gaits), 100, NUM_NODES, NUM_CHANNELS)
        sequences.append(gait_array.astype(np.float32))
        labels.append(task_data["label"])
    label_dtype = np.float32 if task == "regression" else np.int64
    return sequences, np.asarray(labels, dtype=label_dtype), participant_ids


class TangentStandardizer:
    def fit(self, x: np.ndarray) -> "TangentStandardizer":
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_sequence(self, seq: np.ndarray) -> np.ndarray:
        return self.transform(seq[None])[0]


class RawStandardizer:
    def fit(self, sequences: list[np.ndarray]) -> "RawStandardizer":
        flat = np.concatenate([seq.reshape(-1, NUM_NODES, NUM_CHANNELS) for seq in sequences], axis=0)
        self.mean = flat.mean(axis=0, keepdims=True)
        self.std = flat.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform_sequence(self, seq: np.ndarray) -> np.ndarray:
        return ((seq - self.mean) / self.std).astype(np.float32)


def tangent_subject_clips(seq: np.ndarray, window_size: int = 100, stride: int = 25) -> np.ndarray:
    starts = sliding_starts(seq.shape[0], window_size, stride)
    return np.stack([seq[start : start + window_size] for start in starts], axis=0).astype(np.float32)


def raw_subject_clips(seq: np.ndarray, gait_window: int = 2) -> np.ndarray:
    num_gaits = seq.shape[0]
    if num_gaits < gait_window:
        raise ValueError(f"Need at least {gait_window} gaits, got {num_gaits}")
    clips = []
    for start in range(num_gaits - gait_window + 1):
        clip = seq[start : start + gait_window].reshape(gait_window * seq.shape[1], NUM_NODES, NUM_CHANNELS)
        clips.append(clip.astype(np.float32))
    return np.stack(clips, axis=0).astype(np.float32)


def split_fold(participant_ids: np.ndarray, fold_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    val_pids, test_pids = val_test(participant_ids, fold_idx)
    val_pids = set(int(pid) for pid in val_pids)
    test_pids = set(int(pid) for pid in test_pids)
    train_pids = set(int(pid) for pid in participant_ids) - val_pids - test_pids
    train_idx = np.array([i for i, pid in enumerate(participant_ids) if int(pid) in train_pids], dtype=np.int64)
    val_idx = np.array([i for i, pid in enumerate(participant_ids) if int(pid) in val_pids], dtype=np.int64)
    test_idx = np.array([i for i, pid in enumerate(participant_ids) if int(pid) in test_pids], dtype=np.int64)
    return train_idx, val_idx, test_idx


def regression_metrics(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if len(np.unique(targets)) > 1 and len(np.unique(preds)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r = float(np.corrcoef(targets, preds)[0, 1])
    else:
        pearson_r = float("nan")

    return {
        "MAE": float(mean_absolute_error(targets, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
        "R2": float(r2_score(targets, preds)),
        "Pearson r": pearson_r,
    }


def classification_metrics(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "Accuracy": float(accuracy_score(targets, preds)),
        "F1 (weighted)": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "F1 (macro)": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "Precision (weighted)": float(precision_score(targets, preds, average="weighted", zero_division=0)),
        "Precision (macro)": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "Recall (weighted)": float(recall_score(targets, preds, average="weighted", zero_division=0)),
        "Recall (macro)": float(recall_score(targets, preds, average="macro", zero_division=0)),
    }


def evaluate_regression(targets, preds, subject_ids) -> dict:
    targets = np.asarray(targets, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = subject_bootstrap_ci(targets, preds, subject_ids)
    except Exception:
        summary = regression_metrics(targets, preds)
        metrics = {
            "MAE": {"mean": summary["MAE"], "ci": np.array([np.nan, np.nan])},
            "RMSE": {"mean": summary["RMSE"], "ci": np.array([np.nan, np.nan])},
            "R2": {"mean": summary["R2"], "ci": np.array([np.nan, np.nan])},
            "Pearson r": {"mean": summary["Pearson r"], "ci": np.array([np.nan, np.nan])},
        }
    payload = {
        "metrics": metrics,
        "targets": targets.tolist(),
        "preds": preds.tolist(),
        "subjects": subject_ids.tolist(),
    }
    payload["summary"] = regression_metrics(targets, preds)
    return payload


def evaluate_classification(targets, preds, subject_ids) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    try:
        metrics = subject_bootstrap_ci_class(targets, preds, subject_ids)
    except Exception:
        summary = classification_metrics(targets, preds)
        metrics = {
            "Accuracy": {"mean": summary["Accuracy"], "ci": np.array([np.nan, np.nan])},
            "F1 (weighted)": {"mean": summary["F1 (weighted)"], "ci": np.array([np.nan, np.nan])},
            "F1 (macro)": {"mean": summary["F1 (macro)"], "ci": np.array([np.nan, np.nan])},
            "Precision (weighted)": {"mean": summary["Precision (weighted)"], "ci": np.array([np.nan, np.nan])},
            "Precision (macro)": {"mean": summary["Precision (macro)"], "ci": np.array([np.nan, np.nan])},
            "Recall (weighted)": {"mean": summary["Recall (weighted)"], "ci": np.array([np.nan, np.nan])},
            "Recall (macro)": {"mean": summary["Recall (macro)"], "ci": np.array([np.nan, np.nan])},
        }
    payload = {
        "metrics": metrics,
        "targets": targets.tolist(),
        "preds": preds.tolist(),
        "subjects": subject_ids.tolist(),
    }
    payload["summary"] = classification_metrics(targets, preds)
    return payload


def save_json(path: Path, payload: dict) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2), encoding="utf-8")
