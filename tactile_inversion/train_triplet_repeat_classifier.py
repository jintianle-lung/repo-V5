import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ARCHIVE_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(CODE_ARCHIVE_DIR)
if CODE_ARCHIVE_DIR not in sys.path:
    sys.path.insert(0, CODE_ARCHIVE_DIR)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_csv_data(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    mat_cols = [c for c in df.columns if str(c).strip().startswith("MAT_")]
    if mat_cols:
        mat_cols.sort(key=lambda x: int(str(x).strip().split("_")[1]))
        data = df[mat_cols].values.astype(np.float32)
    else:
        data = df.iloc[:, -96:].values.astype(np.float32)
    return data


def normalize_frames(data_96: np.ndarray) -> np.ndarray:
    n = len(data_96)
    out = np.zeros((n, 12, 8), dtype=np.float32)
    for i in range(n):
        fr = data_96[i]
        mn = float(fr.min())
        mx = float(fr.max())
        if mx - mn > 1e-6:
            fr = (fr - mn) / (mx - mn)
        else:
            fr = fr - mn
        out[i] = fr.reshape(12, 8)
    return out


def sanitize_segments(segments, num_frames: int) -> List[Tuple[int, int]]:
    cleaned = []
    for seg in segments or []:
        if not isinstance(seg, (list, tuple)) or len(seg) == 0:
            continue
        if len(seg) == 1:
            start = int(seg[0])
            end = start + 1
        else:
            start = int(seg[0])
            end = int(seg[1])
            if end <= start:
                end = start + 1
        start = max(0, min(start, num_frames))
        end = max(0, min(end, num_frames))
        if end > start:
            cleaned.append((start, end))
    return cleaned


def is_overlap_positive(window_start: int, window_end_inclusive: int, segments: List[Tuple[int, int]]) -> int:
    for start, end in segments:
        if window_start < end and window_end_inclusive >= start:
            return 1
    return 0


def parse_size_depth_from_group(group_key: str) -> Tuple[str, str]:
    parts = group_key.split("|")
    if len(parts) != 2:
        return group_key, "unknown"
    return parts[0], parts[1]


def parse_float_from_cm_text(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return float(m.group(1)) if m else 0.0


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_group_key(rel_path: str, file_name: str) -> str:
    safe = rel_path.replace("\\\\", os.sep).replace("\\", os.sep).replace("/", os.sep)
    parts = safe.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"Invalid rel path for group key: {rel_path}")
    # size/depth/file
    if parts[-1].upper() != file_name.upper():
        raise ValueError(f"File mismatch in rel path: {rel_path}, expect {file_name}")
    size = parts[-3]
    depth = parts[-2]
    return f"{size}|{depth}"


def filter_labels_for_file(label_map: Dict, target_file: str) -> Dict[str, dict]:
    out = {}
    for rel_path, info in label_map.items():
        safe = rel_path.replace("\\\\", os.sep).replace("\\", os.sep).replace("/", os.sep)
        if os.path.basename(safe).upper() != target_file.upper():
            continue
        try:
            g = normalize_group_key(rel_path, target_file)
        except Exception:
            continue
        out[g] = {
            "rel_path": safe,
            "segments": info.get("segments", []),
            "size": info.get("size", ""),
            "depth": info.get("depth", ""),
        }
    return out


def split_groups_by_size(
    groups: List[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[str]] = {}
    for g in groups:
        size, _ = parse_size_depth_from_group(g)
        buckets.setdefault(size, []).append(g)

    train, val, test = [], [], []
    for size in sorted(buckets.keys(), key=lambda s: parse_float_from_cm_text(s)):
        arr = sorted(buckets[size], key=lambda x: parse_float_from_cm_text(parse_size_depth_from_group(x)[1]))
        rng.shuffle(arr)
        m = len(arr)
        if m <= 2:
            train.extend(arr[:1])
            if m > 1:
                val.extend(arr[1:2])
            continue

        n_val = int(round(m * val_ratio))
        n_test = int(round(m * test_ratio))
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        if n_val + n_test >= m:
            n_test = max(1, n_test - 1)
        if n_val + n_test >= m:
            n_val = max(1, n_val - 1)

        val.extend(arr[:n_val])
        test.extend(arr[n_val : n_val + n_test])
        train.extend(arr[n_val + n_test :])

    return train, val, test


def split_groups_balanced_grid(groups: List[str]) -> Tuple[List[str], List[str], List[str]]:
    buckets: Dict[str, List[str]] = {}
    for g in groups:
        size, _depth = parse_size_depth_from_group(g)
        buckets.setdefault(size, []).append(g)

    train, val, test = [], [], []
    size_keys = sorted(buckets.keys(), key=lambda s: parse_float_from_cm_text(s))
    for size_idx, size in enumerate(size_keys):
        arr = sorted(buckets[size], key=lambda x: parse_float_from_cm_text(parse_size_depth_from_group(x)[1]))
        m = len(arr)
        if m == 0:
            continue
        if m == 1:
            train.extend(arr)
            continue
        if m == 2:
            val_idx = size_idx % 2
            val.append(arr[val_idx])
            train.append(arr[1 - val_idx])
            continue

        val_idx = size_idx % m
        test_idx = (size_idx + max(1, m // 2)) % m
        if test_idx == val_idx:
            test_idx = (test_idx + 1) % m

        for i, g in enumerate(arr):
            if i == val_idx:
                val.append(g)
            elif i == test_idx:
                test.append(g)
            else:
                train.append(g)

    if len(val) == 0 and len(train) > 1:
        val.append(train.pop(0))
    if len(test) == 0 and len(train) > 1:
        test.append(train.pop(-1))
    return train, val, test


def compress_samples_by_gap(sample_records: List[dict], min_gap: int) -> List[dict]:
    min_gap = int(max(1, min_gap))
    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for s in sample_records:
        key = (str(s["group_key"]), int(s["label"]))
        grouped.setdefault(key, []).append(s)

    out = []
    for _key, arr in grouped.items():
        arr = sorted(arr, key=lambda x: int(x["end_row"]))
        keep_last = None
        best_candidate = None
        for s in arr:
            end_row = int(s["end_row"])
            if keep_last is None:
                best_candidate = s
                keep_last = end_row
                out.append(best_candidate)
                continue

            if end_row - keep_last >= min_gap:
                best_candidate = s
                keep_last = end_row
                out.append(best_candidate)
            else:
                cur_score = abs(float(best_candidate.get("soft_label", best_candidate["label"])) - 0.5)
                new_score = abs(float(s.get("soft_label", s["label"])) - 0.5)
                if new_score > cur_score:
                    out[-1] = s
                    best_candidate = s

    out.sort(key=lambda x: (str(x["group_key"]), int(x["end_row"])))
    return out


class TripletWindowDataset(Dataset):
    def __init__(
        self,
        group_records: Dict,
        sample_records: List[dict],
        is_train: bool = False,
        aug_noise_std: float = 0.0,
        aug_scale_jitter: float = 0.0,
        aug_frame_dropout: float = 0.0,
    ):
        self.group_records = group_records
        self.samples = sample_records
        self.is_train = bool(is_train)
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_scale_jitter = float(max(0.0, aug_scale_jitter))
        self.aug_frame_dropout = float(min(max(0.0, aug_frame_dropout), 0.5))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        g = s["group_key"]
        end_row = int(s["end_row"])
        y = float(s["label"])
        y_soft = float(s.get("soft_label", y))

        frames_1 = self.group_records[g]["frames_1"]
        frames_2 = self.group_records[g]["frames_2"]
        frames_3 = self.group_records[g]["frames_3"]
        seq_len = int(self.group_records[g]["seq_len"])

        st = end_row - seq_len + 1
        x1 = frames_1[st : end_row + 1]  # (T,12,8)
        x2 = frames_2[st : end_row + 1]
        x3 = frames_3[st : end_row + 1]

        x = np.stack([x1, x2, x3], axis=0).astype(np.float32)  # (R,T,12,8)
        x = np.expand_dims(x, axis=2)  # (R,T,1,12,8)
        if self.is_train:
            if self.aug_scale_jitter > 0.0:
                scale = 1.0 + float(np.random.uniform(-self.aug_scale_jitter, self.aug_scale_jitter))
                x = x * scale
            if self.aug_noise_std > 0.0:
                x = x + np.random.normal(loc=0.0, scale=self.aug_noise_std, size=x.shape).astype(np.float32)
            if self.aug_frame_dropout > 0.0:
                keep = (np.random.rand(x.shape[0], x.shape[1], 1, 1, 1) >= self.aug_frame_dropout).astype(np.float32)
                x = x * keep
            x = np.clip(x, 0.0, 1.0)
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(y_soft, dtype=torch.float32),
        )


class RepeatSequenceEncoder(nn.Module):
    def __init__(self, lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.35):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.temporal_lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.temporal_attn = nn.Linear(lstm_hidden * 2, 1)
        self.spatial_drop = nn.Dropout3d(min(0.4, max(0.0, dropout * 0.6)))
        self.temporal_drop = nn.Dropout(dropout)
        self.out_dim = lstm_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,1,12,8)
        x = x.permute(0, 2, 1, 3, 4)  # (B,1,T,H,W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.spatial_drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # (B,64,T,h,w)
        x = self.spatial_drop(x)

        x = x.mean(dim=[3, 4])  # (B,64,T)
        x = x.permute(0, 2, 1)  # (B,T,64)
        h, _ = self.temporal_lstm(x)  # (B,T,2H)
        h = self.temporal_drop(h)
        attn = torch.softmax(self.temporal_attn(h), dim=1)  # (B,T,1)
        feat = torch.sum(attn * h, dim=1)  # (B,2H)
        return feat


class TripletRepeatClassifier(nn.Module):
    def __init__(self, lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.35):
        super().__init__()
        self.encoder = RepeatSequenceEncoder(
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        d = self.encoder.out_dim
        self.rep_attn = nn.Linear(d, 1)
        self.cls_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: (B,R,T,1,12,8), R=3
        b, r, t, c, h, w = x.shape
        feats = []
        aux_logits = []
        for i in range(r):
            fi = self.encoder(x[:, i])  # (B,D)
            feats.append(fi)
            aux_logits.append(self.cls_head(fi))
        feat = torch.stack(feats, dim=1)  # (B,R,D)
        rep_w = torch.softmax(self.rep_attn(feat), dim=1)  # (B,R,1)
        fused = torch.sum(rep_w * feat, dim=1)  # (B,D)
        logit = self.cls_head(fused)  # (B,1)
        aux_logits = torch.cat(aux_logits, dim=1)  # (B,R)
        return logit, aux_logits, rep_w.squeeze(-1)


def compute_cls_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    pred = (y_score >= float(threshold)).astype(np.int32)
    y = y_true.astype(np.int32)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "specificity": float(spe),
        "f1": float(f1),
    }


def select_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    best = None
    for thr in np.linspace(0.0, 1.0, 1001):
        m = compute_cls_metrics(y_true, y_score, float(thr))
        if best is None:
            best = m
            continue
        if m["f1"] > best["f1"]:
            best = m
            continue
        if m["f1"] == best["f1"] and m["recall"] > best["recall"]:
            best = m
    return best


def build_roc(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = [1.1] + np.linspace(1.0, 0.0, 1001).tolist() + [-0.1]
    pts = []
    for thr in thresholds:
        m = compute_cls_metrics(y_true, y_score, float(thr))
        tpr = m["recall"]
        fpr = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else 0.0
        pts.append((fpr, tpr))
    pts = sorted(pts, key=lambda x: x[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    auc = float(np.trapz(ys, xs))
    return auc


def build_pr(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = [1.1] + np.linspace(1.0, 0.0, 1001).tolist() + [-0.1]
    pts = []
    for thr in thresholds:
        m = compute_cls_metrics(y_true, y_score, float(thr))
        pts.append((m["recall"], m["precision"]))
    pts = sorted(pts, key=lambda x: x[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    ap = float(np.trapz(ys, xs))
    return ap


def smooth_curve(values: List[float], alpha: float = 0.35) -> List[float]:
    if not values:
        return []
    alpha = float(min(max(alpha, 0.01), 1.0))
    out = [float(values[0])]
    for i in range(1, len(values)):
        out.append(alpha * float(values[i]) + (1.0 - alpha) * out[-1])
    return out


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    aux_weight: float = 0.25,
    label_smoothing: float = 0.0,
    grad_clip: float = 0.0,
    use_soft_label: bool = True,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    all_y = []
    all_score = []
    loss_sum = 0.0
    n = 0
    with torch.set_grad_enabled(is_train):
        for x, y, y_soft in loader:
            x = x.to(device)  # (B,R,T,1,12,8)
            y = y.to(device).unsqueeze(1)  # (B,1) hard label for metrics
            y_soft = y_soft.to(device).unsqueeze(1)  # (B,1) soft vote target for optimization
            if is_train:
                optimizer.zero_grad()
            logit, aux_logits, _rep_w = model(x)
            y_loss = y_soft if use_soft_label else y
            if is_train and label_smoothing > 0.0:
                y_loss = y_loss * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)
            main_loss = criterion(logit, y_loss)
            aux_target = y_loss.expand(-1, aux_logits.shape[1])
            aux_loss = criterion(aux_logits, aux_target)
            loss = main_loss + float(aux_weight) * aux_loss
            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()
            bs = x.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs
            all_y.append(y.detach().cpu().numpy().reshape(-1))
            all_score.append(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_y, axis=0).astype(np.int32)
    y_score = np.concatenate(all_score, axis=0).astype(np.float64)
    return {
        "loss": float(loss_sum / max(n, 1)),
        "y_true": y_true,
        "y_score": y_score,
    }


def plot_curves(history: List[dict], out_path: str):
    epochs = [int(x["epoch"]) for x in history]
    train_loss = [float(x["train_loss"]) for x in history]
    train_eval_loss = [float(x.get("train_eval_loss", np.nan)) for x in history]
    val_loss = [float(x["val_loss"]) for x in history]
    val_f1 = [float(x["val_f1"]) for x in history]
    val_auc = [float(x["val_auc"]) for x in history]
    val_ap = [float(x["val_ap"]) for x in history]
    lrs = [float(x.get("lr", np.nan)) for x in history]

    s_train = smooth_curve(train_loss, alpha=0.35)
    s_train_eval = smooth_curve(train_eval_loss, alpha=0.35)
    s_val = smooth_curve(val_loss, alpha=0.35)
    s_f1 = smooth_curve(val_f1, alpha=0.35)
    s_auc = smooth_curve(val_auc, alpha=0.35)
    s_ap = smooth_curve(val_ap, alpha=0.35)
    gap = [float(a) - float(b) for a, b in zip(train_eval_loss, val_loss)]
    s_gap = smooth_curve(gap, alpha=0.35)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(epochs, train_loss, color="#A0A4AA", linewidth=1.2, alpha=0.5, label="train_loss(raw)")
    axes[0, 0].plot(epochs, s_train, color="#1f77b4", linewidth=2.2, label="train_loss(smooth)")
    if np.isfinite(np.array(train_eval_loss, dtype=np.float64)).any():
        axes[0, 0].plot(
            epochs,
            train_eval_loss,
            color="#E0A458",
            linewidth=1.2,
            alpha=0.45,
            label="train_eval_loss(raw)",
        )
        axes[0, 0].plot(
            epochs,
            s_train_eval,
            color="#ff7f0e",
            linewidth=2.2,
            label="train_eval_loss(smooth)",
        )
    axes[0, 0].plot(epochs, val_loss, color="#90B887", linewidth=1.2, alpha=0.5, label="val_loss(raw)")
    axes[0, 0].plot(epochs, s_val, color="#2ca02c", linewidth=2.2, label="val_loss(smooth)")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].plot(epochs, val_f1, color="#8FA0B8", linewidth=1.2, alpha=0.45, label="val_f1(raw)")
    axes[0, 1].plot(epochs, s_f1, color="#1f77b4", linewidth=2.2, label="val_f1(smooth)")
    axes[0, 1].plot(epochs, val_auc, color="#F3B46C", linewidth=1.2, alpha=0.45, label="val_auc(raw)")
    axes[0, 1].plot(epochs, s_auc, color="#ff7f0e", linewidth=2.2, label="val_auc(smooth)")
    axes[0, 1].plot(epochs, val_ap, color="#9DD39A", linewidth=1.2, alpha=0.45, label="val_ap(raw)")
    axes[0, 1].plot(epochs, s_ap, color="#2ca02c", linewidth=2.2, label="val_ap(smooth)")
    axes[0, 1].set_title("Validation Metrics")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("score")
    axes[0, 1].legend(fontsize=8, ncol=2)

    axes[1, 0].axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axes[1, 0].plot(epochs, gap, color="#C9CDD2", linewidth=1.3, alpha=0.55, label="gap(raw)")
    axes[1, 0].plot(epochs, s_gap, color="#d62728", linewidth=2.2, label="gap(smooth)")
    axes[1, 0].set_title("Generalization Gap (train_eval_loss - val_loss)")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("gap")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(epochs, lrs, color="#9467bd", linewidth=2.2)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("lr")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_curves_plotly(history: List[dict], out_html: str) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    epochs = [int(x["epoch"]) for x in history]
    train_loss = [float(x["train_loss"]) for x in history]
    train_eval_loss = [float(x.get("train_eval_loss", np.nan)) for x in history]
    val_loss = [float(x["val_loss"]) for x in history]
    val_f1 = [float(x["val_f1"]) for x in history]
    val_auc = [float(x["val_auc"]) for x in history]
    val_ap = [float(x["val_ap"]) for x in history]
    lrs = [float(x.get("lr", np.nan)) for x in history]
    gap = [float(a) - float(b) for a, b in zip(train_eval_loss, val_loss)]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Loss Curves",
            "Validation Metrics",
            "Generalization Gap",
            "Learning Rate",
        ),
    )
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="train_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=train_eval_loss, name="train_eval_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="val_loss"), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=val_f1, name="val_f1"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_auc, name="val_auc"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_ap, name="val_ap"), row=1, col=2)

    fig.add_trace(go.Scatter(x=epochs, y=gap, name="gap"), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=lrs, name="lr"), row=2, col=2)

    fig.update_layout(height=780, width=1250, title_text="Triplet Repeat Training Dashboard")
    fig.write_html(out_html, include_plotlyjs="cdn")
    return True


@dataclass
class Config:
    seed: int = int(os.environ.get("TRIPLET_SEED", "2026"))
    seq_len: int = int(os.environ.get("TRIPLET_SEQ_LEN", "10"))
    stride: int = int(os.environ.get("TRIPLET_STRIDE", "2"))
    epochs: int = int(os.environ.get("TRIPLET_EPOCHS", "120"))
    batch_size: int = int(os.environ.get("TRIPLET_BATCH_SIZE", "48"))
    lr: float = float(os.environ.get("TRIPLET_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("TRIPLET_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("TRIPLET_DROPOUT", "0.40"))
    lstm_hidden: int = int(os.environ.get("TRIPLET_LSTM_HIDDEN", "64"))
    lstm_layers: int = int(os.environ.get("TRIPLET_LSTM_LAYERS", "1"))
    aux_weight: float = float(os.environ.get("TRIPLET_AUX_WEIGHT", "0.20"))
    patience: int = int(os.environ.get("TRIPLET_PATIENCE", "20"))
    val_ratio: float = float(os.environ.get("TRIPLET_VAL_RATIO", "0.17"))
    test_ratio: float = float(os.environ.get("TRIPLET_TEST_RATIO", "0.17"))
    max_neg_pos_ratio: float = float(os.environ.get("TRIPLET_MAX_NEG_POS_RATIO", "2.5"))
    num_workers: int = int(os.environ.get("TRIPLET_NUM_WORKERS", "0"))
    use_weighted_sampler: bool = env_bool("TRIPLET_USE_WEIGHTED_SAMPLER", False)
    use_pos_weight: bool = env_bool("TRIPLET_USE_POS_WEIGHT", False)
    label_smoothing: float = float(os.environ.get("TRIPLET_LABEL_SMOOTHING", "0.04"))
    grad_clip: float = float(os.environ.get("TRIPLET_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("TRIPLET_AUG_NOISE_STD", "0.015"))
    aug_scale_jitter: float = float(os.environ.get("TRIPLET_AUG_SCALE_JITTER", "0.10"))
    aug_frame_dropout: float = float(os.environ.get("TRIPLET_AUG_FRAME_DROPOUT", "0.03"))
    min_lr_ratio: float = float(os.environ.get("TRIPLET_MIN_LR_RATIO", "0.08"))
    selection_metric: str = os.environ.get("TRIPLET_SELECTION_METRIC", "auc").strip().lower()
    split_mode: str = os.environ.get("TRIPLET_SPLIT_MODE", "balanced_grid").strip().lower()
    window_dedup_gap: int = int(os.environ.get("TRIPLET_WINDOW_DEDUP_GAP", "6"))
    use_soft_label: bool = env_bool("TRIPLET_USE_SOFT_LABEL", True)
    train_eval_each_epoch: bool = env_bool("TRIPLET_TRAIN_EVAL_EACH_EPOCH", True)
    save_plotly_html: bool = env_bool("TRIPLET_SAVE_PLOTLY_HTML", True)

    def __post_init__(self):
        if self.selection_metric not in {"auc", "f1"}:
            self.selection_metric = "auc"
        if self.split_mode not in {"balanced_grid", "random_by_size"}:
            self.split_mode = "balanced_grid"
        self.window_dedup_gap = int(max(1, self.window_dedup_gap))
        self.data_root = os.environ.get(
            "TRIPLET_DATA_ROOT",
            os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"),
        )
        self.file1_labels = os.environ.get(
            "TRIPLET_FILE1_LABELS",
            os.path.join(REPO_ROOT, "manual_keyframe_labels.json"),
        )
        self.file2_labels = os.environ.get(
            "TRIPLET_FILE2_LABELS",
            os.path.join(self.data_root, "manual_keyframe_labels_file2.json"),
        )
        self.file3_labels = os.environ.get(
            "TRIPLET_FILE3_LABELS",
            os.path.join(self.data_root, "manual_keyframe_labels_file3.json"),
        )
        self.output_dir = os.environ.get(
            "TRIPLET_OUTPUT_DIR",
            os.path.join(CURRENT_DIR, "outputs"),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    label1_all = load_json(cfg.file1_labels)
    label2_all = load_json(cfg.file2_labels)
    label3_all = load_json(cfg.file3_labels)

    l1 = filter_labels_for_file(label1_all, "1.CSV")
    l2 = filter_labels_for_file(label2_all, "2.CSV")
    l3 = filter_labels_for_file(label3_all, "3.CSV")
    common_groups = sorted(list(set(l1.keys()) & set(l2.keys()) & set(l3.keys())))
    if len(common_groups) < 10:
        raise RuntimeError(f"Too few common groups across file1/2/3 labels: {len(common_groups)}")

    # Build group records with preloaded normalized frames.
    group_records = {}
    sample_records = []
    for g in common_groups:
        size, depth = parse_size_depth_from_group(g)
        p1 = os.path.join(cfg.data_root, size, depth, "1.CSV")
        p2 = os.path.join(cfg.data_root, size, depth, "2.CSV")
        p3 = os.path.join(cfg.data_root, size, depth, "3.CSV")
        if not (os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3)):
            continue

        raw1 = read_csv_data(p1)
        raw2 = read_csv_data(p2)
        raw3 = read_csv_data(p3)
        n = min(len(raw1), len(raw2), len(raw3))
        if n < cfg.seq_len:
            continue

        seg1 = sanitize_segments(l1[g]["segments"], n)
        seg2 = sanitize_segments(l2[g]["segments"], n)
        seg3 = sanitize_segments(l3[g]["segments"], n)

        group_records[g] = {
            "frames_1": normalize_frames(raw1[:n]),
            "frames_2": normalize_frames(raw2[:n]),
            "frames_3": normalize_frames(raw3[:n]),
            "raw_1": raw1[:n],
            "raw_2": raw2[:n],
            "raw_3": raw3[:n],
            "segments_1": seg1,
            "segments_2": seg2,
            "segments_3": seg3,
            "seq_len": int(cfg.seq_len),
            "n_frames": int(n),
            "size": size,
            "depth": depth,
        }

    # Build window samples.
    for g in sorted(group_records.keys()):
        rec = group_records[g]
        n = rec["n_frames"]
        raw1 = rec["raw_1"]
        raw2 = rec["raw_2"]
        raw3 = rec["raw_3"]
        for end in range(cfg.seq_len - 1, n, cfg.stride):
            st = end - cfg.seq_len + 1
            seq1 = raw1[st : end + 1]
            seq2 = raw2[st : end + 1]
            seq3 = raw3[st : end + 1]
            if (
                np.isnan(seq1).any()
                or np.isnan(seq2).any()
                or np.isnan(seq3).any()
                or np.all(seq1 == 0)
                or np.all(seq2 == 0)
                or np.all(seq3 == 0)
            ):
                continue
            y1 = is_overlap_positive(st, end, rec["segments_1"])
            y2 = is_overlap_positive(st, end, rec["segments_2"])
            y3 = is_overlap_positive(st, end, rec["segments_3"])
            vote = float(y1 + y2 + y3) / 3.0
            y = 1 if vote >= 0.5 else 0
            sample_records.append(
                {
                    "group_key": g,
                    "end_row": int(end),
                    "label": int(y),
                    "soft_label": float(vote),
                    "votes": [int(y1), int(y2), int(y3)],
                }
            )

    # release raw arrays after sample construction.
    for g in group_records:
        group_records[g].pop("raw_1", None)
        group_records[g].pop("raw_2", None)
        group_records[g].pop("raw_3", None)

    sample_records = compress_samples_by_gap(sample_records, min_gap=cfg.window_dedup_gap)

    if len(sample_records) < 1000:
        raise RuntimeError(f"Too few valid triplet samples: {len(sample_records)}")

    if cfg.split_mode == "balanced_grid":
        train_groups, val_groups, test_groups = split_groups_balanced_grid(groups=sorted(group_records.keys()))
    else:
        train_groups, val_groups, test_groups = split_groups_by_size(
            groups=sorted(group_records.keys()),
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.seed,
        )
    split_group_set = {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }

    train_samples = [s for s in sample_records if s["group_key"] in split_group_set["train"]]
    val_samples = [s for s in sample_records if s["group_key"] in split_group_set["val"]]
    test_samples = [s for s in sample_records if s["group_key"] in split_group_set["test"]]

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Split produced empty train/val/test samples.")

    # Optional train split balancing by downsampling negatives.
    pos_train = [s for s in train_samples if int(s["label"]) == 1]
    neg_train = [s for s in train_samples if int(s["label"]) == 0]
    max_neg = int(round(len(pos_train) * cfg.max_neg_pos_ratio))
    if len(pos_train) > 0 and len(neg_train) > max_neg > 0:
        rng = np.random.default_rng(cfg.seed)
        keep_idx = rng.choice(len(neg_train), size=max_neg, replace=False)
        neg_train = [neg_train[i] for i in keep_idx]
        train_samples = pos_train + neg_train
        rng.shuffle(train_samples)

    ds_train = TripletWindowDataset(
        group_records,
        train_samples,
        is_train=True,
        aug_noise_std=cfg.aug_noise_std,
        aug_scale_jitter=cfg.aug_scale_jitter,
        aug_frame_dropout=cfg.aug_frame_dropout,
    )
    ds_train_eval = TripletWindowDataset(group_records, train_samples, is_train=False)
    ds_val = TripletWindowDataset(group_records, val_samples, is_train=False)
    ds_test = TripletWindowDataset(group_records, test_samples, is_train=False)

    # Optional weighted sampler on train to reduce class imbalance pressure.
    y_train = np.array([int(s["label"]) for s in train_samples], dtype=np.int32)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = float(n_neg / max(n_pos, 1))
    sampler = None
    if cfg.use_weighted_sampler:
        cls_count = np.bincount(y_train, minlength=2).astype(np.float64)
        sample_weights = np.array([1.0 / max(cls_count[y], 1.0) for y in y_train], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_train_eval = DataLoader(
        ds_train_eval,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletRepeatClassifier(
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * cfg.min_lr_ratio
    )
    if cfg.use_pos_weight:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_path = os.path.join(cfg.output_dir, "triplet_repeat_classifier_best.pth")
    last_path = os.path.join(cfg.output_dir, "triplet_repeat_classifier_last.pth")
    curve_path = os.path.join(cfg.output_dir, "triplet_repeat_training_curves.png")
    curve_html_path = os.path.join(cfg.output_dir, "triplet_repeat_training_curves.html")
    summary_path = os.path.join(cfg.output_dir, "triplet_repeat_summary.json")

    history = []
    best_key = None
    best_rec = None
    no_improve = 0
    selection_rule = "max(val_auc) -> max(val_ap) -> max(val_f1) -> min(val_loss)"
    if cfg.selection_metric == "f1":
        selection_rule = "max(val_f1@best_thr) -> max(val_auc) -> max(val_ap) -> min(val_loss)"

    for epoch in range(1, cfg.epochs + 1):
        tr = run_one_epoch(
            model=model,
            loader=loader_train,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            aux_weight=cfg.aux_weight,
            label_smoothing=cfg.label_smoothing,
            grad_clip=cfg.grad_clip,
            use_soft_label=cfg.use_soft_label,
        )
        va = run_one_epoch(
            model=model,
            loader=loader_val,
            device=device,
            criterion=criterion,
            optimizer=None,
            aux_weight=cfg.aux_weight,
            label_smoothing=0.0,
            grad_clip=0.0,
            use_soft_label=cfg.use_soft_label,
        )
        tr_eval_epoch = {"loss": float("nan")}
        if cfg.train_eval_each_epoch:
            tr_eval_epoch = run_one_epoch(
                model=model,
                loader=loader_train_eval,
                device=device,
                criterion=criterion,
                optimizer=None,
                aux_weight=cfg.aux_weight,
                label_smoothing=0.0,
                grad_clip=0.0,
                use_soft_label=cfg.use_soft_label,
            )
        scheduler.step()

        best_thr_val = select_best_f1_threshold(va["y_true"], va["y_score"])
        auc_val = build_roc(va["y_true"], va["y_score"])
        ap_val = build_pr(va["y_true"], va["y_score"])
        rec = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(tr["loss"]),
            "train_eval_loss": float(tr_eval_epoch["loss"]),
            "val_loss": float(va["loss"]),
            "val_f1": float(best_thr_val["f1"]),
            "val_precision": float(best_thr_val["precision"]),
            "val_recall": float(best_thr_val["recall"]),
            "val_best_threshold": float(best_thr_val["threshold"]),
            "val_auc": float(auc_val),
            "val_ap": float(ap_val),
        }
        history.append(rec)
        print(
            f"[{epoch:02d}/{cfg.epochs}] "
            f"train_loss={rec['train_loss']:.4f} "
            f"train_eval_loss={rec['train_eval_loss']:.4f} "
            f"val_loss={rec['val_loss']:.4f} "
            f"val_f1={rec['val_f1']:.4f} "
            f"val_auc={rec['val_auc']:.4f} "
            f"val_ap={rec['val_ap']:.4f} "
            f"thr={rec['val_best_threshold']:.3f}"
        )

        if cfg.selection_metric == "f1":
            cur_key = (rec["val_f1"], rec["val_auc"], rec["val_ap"], -rec["val_loss"])
        else:
            cur_key = (rec["val_auc"], rec["val_ap"], rec["val_f1"], -rec["val_loss"])
        if best_key is None or cur_key > best_key:
            best_key = cur_key
            best_rec = rec
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"Early stop at epoch {epoch} (patience={cfg.patience})")
            break

    torch.save(model.state_dict(), last_path)
    plot_curves(history, curve_path)
    plotly_saved = False
    if cfg.save_plotly_html:
        plotly_saved = plot_curves_plotly(history, curve_html_path)

    # evaluate best
    try:
        best_state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    va = run_one_epoch(
        model,
        loader_val,
        device,
        criterion,
        optimizer=None,
        aux_weight=cfg.aux_weight,
        use_soft_label=cfg.use_soft_label,
    )
    te = run_one_epoch(
        model,
        loader_test,
        device,
        criterion,
        optimizer=None,
        aux_weight=cfg.aux_weight,
        use_soft_label=cfg.use_soft_label,
    )
    tr_eval = run_one_epoch(
        model,
        loader_train_eval,
        device,
        criterion,
        optimizer=None,
        aux_weight=cfg.aux_weight,
        use_soft_label=cfg.use_soft_label,
    )

    best_thr_val = select_best_f1_threshold(va["y_true"], va["y_score"])
    thr = float(best_thr_val["threshold"])
    val_metrics_best_thr = compute_cls_metrics(va["y_true"], va["y_score"], thr)
    test_metrics_val_thr = compute_cls_metrics(te["y_true"], te["y_score"], thr)
    train_metrics_val_thr = compute_cls_metrics(tr_eval["y_true"], tr_eval["y_score"], thr)
    test_metrics_best_thr = select_best_f1_threshold(te["y_true"], te["y_score"])

    summary = {
        "config": {
            "seed": cfg.seed,
            "seq_len": cfg.seq_len,
            "stride": cfg.stride,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "lstm_hidden": cfg.lstm_hidden,
            "lstm_layers": cfg.lstm_layers,
            "aux_weight": cfg.aux_weight,
            "patience": cfg.patience,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "max_neg_pos_ratio": cfg.max_neg_pos_ratio,
            "pos_weight_train_raw": pos_weight,
            "use_weighted_sampler": bool(cfg.use_weighted_sampler),
            "use_pos_weight": bool(cfg.use_pos_weight),
            "label_smoothing": cfg.label_smoothing,
            "grad_clip": cfg.grad_clip,
            "aug_noise_std": cfg.aug_noise_std,
            "aug_scale_jitter": cfg.aug_scale_jitter,
            "aug_frame_dropout": cfg.aug_frame_dropout,
            "min_lr_ratio": cfg.min_lr_ratio,
            "selection_metric": cfg.selection_metric,
            "split_mode": cfg.split_mode,
            "window_dedup_gap": cfg.window_dedup_gap,
            "use_soft_label": bool(cfg.use_soft_label),
            "train_eval_each_epoch": bool(cfg.train_eval_each_epoch),
            "save_plotly_html": bool(cfg.save_plotly_html),
            "data_root": os.path.abspath(cfg.data_root),
            "file1_labels": os.path.abspath(cfg.file1_labels),
            "file2_labels": os.path.abspath(cfg.file2_labels),
            "file3_labels": os.path.abspath(cfg.file3_labels),
            "output_dir": os.path.abspath(cfg.output_dir),
        },
        "split": {
            "group_count_common": int(len(group_records)),
            "train_groups": int(len(train_groups)),
            "val_groups": int(len(val_groups)),
            "test_groups": int(len(test_groups)),
            "train_samples": int(len(train_samples)),
            "val_samples": int(len(val_samples)),
            "test_samples": int(len(test_samples)),
            "train_positive": int(np.sum(y_train == 1)),
            "train_negative": int(np.sum(y_train == 0)),
        },
        "split_groups_detail": {
            "train": sorted(train_groups),
            "val": sorted(val_groups),
            "test": sorted(test_groups),
        },
        "selection_rule": selection_rule,
        "best_record": best_rec,
        "last_record": history[-1] if history else None,
        "history": history,
        "val_metrics_at_val_best_threshold": val_metrics_best_thr,
        "train_metrics_at_val_best_threshold": train_metrics_val_thr,
        "test_metrics_at_val_best_threshold": test_metrics_val_thr,
        "test_metrics_at_test_best_threshold": test_metrics_best_thr,
        "val_auc": float(build_roc(va["y_true"], va["y_score"])),
        "val_ap": float(build_pr(va["y_true"], va["y_score"])),
        "test_auc": float(build_roc(te["y_true"], te["y_score"])),
        "test_ap": float(build_pr(te["y_true"], te["y_score"])),
        "model_path_best": os.path.abspath(best_path),
        "model_path_last": os.path.abspath(last_path),
        "curve_path": os.path.abspath(curve_path),
        "curve_html_path": os.path.abspath(curve_html_path) if plotly_saved else None,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(os.path.abspath(best_path))
    print(os.path.abspath(last_path))
    print(os.path.abspath(curve_path))
    if plotly_saved:
        print(os.path.abspath(curve_html_path))
    print(os.path.abspath(summary_path))


if __name__ == "__main__":
    main()
