import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"

for path in (PROJECT_ROOT, CURRENT_DIR, MODELS_DIR, UTILS_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


from input_normalization_v1 import normalize_raw_frames_window_minmax
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM
from task_protocol_v1 import depth_to_coarse_index, size_to_class_index
from train_detection_oracle_conditioned_3fold import split_base_groups_train_val_balanced
from train_shared_cnn_mstcn_cascade_file3 import SharedCNNMSTCNCascade, load_file_records
from train_triplet_repeat_classifier import build_roc, compute_cls_metrics, select_best_f1_threshold, set_seed


SIZE_COARSE_NAMES = ("small", "medium", "large")
SIZE_TO_COARSE = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.long)
SIZE_TO_COARSE_NP = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detection-gated residual inversion branch attached to a frozen CNN+MS-TCN detector."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(PROJECT_ROOT / "latest_algorithm" / "runs" / "shared_cnn_mstcn_cascade_file3_20260426_active_best"),
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--morph-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--size-reg-mode", choices=["absolute", "expected_residual"], default="absolute")
    parser.add_argument("--size-residual-span", type=float, default=0.35)
    parser.add_argument("--depth-conditioning", choices=["size7", "size7_coarse"], default="size7")
    parser.add_argument("--gate-loss-alpha", type=float, default=0.0)
    parser.add_argument("--gate-loss-min", type=float, default=0.55)
    parser.add_argument("--selection-mode", choices=["legacy", "composite"], default="legacy")
    parser.add_argument("--size-cls-weight", type=float, default=1.0)
    parser.add_argument("--size-coarse-weight", type=float, default=0.45)
    parser.add_argument("--size-reg-weight", type=float, default=0.65)
    parser.add_argument("--depth-cls-weight", type=float, default=0.85)
    parser.add_argument("--depth-binary-weight", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aug-noise-std", type=float, default=0.012)
    parser.add_argument("--aug-scale-jitter", type=float, default=0.08)
    parser.add_argument("--aug-frame-dropout", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class PositiveWindowDataset(Dataset):
    def __init__(
        self,
        records_by_key: Dict[str, dict],
        sample_records: Sequence[dict],
        is_train: bool = False,
        aug_noise_std: float = 0.0,
        aug_scale_jitter: float = 0.0,
        aug_frame_dropout: float = 0.0,
    ):
        self.records_by_key = records_by_key
        self.samples = [s for s in sample_records if int(s["label"]) == 1]
        self.is_train = bool(is_train)
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_scale_jitter = float(max(0.0, aug_scale_jitter))
        self.aug_frame_dropout = float(min(max(0.0, aug_frame_dropout), 0.5))
        self.x_cache: List[np.ndarray] = []
        for sample in self.samples:
            rec = self.records_by_key[sample["group_key"]]
            end_row = int(sample["end_row"])
            seq_len = int(rec.get("seq_len", INPUT_SEQ_LEN))
            st = end_row - seq_len + 1
            raw_window = rec["raw_frames"][st : end_row + 1].astype(np.float32)
            if raw_window.ndim == 2 and raw_window.shape[1] == 96:
                raw_window = raw_window.reshape(raw_window.shape[0], 12, 8)
            self.x_cache.append(normalize_raw_frames_window_minmax(raw_window).astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = self.x_cache[idx].copy()
        if self.is_train:
            if self.aug_scale_jitter > 0.0:
                x *= 1.0 + float(np.random.uniform(-self.aug_scale_jitter, self.aug_scale_jitter))
            if self.aug_noise_std > 0.0:
                x += np.random.normal(0.0, self.aug_noise_std, size=x.shape).astype(np.float32)
            if self.aug_frame_dropout > 0.0:
                keep = (np.random.rand(x.shape[0], 1, 1) >= self.aug_frame_dropout).astype(np.float32)
                x *= keep
            x = np.clip(x, 0.0, 1.0)
        size_idx = int(size_to_class_index(float(sample["size_cm"])))
        depth_idx = int(depth_to_coarse_index(float(sample["depth_cm"])))
        return (
            torch.from_numpy(np.expand_dims(x, axis=1)),
            torch.tensor(size_idx, dtype=torch.long),
            torch.tensor(depth_idx, dtype=torch.long),
            torch.tensor(float(sample["size_cm"]), dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


class AllWindowDataset(Dataset):
    def __init__(self, records_by_key: Dict[str, dict], sample_records: Sequence[dict]):
        self.records_by_key = records_by_key
        self.samples = list(sample_records)
        self.x_cache: List[np.ndarray] = []
        for sample in self.samples:
            rec = self.records_by_key[sample["group_key"]]
            end_row = int(sample["end_row"])
            seq_len = int(rec.get("seq_len", INPUT_SEQ_LEN))
            st = end_row - seq_len + 1
            raw_window = rec["raw_frames"][st : end_row + 1].astype(np.float32)
            if raw_window.ndim == 2 and raw_window.shape[1] == 96:
                raw_window = raw_window.reshape(raw_window.shape[0], 12, 8)
            self.x_cache.append(normalize_raw_frames_window_minmax(raw_window).astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return torch.from_numpy(np.expand_dims(self.x_cache[idx], axis=1)), torch.tensor(float(sample["label"]), dtype=torch.float32)


class Residual2DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout) * 0.35),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class ResidualMorphologyBranch(nn.Module):
    def __init__(self, seq_len: int, morph_dim: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(seq_len, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = Residual2DBlock(32, dropout)
        self.proj = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = Residual2DBlock(64, dropout)
        self.out = nn.Sequential(
            nn.Linear(64 * 2, int(morph_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.last_feature_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,1,12,8). Treat the temporal frames as input channels for a compact morphology encoder.
        x2 = x[:, :, 0]
        fmap = self.block2(self.proj(self.block1(self.stem(x2))))
        self.last_feature_map = fmap
        avg = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
        mx = F.adaptive_max_pool2d(fmap, (1, 1)).flatten(1)
        return self.out(torch.cat([avg, mx], dim=1))


class FrozenDetectorResidualInversion(nn.Module):
    def __init__(
        self,
        frozen_detector: SharedCNNMSTCNCascade,
        morph_dim: int,
        hidden_dim: int,
        dropout: float,
        size_reg_mode: str = "absolute",
        size_residual_span: float = 0.35,
        depth_conditioning: str = "size7",
    ):
        super().__init__()
        self.size_reg_mode = str(size_reg_mode).strip().lower()
        if self.size_reg_mode not in {"absolute", "expected_residual"}:
            raise ValueError(f"Unsupported size_reg_mode: {size_reg_mode}")
        self.size_residual_span = float(size_residual_span)
        self.depth_conditioning = str(depth_conditioning).strip().lower()
        if self.depth_conditioning not in {"size7", "size7_coarse"}:
            raise ValueError(f"Unsupported depth_conditioning: {depth_conditioning}")
        self.detector = frozen_detector
        for p in self.detector.parameters():
            p.requires_grad_(False)
        self.detector.eval()
        self.morph = ResidualMorphologyBranch(INPUT_SEQ_LEN, morph_dim, dropout)
        feat_dim = int(self.detector.backbone.feature_dim)
        fused_dim = feat_dim + 1 + int(morph_dim)
        hidden_dim = max(int(hidden_dim), fused_dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.size_head = nn.Linear(hidden_dim, len(SIZE_VALUES_CM))
        self.size_coarse_head = nn.Linear(hidden_dim, len(SIZE_COARSE_NAMES))
        self.size_reg_head = nn.Linear(hidden_dim, 1)
        depth_in = hidden_dim + len(SIZE_VALUES_CM) + 1
        if self.depth_conditioning == "size7_coarse":
            depth_in += len(SIZE_COARSE_NAMES)
        self.depth_head = nn.Sequential(
            nn.LayerNorm(depth_in),
            nn.Linear(depth_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, len(COARSE_DEPTH_ORDER)),
        )
        self.deep_head = nn.Sequential(
            nn.LayerNorm(depth_in),
            nn.Linear(depth_in, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.register_buffer("size_values", torch.tensor(SIZE_VALUES_CM, dtype=torch.float32).view(1, -1))
        self.size_min = float(min(SIZE_VALUES_CM))
        self.size_span = float(max(SIZE_VALUES_CM) - min(SIZE_VALUES_CM))

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            det_logit, _size_logits0, size_reg0, _depth_logits0, extra = self.detector(x, return_features=True)
            z = extra["shared_features"]
            p_det = torch.sigmoid(det_logit)
        m = self.morph(x)
        h = self.fuse(torch.cat([z, p_det, m], dim=1))
        size_logits = self.size_head(h)
        size_coarse_logits = self.size_coarse_head(h)
        size_probs = torch.softmax(size_logits, dim=1)
        size_expected = torch.sum(size_probs * self.size_values.to(size_probs.device), dim=1, keepdim=True)
        if self.size_reg_mode == "expected_residual":
            residual = torch.tanh(self.size_reg_head(h)) * float(max(self.size_residual_span, 0.0))
            size_reg_cm = torch.clamp(size_expected + residual, self.size_min, float(max(SIZE_VALUES_CM)))
        else:
            size_reg_norm = torch.sigmoid(self.size_reg_head(h))
            size_reg_cm = self.size_min + size_reg_norm * max(self.size_span, 1e-6)
        depth_parts = [h, size_probs, size_expected / float(max(SIZE_VALUES_CM))]
        if self.depth_conditioning == "size7_coarse":
            depth_parts.append(torch.softmax(size_coarse_logits, dim=1))
        depth_input = torch.cat(depth_parts, dim=1)
        depth_logits = self.depth_head(depth_input)
        deep_logit = self.deep_head(depth_input)
        return {
            "det_logit": det_logit,
            "det_prob": p_det,
            "size_logits": size_logits,
            "size_coarse_logits": size_coarse_logits,
            "size_reg_cm": size_reg_cm,
            "depth_logits": depth_logits,
            "deep_logit": deep_logit,
        }


def load_frozen_detector(run_dir: Path, device: torch.device):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = summary["config"]
    model = SharedCNNMSTCNCascade(
        frame_feature_dim=int(cfg.get("frame_feature_dim", 32)),
        temporal_channels=int(cfg.get("temporal_channels", 48)),
        temporal_blocks=int(cfg.get("temporal_blocks", 2)),
        temporal_pooling=str(cfg.get("temporal_pooling", "mean")),
        dropout=float(cfg.get("dropout", 0.30)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    threshold = float(summary["val_best_f1_threshold_metrics"]["threshold"])
    return cfg, model, threshold


def build_datasets(cfg: dict):
    args = SimpleNamespace(
        file1_labels=str(cfg["file1_labels"]),
        file2_labels=str(cfg["file2_labels"]),
        file3_labels=str(cfg["file3_labels"]),
        data_root=str(cfg["data_root"]),
        label_mode=str(cfg.get("label_mode", "window_overlap_positive")),
        input_normalization=str(cfg.get("input_normalization", "window_minmax")),
    )
    rec1, samples1, rec2, samples2, rec3, samples3, common_groups = load_file_records(args)
    train_groups, val_groups = split_base_groups_train_val_balanced(common_groups)
    train_set, val_set = set(train_groups), set(val_groups)
    records_all = {}
    records_all.update(rec1)
    records_all.update(rec2)
    train_records = {k: v for k, v in records_all.items() if v["file_name"] in {"1.CSV", "2.CSV"} and v["base_group"] in train_set}
    val_records = {k: v for k, v in records_all.items() if v["file_name"] in {"1.CSV", "2.CSV"} and v["base_group"] in val_set}
    train_samples = [s for s in samples1 + samples2 if s["base_group"] in train_set]
    val_samples = [s for s in samples1 + samples2 if s["base_group"] in val_set]
    return rec1, rec2, rec3, train_records, val_records, rec3, train_samples, val_samples, samples3


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def top2_acc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(top_k_accuracy_score(y_true, proba, k=min(2, proba.shape[1]), labels=list(range(proba.shape[1]))))


def _weighted_mean(loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(loss * sample_weight) / torch.clamp(torch.sum(sample_weight), min=1e-6)


def run_epoch(model, loader, device, optimizer, loss_weights, weights, grad_clip=0.0, gate_loss_alpha=0.0, gate_loss_min=0.55):
    train = optimizer is not None
    model.train() if train else model.eval()
    # Keep the detector truly frozen: BatchNorm/Dropout must remain in eval mode
    # even while the residual inversion branch is being trained.
    model.detector.eval()
    total_loss = 0.0
    total_n = 0
    buckets = {k: [] for k in ["size_true", "depth_true", "size_cm", "det_prob", "size_logits", "size_coarse_logits", "size_reg", "depth_logits", "deep_logit"]}
    with torch.set_grad_enabled(train):
        for x, size_idx, depth_idx, size_cm, _sample_idx in loader:
            x = x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            size_cm = size_cm.to(device).unsqueeze(1)
            if train:
                optimizer.zero_grad()
            out = model(x)
            size_coarse_idx = SIZE_TO_COARSE.to(device)[size_idx]
            deep_true = (depth_idx == 2).float().unsqueeze(1)
            sample_weight = torch.ones_like(size_cm.reshape(-1), device=device)
            if float(gate_loss_alpha) > 0.0:
                gate_weight = torch.clamp(out["det_prob"].detach().reshape(-1), min=float(gate_loss_min), max=1.0)
                gate_weight = gate_weight / torch.clamp(gate_weight.mean(), min=1e-6)
                sample_weight = (1.0 - float(gate_loss_alpha)) + float(gate_loss_alpha) * gate_weight
            loss = (
                float(loss_weights["size_cls"])
                * _weighted_mean(F.cross_entropy(out["size_logits"], size_idx, weight=weights["size"], reduction="none"), sample_weight)
                + float(loss_weights["size_coarse"])
                * _weighted_mean(F.cross_entropy(out["size_coarse_logits"], size_coarse_idx, weight=weights["size_coarse"], reduction="none"), sample_weight)
                + float(loss_weights["size_reg"])
                * _weighted_mean(F.smooth_l1_loss(out["size_reg_cm"], size_cm, reduction="none").reshape(-1), sample_weight)
                + float(loss_weights["depth_cls"])
                * _weighted_mean(F.cross_entropy(out["depth_logits"], depth_idx, weight=weights["depth"], reduction="none"), sample_weight)
                + float(loss_weights["depth_binary"])
                * _weighted_mean(F.binary_cross_entropy_with_logits(out["deep_logit"], deep_true, reduction="none").reshape(-1), sample_weight)
            )
            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                optimizer.step()
            bs = int(x.size(0))
            total_loss += float(loss.detach().cpu()) * bs
            total_n += bs
            buckets["size_true"].append(size_idx.detach().cpu().numpy())
            buckets["depth_true"].append(depth_idx.detach().cpu().numpy())
            buckets["size_cm"].append(size_cm.detach().cpu().numpy().reshape(-1))
            buckets["det_prob"].append(out["det_prob"].detach().cpu().numpy().reshape(-1))
            buckets["size_logits"].append(out["size_logits"].detach().cpu().numpy())
            buckets["size_coarse_logits"].append(out["size_coarse_logits"].detach().cpu().numpy())
            buckets["size_reg"].append(out["size_reg_cm"].detach().cpu().numpy().reshape(-1))
            buckets["depth_logits"].append(out["depth_logits"].detach().cpu().numpy())
            buckets["deep_logit"].append(out["deep_logit"].detach().cpu().numpy().reshape(-1))
    pred = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    pred["loss"] = total_loss / max(total_n, 1)
    return pred


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics_from_pred(pred: dict, threshold: float) -> dict:
    y_size = pred["size_true"].astype(np.int32)
    y_depth = pred["depth_true"].astype(np.int32)
    y_size_coarse = SIZE_TO_COARSE_NP[y_size]
    y_deep = (y_depth == 2).astype(np.int32)
    size_proba = softmax_np(pred["size_logits"])
    size_coarse_proba = softmax_np(pred["size_coarse_logits"])
    depth_proba = softmax_np(pred["depth_logits"])
    deep_score = sigmoid_np(pred["deep_logit"])
    size_hat = np.argmax(size_proba, axis=1)
    size_coarse_hat = np.argmax(size_coarse_proba, axis=1)
    depth_hat = np.argmax(depth_proba, axis=1)
    deep_hat = (deep_score >= 0.5).astype(np.int32)
    gate = pred["det_prob"] >= float(threshold)
    out = {
        "loss": float(pred["loss"]),
        "gate_coverage": float(np.mean(gate)),
        "size_acc": float(accuracy_score(y_size, size_hat)),
        "size_top2": top2_acc(y_size, size_proba),
        "size_coarse_acc": float(accuracy_score(y_size_coarse, size_coarse_hat)),
        "size_coarse_top2": top2_acc(y_size_coarse, size_coarse_proba),
        "size_mae_cm": float(mean_absolute_error(pred["size_cm"], pred["size_reg"])),
        "depth_acc": float(accuracy_score(y_depth, depth_hat)),
        "depth_top2": top2_acc(y_depth, depth_proba),
        "deep_vs_rest_acc": float(accuracy_score(y_deep, deep_hat)),
        "deep_vs_rest_bal_acc": float(balanced_accuracy_score(y_deep, deep_hat)),
    }
    if len(np.unique(y_deep)) == 2:
        out["deep_vs_rest_auc"] = float(roc_auc_score(y_deep, deep_score))
    for prefix, mask in [("gated", gate)]:
        if int(np.sum(mask)) > 0:
            out[f"{prefix}_n"] = int(np.sum(mask))
            out[f"{prefix}_size_acc"] = float(accuracy_score(y_size[mask], size_hat[mask]))
            out[f"{prefix}_size_top2"] = top2_acc(y_size[mask], size_proba[mask])
            out[f"{prefix}_size_mae_cm"] = float(mean_absolute_error(pred["size_cm"][mask], pred["size_reg"][mask]))
            out[f"{prefix}_depth_acc"] = float(accuracy_score(y_depth[mask], depth_hat[mask]))
            out[f"{prefix}_depth_top2"] = top2_acc(y_depth[mask], depth_proba[mask])
    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(path: Path, cm: np.ndarray, labels: Sequence[str], title: str, cmap: str = "Blues"):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_roc_curves(path: Path, pred: dict):
    y_size = pred["size_true"].astype(np.int32)
    y_depth = pred["depth_true"].astype(np.int32)
    y_size_coarse = SIZE_TO_COARSE_NP[y_size]
    y_deep = (y_depth == 2).astype(np.int32)
    curves = [
        ("size 7-class micro", y_size, softmax_np(pred["size_logits"]), list(range(7))),
        ("size 3-bin micro", y_size_coarse, softmax_np(pred["size_coarse_logits"]), list(range(3))),
        ("depth 3-class micro", y_depth, softmax_np(pred["depth_logits"]), list(range(3))),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for name, y, score, labels in curves:
        y_bin = label_binarize(y, classes=labels)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), score.ravel())
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    if len(np.unique(y_deep)) == 2:
        fpr, tpr, _ = roc_curve(y_deep, sigmoid_np(pred["deep_logit"]))
        ax.plot(fpr, tpr, lw=2, label=f"deep-vs-rest AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Residual inversion probe ROC curves on File3 positives")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_detection_roc(path: Path, y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, lw=2, color="#4fc3f7", label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    metrics = compute_cls_metrics(y_true, y_score, threshold)
    ax.scatter([1 - metrics["specificity"]], [metrics["recall"]], color="#ef5350", zorder=3, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Frozen detector ROC on File3")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def collect_detection_scores(frozen: SharedCNNMSTCNCascade, dataset: AllWindowDataset, device: torch.device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores, labels = [], []
    frozen.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            det_logit, *_ = frozen(x)
            scores.append(torch.sigmoid(det_logit).cpu().numpy().reshape(-1))
            labels.append(y.numpy().reshape(-1))
    return np.concatenate(labels).astype(np.int32), np.concatenate(scores)


def save_prediction_rows(path: Path, dataset: PositiveWindowDataset, pred: dict, threshold: float):
    size_proba = softmax_np(pred["size_logits"])
    size_coarse_proba = softmax_np(pred["size_coarse_logits"])
    depth_proba = softmax_np(pred["depth_logits"])
    deep_score = sigmoid_np(pred["deep_logit"])
    rows = []
    for i, sample in enumerate(dataset.samples):
        rows.append(
            {
                "group_key": sample["group_key"],
                "base_group": sample["base_group"],
                "end_row": int(sample["end_row"]),
                "p_det": float(pred["det_prob"][i]),
                "gate_open": int(pred["det_prob"][i] >= threshold),
                "true_size_cm": float(sample["size_cm"]),
                "pred_size_idx": int(np.argmax(size_proba[i])),
                "pred_size_cm_class": float(SIZE_VALUES_CM[int(np.argmax(size_proba[i]))]),
                "pred_size_cm_reg": float(pred["size_reg"][i]),
                "size_conf": float(np.max(size_proba[i])),
                "true_size_coarse": SIZE_COARSE_NAMES[int(SIZE_TO_COARSE_NP[pred["size_true"][i]])],
                "pred_size_coarse": SIZE_COARSE_NAMES[int(np.argmax(size_coarse_proba[i]))],
                "size_coarse_conf": float(np.max(size_coarse_proba[i])),
                "true_depth": COARSE_DEPTH_ORDER[int(pred["depth_true"][i])],
                "pred_depth": COARSE_DEPTH_ORDER[int(np.argmax(depth_proba[i]))],
                "depth_conf": float(np.max(depth_proba[i])),
                "true_deep_vs_rest": int(pred["depth_true"][i] == 2),
                "pred_deep_vs_rest": int(deep_score[i] >= 0.5),
                "deep_score": float(deep_score[i]),
            }
        )
    write_csv(path, rows)


def agreement_rows(pred: dict, threshold: float) -> List[dict]:
    size_proba = softmax_np(pred["size_logits"])
    size_coarse_proba = softmax_np(pred["size_coarse_logits"])
    depth_proba = softmax_np(pred["depth_logits"])
    deep_score = sigmoid_np(pred["deep_logit"])
    size_pred = np.argmax(size_proba, axis=1)
    size_coarse_from_7 = SIZE_TO_COARSE_NP[size_pred]
    size_coarse_pred = np.argmax(size_coarse_proba, axis=1)
    size_reg_nearest = np.argmin(np.abs(np.array(SIZE_VALUES_CM)[None, :] - pred["size_reg"][:, None]), axis=1)
    size_reg_coarse = SIZE_TO_COARSE_NP[size_reg_nearest]
    depth_pred = np.argmax(depth_proba, axis=1)
    deep_pred = (deep_score >= 0.5).astype(np.int32)
    deep_from_depth = (depth_pred == 2).astype(np.int32)
    gate = pred["det_prob"] >= threshold

    def row(name, a, b, mask=None):
        if mask is None:
            mask = np.ones_like(a, dtype=bool)
        return {
            "comparison": name,
            "n": int(np.sum(mask)),
            "agreement": float(np.mean(a[mask] == b[mask])) if int(np.sum(mask)) else float("nan"),
            "cohen_kappa": float(cohen_kappa_score(a[mask], b[mask])) if int(np.sum(mask)) and len(np.unique(a[mask])) > 1 and len(np.unique(b[mask])) > 1 else float("nan"),
        }

    return [
        row("size_7class_coarse_vs_size_3bin", size_coarse_from_7, size_coarse_pred),
        row("size_reg_nearest_coarse_vs_size_3bin", size_reg_coarse, size_coarse_pred),
        row("depth_3class_deep_flag_vs_deep_binary", deep_from_depth, deep_pred),
        row("gated_size_7class_coarse_vs_size_3bin", size_coarse_from_7, size_coarse_pred, gate),
        row("gated_depth_3class_deep_flag_vs_deep_binary", deep_from_depth, deep_pred, gate),
    ]


def grad_cam_for_sample(model, x: torch.Tensor, target_kind: str, target_class: int, device: torch.device):
    model.eval()
    x = x.to(device).unsqueeze(0)
    out = model(x)
    fmap = model.morph.last_feature_map
    fmap.retain_grad()
    if target_kind == "size":
        target = out["size_logits"][0, int(target_class)]
    elif target_kind == "depth":
        target = out["depth_logits"][0, int(target_class)]
    elif target_kind == "deep":
        target = out["deep_logit"][0, 0]
    else:
        target = out["size_reg_cm"][0, 0]
    model.zero_grad(set_to_none=True)
    target.backward()
    grad = fmap.grad.detach()[0]
    feat = fmap.detach()[0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu(torch.sum(weights * feat, dim=0))
    cam_np = cam.cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / max(float(cam_np.max() - cam_np.min()), 1e-8)
    img = x.detach().cpu().numpy()[0, :, 0].mean(axis=0)
    return img, cam_np


def plot_cam_contact_sheet(path: Path, model, dataset: PositiveWindowDataset, pred: dict, device: torch.device):
    size_proba = softmax_np(pred["size_logits"])
    depth_proba = softmax_np(pred["depth_logits"])
    deep_score = sigmoid_np(pred["deep_logit"])
    size_pred = np.argmax(size_proba, axis=1)
    depth_pred = np.argmax(depth_proba, axis=1)
    candidates = [
        ("Size CAM", "size", int(size_pred[np.argmax(np.max(size_proba, axis=1))]), int(np.argmax(np.max(size_proba, axis=1)))),
        ("Depth CAM", "depth", int(depth_pred[np.argmax(np.max(depth_proba, axis=1))]), int(np.argmax(np.max(depth_proba, axis=1)))),
        ("Deep-risk CAM", "deep", 1, int(np.argmax(deep_score))),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 8.2))
    for row, (title, kind, cls, idx) in enumerate(candidates):
        x, _s, _d, _cm, _idx = dataset[idx]
        img, cam = grad_cam_for_sample(model, x, kind, cls, device)
        axes[row, 0].imshow(img, cmap="turbo")
        axes[row, 0].set_title(f"{title}: mean window")
        axes[row, 1].imshow(cam, cmap="turbo", vmin=0, vmax=1)
        axes[row, 1].set_title("Residual-CNN Grad-CAM")
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(cam, cmap="turbo", alpha=0.55, vmin=0, vmax=1)
        axes[row, 2].set_title("Overlay")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("Residual inversion branch CAM examples on File3 positives", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "residual_gated_inversion_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg, frozen, default_threshold = load_frozen_detector(run_dir, device)
    threshold = float(default_threshold if args.threshold < 0 else args.threshold)

    _rec1, _rec2, rec3, train_records, val_records, test_records, train_samples, val_samples, test_samples = build_datasets(cfg)
    ds_train = PositiveWindowDataset(train_records, train_samples, True, args.aug_noise_std, args.aug_scale_jitter, args.aug_frame_dropout)
    ds_val = PositiveWindowDataset(val_records, val_samples, False)
    ds_test = PositiveWindowDataset(test_records, test_samples, False)
    ds_test_all = AllWindowDataset(rec3, test_samples)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "experiment_type": "detection_gated_residual_inversion_from_frozen_detector",
        "frozen_run_dir": str(run_dir.resolve()),
        "gate_threshold": threshold,
        "counts": {"train_pos": len(ds_train), "val_pos": len(ds_val), "test_pos": len(ds_test), "test_all": len(ds_test_all)},
        "config": vars(args),
        "note": "The detector/backbone is frozen. Only the residual CNN morphology branch and inversion heads are trained.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = FrozenDetectorResidualInversion(
        frozen,
        args.morph_dim,
        args.hidden_dim,
        args.dropout,
        args.size_reg_mode,
        args.size_residual_span,
        args.depth_conditioning,
    ).to(device)
    train_size = np.array([size_to_class_index(float(s["size_cm"])) for s in ds_train.samples], dtype=np.int32)
    train_depth = np.array([depth_to_coarse_index(float(s["depth_cm"])) for s in ds_train.samples], dtype=np.int32)
    weights = {
        "size": class_weights(train_size, len(SIZE_VALUES_CM)).to(device),
        "size_coarse": class_weights(SIZE_TO_COARSE_NP[train_size], len(SIZE_COARSE_NAMES)).to(device),
        "depth": class_weights(train_depth, len(COARSE_DEPTH_ORDER)).to(device),
    }
    loss_weights = {
        "size_cls": args.size_cls_weight,
        "size_coarse": args.size_coarse_weight,
        "size_reg": args.size_reg_weight,
        "depth_cls": args.depth_cls_weight,
        "depth_binary": args.depth_binary_weight,
    }
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=max(args.lr * 0.05, 1e-6))
    history = []
    best_key = None
    best_state = None
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(
            model,
            loader_train,
            device,
            optimizer,
            loss_weights,
            weights,
            args.grad_clip,
            args.gate_loss_alpha,
            args.gate_loss_min,
        )
        va = run_epoch(model, loader_val, device, None, loss_weights, weights, gate_loss_alpha=args.gate_loss_alpha, gate_loss_min=args.gate_loss_min)
        tm = metrics_from_pred(tr, threshold)
        vm = metrics_from_pred(va, threshold)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": tm["loss"],
            "val_loss": vm["loss"],
            "val_size_acc": vm["size_acc"],
            "val_size_top2": vm["size_top2"],
            "val_size_coarse_acc": vm["size_coarse_acc"],
            "val_size_mae_cm": vm["size_mae_cm"],
            "val_depth_acc": vm["depth_acc"],
            "val_depth_top2": vm["depth_top2"],
            "val_deep_acc": vm["deep_vs_rest_acc"],
            "val_deep_auc": vm.get("deep_vs_rest_auc", float("nan")),
            "val_gate_coverage": vm["gate_coverage"],
        }
        history.append(row)
        composite_score = (
            1.50 * row["val_size_acc"]
            + 1.20 * row["val_size_top2"]
            - 0.80 * row["val_size_mae_cm"]
            + 0.70 * row["val_depth_acc"]
            + 0.35 * row["val_depth_top2"]
            + 0.25 * (row["val_deep_auc"] if np.isfinite(row["val_deep_auc"]) else 0.0)
        )
        row["val_composite_score"] = float(composite_score)
        scheduler.step(composite_score)
        if args.selection_mode == "composite":
            key = (composite_score, row["val_size_top2"], row["val_size_acc"], -row["val_size_mae_cm"], row["val_depth_acc"], -row["val_loss"])
        else:
            key = (
                row["val_size_acc"],
                row["val_size_top2"],
                -row["val_size_mae_cm"],
                row["val_depth_acc"],
                row["val_depth_top2"],
                row["val_deep_auc"],
                -row["val_loss"],
            )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"epoch={epoch:03d} val_size={row['val_size_acc']:.4f}/{row['val_size_top2']:.4f} "
            f"mae={row['val_size_mae_cm']:.3f} depth={row['val_depth_acc']:.4f}/{row['val_depth_top2']:.4f} "
            f"deep_auc={row['val_deep_auc']:.4f}",
            flush=True,
        )
        if no_improve >= int(args.patience):
            break

    write_csv(out_dir / "history.csv", history)
    torch.save(best_state, out_dir / "best_model.pth")
    model.load_state_dict(best_state)
    train_pred = run_epoch(model, DataLoader(ds_train, batch_size=args.batch_size, shuffle=False), device, None, loss_weights, weights, gate_loss_alpha=args.gate_loss_alpha, gate_loss_min=args.gate_loss_min)
    val_pred = run_epoch(model, loader_val, device, None, loss_weights, weights, gate_loss_alpha=args.gate_loss_alpha, gate_loss_min=args.gate_loss_min)
    test_pred = run_epoch(model, loader_test, device, None, loss_weights, weights, gate_loss_alpha=args.gate_loss_alpha, gate_loss_min=args.gate_loss_min)
    y_det, s_det = collect_detection_scores(frozen, ds_test_all, device, args.batch_size)
    det_metrics = compute_cls_metrics(y_det, s_det, threshold)

    train_metrics = metrics_from_pred(train_pred, threshold)
    val_metrics = metrics_from_pred(val_pred, threshold)
    test_metrics = metrics_from_pred(test_pred, threshold)
    save_prediction_rows(out_dir / "test_positive_predictions.csv", ds_test, test_pred, threshold)
    agreement = agreement_rows(test_pred, threshold)
    write_csv(out_dir / "agreement_consistency.csv", agreement)

    size_p = np.argmax(softmax_np(test_pred["size_logits"]), axis=1)
    size_coarse_p = np.argmax(softmax_np(test_pred["size_coarse_logits"]), axis=1)
    depth_p = np.argmax(softmax_np(test_pred["depth_logits"]), axis=1)
    deep_p = (sigmoid_np(test_pred["deep_logit"]) >= 0.5).astype(np.int32)
    plot_confusion(out_dir / "confusion_size_7class.png", confusion_matrix(test_pred["size_true"], size_p, labels=list(range(7))), [f"{v:g}" for v in SIZE_VALUES_CM], "Size 7-class confusion")
    plot_confusion(out_dir / "confusion_size_3bin.png", confusion_matrix(SIZE_TO_COARSE_NP[test_pred["size_true"]], size_coarse_p, labels=list(range(3))), SIZE_COARSE_NAMES, "Size 3-bin confusion")
    plot_confusion(out_dir / "confusion_depth_3class.png", confusion_matrix(test_pred["depth_true"], depth_p, labels=list(range(3))), COARSE_DEPTH_ORDER, "Depth 3-class confusion", cmap="Oranges")
    plot_confusion(out_dir / "confusion_deep_vs_rest.png", confusion_matrix((test_pred["depth_true"] == 2).astype(np.int32), deep_p, labels=[0, 1]), ["rest", "deep"], "Deep-vs-rest confusion", cmap="Oranges")
    plot_roc_curves(out_dir / "roc_inversion_tasks.png", test_pred)
    plot_detection_roc(out_dir / "roc_frozen_detector_file3.png", y_det, s_det, threshold)
    plot_cam_contact_sheet(out_dir / "cam_residual_inversion_contact_sheet.png", model, ds_test, test_pred, device)

    summary = {
        "created_at": datetime.now().isoformat(),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "frozen_detector_file3_auc": float(build_roc(y_det, s_det)),
        "frozen_detector_metrics_at_threshold": det_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_positive_metrics": test_metrics,
        "agreement_consistency": agreement,
        "outputs": {
            "confusion_size_7class": str((out_dir / "confusion_size_7class.png").resolve()),
            "confusion_size_3bin": str((out_dir / "confusion_size_3bin.png").resolve()),
            "confusion_depth_3class": str((out_dir / "confusion_depth_3class.png").resolve()),
            "confusion_deep_vs_rest": str((out_dir / "confusion_deep_vs_rest.png").resolve()),
            "roc_inversion_tasks": str((out_dir / "roc_inversion_tasks.png").resolve()),
            "roc_frozen_detector_file3": str((out_dir / "roc_frozen_detector_file3.png").resolve()),
            "cam_contact_sheet": str((out_dir / "cam_residual_inversion_contact_sheet.png").resolve()),
        },
        "config": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
