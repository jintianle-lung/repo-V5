import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"

for path in (PROJECT_ROOT, MODELS_DIR, UTILS_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


from dual_stream_mstcn_detection import DualStreamMSTCNDetector
from input_normalization_v1 import normalize_raw_frames_global, normalize_raw_frames_window_minmax
from task_protocol_v1 import (
    COARSE_DEPTH_ORDER,
    INPUT_SEQ_LEN,
    SIZE_VALUES_CM,
    WINDOW_STRIDE,
    depth_to_coarse_index,
    size_to_class_index,
)
from train_triplet_repeat_classifier import (
    build_pr,
    build_roc,
    compress_samples_by_gap,
    compute_cls_metrics,
    filter_labels_for_file,
    load_json,
    parse_float_from_cm_text,
    parse_size_depth_from_group,
    read_csv_data,
    select_best_f1_threshold,
    sanitize_segments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory oracle-conditioned detection with 3-fold group CV."
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1.2e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--frame-feature-dim", type=int, default=32)
    parser.add_argument("--temporal-channels", type=int, default=64)
    parser.add_argument("--temporal-blocks", type=int, default=3)
    parser.add_argument("--soft-loss-weight", type=float, default=0.10)
    parser.add_argument("--size-aux-weight", type=float, default=0.15)
    parser.add_argument("--depth-aux-weight", type=float, default=0.10)
    parser.add_argument("--cond-embed-dim", type=int, default=16)
    parser.add_argument("--fusion-hidden-dim", type=int, default=96)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-neg-pos-ratio", type=float, default=2.5)
    parser.add_argument("--aug-noise-std", type=float, default=0.015)
    parser.add_argument("--aug-scale-jitter", type=float, default=0.10)
    parser.add_argument("--aug-frame-dropout", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--label-mode", choices=["center_frame_positive", "window_overlap_positive"], default="window_overlap_positive")
    parser.add_argument("--input-normalization", choices=["fixed_global_clipped", "window_minmax"], default="window_minmax")
    parser.add_argument("--high-sensitivity-min-recall", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "latest_algorithm" / "runs" / f"oracle_conditioned_detection_3fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    return parser.parse_args()


@dataclass
class FoldStats:
    fold_index: int
    train_groups: int
    val_groups: int
    test_groups: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_positive: int
    val_positive: int
    test_positive: int


def is_frame_positive(frame_idx: int, segments: Sequence[Tuple[int, int]]) -> int:
    for start, end in segments:
        if int(start) <= int(frame_idx) < int(end):
            return 1
    return 0


def is_center_positive(center_idx: int, segments: Sequence[Tuple[int, int]]) -> int:
    return is_frame_positive(center_idx, segments)


def compute_positive_fraction(window_start: int, window_end_inclusive: int, segments: Sequence[Tuple[int, int]]) -> float:
    total = max(0, int(window_end_inclusive) - int(window_start) + 1)
    if total <= 0:
        return 0.0
    positive = 0
    for frame_idx in range(int(window_start), int(window_end_inclusive) + 1):
        if is_frame_positive(frame_idx, segments):
            positive += 1
    return float(positive / total)


def is_window_overlap_positive(window_start: int, window_end_inclusive: int, segments: Sequence[Tuple[int, int]]) -> int:
    return int(compute_positive_fraction(window_start, window_end_inclusive, segments) > 0.0)


def build_detection_samples_for_file(
    label_map: Dict,
    target_file: str,
    data_root: str,
    seq_len: int,
    stride: int,
    dedup_gap: int,
    label_mode: str = "window_overlap_positive",
    input_normalization: str = "window_minmax",
) -> Tuple[Dict, List[dict]]:
    label_mode = str(label_mode).strip().lower()
    input_normalization = str(input_normalization).strip().lower()
    if label_mode not in {"center_frame_positive", "window_overlap_positive"}:
        raise ValueError(f"Unsupported label_mode={label_mode}")
    if input_normalization not in {"fixed_global_clipped", "window_minmax"}:
        raise ValueError(f"Unsupported input_normalization={input_normalization}")

    filtered = filter_labels_for_file(label_map, target_file)
    records_by_key: Dict[str, dict] = {}
    sample_records: List[dict] = []

    for base_group in sorted(filtered.keys()):
        size_text, depth_text = parse_size_depth_from_group(base_group)
        file_path = os.path.join(data_root, size_text, depth_text, target_file)
        if not os.path.exists(file_path):
            continue

        raw = read_csv_data(file_path)
        n = len(raw)
        if n < seq_len:
            continue

        segments = sanitize_segments(filtered[base_group]["segments"], n)
        raw_frames = np.asarray(raw[:n], dtype=np.float32)
        group_key = f"{base_group}|{target_file}"
        rec = {
            "raw_frames": raw_frames,
            "seq_len": int(seq_len),
            "file_name": target_file,
            "base_group": base_group,
            "size": size_text,
            "depth": depth_text,
            "n_frames": int(n),
            "segments": segments,
            "input_normalization": input_normalization,
        }
        if input_normalization == "fixed_global_clipped":
            rec["frames"] = normalize_raw_frames_global(raw_frames)
        records_by_key[group_key] = rec

        for end in range(seq_len - 1, n, stride):
            st = end - seq_len + 1
            seq = raw[st : end + 1]
            if np.isnan(seq).any() or np.all(seq == 0):
                continue

            center_idx = st + (seq_len // 2)
            soft_label = compute_positive_fraction(st, end, segments)
            if label_mode == "center_frame_positive":
                det_label = is_center_positive(center_idx, segments)
            else:
                det_label = is_window_overlap_positive(st, end, segments)
            sample_records.append(
                {
                    "group_key": group_key,
                    "end_row": int(end),
                    "center_row": int(center_idx),
                    "label": int(det_label),
                    "soft_label": float(soft_label),
                    "base_group": base_group,
                    "file_name": target_file,
                    "size_text": size_text,
                    "depth_text": depth_text,
                    "size_cm": float(parse_float_from_cm_text(size_text)),
                    "depth_cm": float(parse_float_from_cm_text(depth_text)),
                }
            )

    sample_records = compress_samples_by_gap(sample_records, min_gap=dedup_gap)
    return records_by_key, sample_records


def downsample_negatives(sample_records: Sequence[dict], max_neg_pos_ratio: float, seed: int) -> List[dict]:
    pos = [s for s in sample_records if int(s["label"]) == 1]
    neg = [s for s in sample_records if int(s["label"]) == 0]
    max_neg = int(round(len(pos) * float(max_neg_pos_ratio)))
    if len(pos) > 0 and len(neg) > max_neg > 0:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(neg), size=max_neg, replace=False)
        neg = [neg[i] for i in keep_idx]
    out = pos + neg
    rng = np.random.default_rng(seed + 17)
    rng.shuffle(out)
    return out


def split_base_groups_train_val_balanced(groups: Sequence[str]) -> Tuple[List[str], List[str]]:
    buckets: Dict[str, List[str]] = {}
    for group in groups:
        size_text, _depth_text = parse_size_depth_from_group(group)
        buckets.setdefault(size_text, []).append(group)

    train, val = [], []
    size_keys = sorted(buckets.keys(), key=lambda s: parse_float_from_cm_text(s))
    for size_idx, size_text in enumerate(size_keys):
        arr = sorted(buckets[size_text], key=lambda g: parse_float_from_cm_text(parse_size_depth_from_group(g)[1]))
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
        for idx, group in enumerate(arr):
            if idx == val_idx:
                val.append(group)
            else:
                train.append(group)
    if len(val) == 0 and len(train) > 1:
        val.append(train.pop(0))
    return train, val


class CenterLabelSequenceDataset(Dataset):
    def __init__(
        self,
        records_by_key: Dict,
        sample_records: List[dict],
        is_train: bool = False,
        aug_noise_std: float = 0.0,
        aug_scale_jitter: float = 0.0,
        aug_frame_dropout: float = 0.0,
        input_normalization: str = "window_minmax",
    ):
        self.records_by_key = records_by_key
        self.samples = sample_records
        self.is_train = bool(is_train)
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_scale_jitter = float(max(0.0, aug_scale_jitter))
        self.aug_frame_dropout = float(min(max(0.0, aug_frame_dropout), 0.5))
        self.input_normalization = str(input_normalization).strip().lower()

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_window(self, sample_index: int) -> np.ndarray:
        sample = self.samples[sample_index]
        key = sample["group_key"]
        end_row = int(sample["end_row"])
        rec = self.records_by_key[key]
        seq_len = int(rec["seq_len"])
        st = end_row - seq_len + 1
        if "raw_frames" in rec:
            raw_window = rec["raw_frames"][st : end_row + 1].astype(np.float32)
            norm_mode = str(rec.get("input_normalization", self.input_normalization)).strip().lower()
            if norm_mode == "window_minmax":
                x = normalize_raw_frames_window_minmax(raw_window)
            elif norm_mode == "fixed_global_clipped":
                x = normalize_raw_frames_global(raw_window)
            else:
                raise ValueError(f"Unsupported input_normalization={norm_mode}")
        else:
            x = rec["frames"][st : end_row + 1].astype(np.float32)
        return np.expand_dims(x, axis=1)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        y_hard = float(sample["label"])
        y_soft = float(np.clip(sample.get("soft_label", y_hard), 0.0, 1.0))
        x = self._extract_window(idx)
        if self.is_train:
            if self.aug_scale_jitter > 0.0:
                scale = 1.0 + float(np.random.uniform(-self.aug_scale_jitter, self.aug_scale_jitter))
                x = x * scale
            if self.aug_noise_std > 0.0:
                x = x + np.random.normal(loc=0.0, scale=self.aug_noise_std, size=x.shape).astype(np.float32)
            if self.aug_frame_dropout > 0.0:
                keep = (np.random.rand(x.shape[0], 1, 1, 1) >= self.aug_frame_dropout).astype(np.float32)
                x = x * keep
            x = np.clip(x, 0.0, 1.0)
        return (
            torch.from_numpy(x),
            torch.tensor(y_hard, dtype=torch.float32),
            torch.tensor(y_soft, dtype=torch.float32),
        )


class OracleConditionedDataset(Dataset):
    def __init__(self, base_dataset: CenterLabelSequenceDataset, sample_records: Sequence[dict]):
        self.base_dataset = base_dataset
        self.sample_records = list(sample_records)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y_hard, y_soft = self.base_dataset[idx]
        sample = self.sample_records[idx]
        size_idx = int(size_to_class_index(float(sample["size_cm"])))
        coarse_depth_idx = int(depth_to_coarse_index(float(sample["depth_cm"])))
        return (
            x,
            y_hard,
            y_soft,
            torch.tensor(size_idx, dtype=torch.long),
            torch.tensor(coarse_depth_idx, dtype=torch.long),
        )


class OracleConditionedDetector(nn.Module):
    def __init__(
        self,
        frame_feature_dim: int = 32,
        temporal_channels: int = 64,
        temporal_blocks: int = 3,
        dropout: float = 0.35,
        cond_embed_dim: int = 16,
        fusion_hidden_dim: int = 96,
    ):
        super().__init__()
        self.backbone = DualStreamMSTCNDetector(
            seq_len=INPUT_SEQ_LEN,
            frame_feature_dim=int(frame_feature_dim),
            temporal_channels=int(temporal_channels),
            temporal_blocks=int(temporal_blocks),
            dropout=float(dropout),
            use_delta_branch=False,
        )
        feat_dim = int(self.backbone.feature_dim)
        self.size_embedding = nn.Embedding(len(SIZE_VALUES_CM), int(cond_embed_dim))
        self.depth_embedding = nn.Embedding(len(COARSE_DEPTH_ORDER), int(cond_embed_dim))
        self.cond_proj = nn.Sequential(
            nn.Linear(int(cond_embed_dim) * 2 + 2, int(fusion_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fusion_hidden_dim), feat_dim * 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, int(fusion_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fusion_hidden_dim), feat_dim),
            nn.ReLU(inplace=True),
        )
        self.det_head = nn.Sequential(
            nn.Linear(feat_dim, max(32, feat_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(max(32, feat_dim // 2), 1),
        )
        self.size_aux_head = nn.Linear(feat_dim, len(SIZE_VALUES_CM))
        self.depth_aux_head = nn.Linear(feat_dim, len(COARSE_DEPTH_ORDER))
        self.feature_dim = feat_dim

    def forward(self, x: torch.Tensor, size_idx: torch.Tensor, depth_idx: torch.Tensor, return_features: bool = False):
        _baseline_logit, feats = self.backbone(x, return_features=True)
        pooled = feats["pooled_features"]
        size_emb = self.size_embedding(size_idx)
        depth_emb = self.depth_embedding(depth_idx)
        size_norm = size_idx.float().unsqueeze(1) / float(max(len(SIZE_VALUES_CM) - 1, 1))
        depth_norm = depth_idx.float().unsqueeze(1) / float(max(len(COARSE_DEPTH_ORDER) - 1, 1))
        cond_vec = torch.cat([size_emb, depth_emb, size_norm, depth_norm], dim=1)
        gamma_beta = self.cond_proj(cond_vec)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        modulated = pooled * (1.0 + 0.5 * torch.tanh(gamma)) + beta
        cross_term = pooled * torch.sigmoid(modulated)
        fused = self.fusion(torch.cat([pooled, modulated, cross_term], dim=1))
        det_logit = self.det_head(fused)
        size_aux = self.size_aux_head(pooled)
        depth_aux = self.depth_aux_head(pooled)
        if return_features:
            return det_logit, size_aux, depth_aux, {
                "pooled_features": pooled,
                "fused_features": fused,
                "attn_weights": feats.get("attn_weights"),
            }
        return det_logit, size_aux, depth_aux


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    det_criterion: nn.Module,
    size_criterion: nn.Module,
    depth_criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    grad_clip: float = 0.0,
    soft_loss_weight: float = 0.0,
    size_aux_weight: float = 0.0,
    depth_aux_weight: float = 0.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_y = []
    all_score = []
    loss_sum = 0.0
    n = 0
    with torch.set_grad_enabled(is_train):
        for x, y_hard, y_soft, size_idx, depth_idx in loader:
            x = x.to(device)
            y_hard = y_hard.to(device).unsqueeze(1)
            y_soft = y_soft.to(device).unsqueeze(1)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            if is_train:
                optimizer.zero_grad()

            det_logit, size_aux, depth_aux = model(x, size_idx, depth_idx)
            det_loss = det_criterion(det_logit, y_hard)
            soft_loss = det_criterion(det_logit, y_soft) if float(soft_loss_weight) > 0.0 else torch.zeros((), device=device)
            size_loss = size_criterion(size_aux, size_idx)
            depth_loss = depth_criterion(depth_aux, depth_idx)
            loss = (
                det_loss
                + float(soft_loss_weight) * soft_loss
                + float(size_aux_weight) * size_loss
                + float(depth_aux_weight) * depth_loss
            )

            if is_train:
                loss.backward()
                if float(grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

            bs = x.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs
            all_y.append(y_hard.detach().cpu().numpy().reshape(-1))
            all_score.append(torch.sigmoid(det_logit).detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_y, axis=0).astype(np.int32)
    y_score = np.concatenate(all_score, axis=0).astype(np.float64)
    return {
        "loss": float(loss_sum / max(n, 1)),
        "y_true": y_true,
        "y_score": y_score,
    }


def select_high_sensitivity_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float,
) -> dict:
    best = None
    min_recall = float(np.clip(min_recall, 0.0, 1.0))
    for thr in np.linspace(0.0, 1.0, 1001):
        metrics = compute_cls_metrics(y_true, y_score, float(thr))
        if metrics["recall"] + 1e-12 < min_recall:
            continue
        if best is None:
            best = metrics
            continue
        cur_key = (metrics["f1"], metrics["precision"], metrics["specificity"], -metrics["threshold"])
        best_key = (best["f1"], best["precision"], best["specificity"], -best["threshold"])
        if cur_key > best_key:
            best = metrics
    if best is not None:
        return best

    fallback = None
    for thr in np.linspace(0.0, 1.0, 1001):
        metrics = compute_cls_metrics(y_true, y_score, float(thr))
        if fallback is None:
            fallback = metrics
            continue
        cur_key = (metrics["recall"], metrics["f1"], metrics["precision"], -metrics["threshold"])
        fallback_key = (fallback["recall"], fallback["f1"], fallback["precision"], -fallback["threshold"])
        if cur_key > fallback_key:
            fallback = metrics
    return fallback


def group_sort_key(group_name: str) -> Tuple[float, float]:
    size_text, depth_text = group_name.split("|")
    return (float(size_text.replace("cm大", "")), float(depth_text.replace("cm深", "")))


def build_size_balanced_3fold(groups: Sequence[str]) -> List[List[str]]:
    buckets: Dict[str, List[str]] = {}
    for group in groups:
        size_text, _depth_text = group.split("|")
        buckets.setdefault(size_text, []).append(group)

    folds = [[] for _ in range(3)]
    for size_order, size_text in enumerate(sorted(buckets.keys(), key=lambda x: float(x.replace("cm大", "")))):
        arr = sorted(buckets[size_text], key=group_sort_key)
        if not arr:
            continue
        shift = size_order % len(arr)
        arr = arr[shift:] + arr[:shift]
        for idx, group in enumerate(arr):
            folds[idx % 3].append(group)

    for fold in folds:
        fold.sort(key=group_sort_key)
    return folds


def load_all_common_records_and_samples(
    data_root: str,
    label_mode: str,
    input_normalization: str,
    dedup_gap: int = 6,
) -> Tuple[Dict[str, dict], List[dict], List[str]]:
    file1_labels = load_json(str(PROJECT_ROOT / "manual_keyframe_labels.json"))
    file2_labels = load_json(str(Path(data_root) / "manual_keyframe_labels_file2.json"))
    file3_labels = load_json(str(Path(data_root) / "manual_keyframe_labels_file3.json"))

    rec1, samples1 = build_detection_samples_for_file(
        file1_labels,
        "1.CSV",
        data_root,
        INPUT_SEQ_LEN,
        WINDOW_STRIDE,
        dedup_gap,
        label_mode,
        input_normalization,
    )
    rec2, samples2 = build_detection_samples_for_file(
        file2_labels,
        "2.CSV",
        data_root,
        INPUT_SEQ_LEN,
        WINDOW_STRIDE,
        dedup_gap,
        label_mode,
        input_normalization,
    )
    rec3, samples3 = build_detection_samples_for_file(
        file3_labels,
        "3.CSV",
        data_root,
        INPUT_SEQ_LEN,
        WINDOW_STRIDE,
        dedup_gap,
        label_mode,
        input_normalization,
    )

    common_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        ),
        key=group_sort_key,
    )
    common_group_set = set(common_groups)

    all_records = {}
    for records in (rec1, rec2, rec3):
        all_records.update({k: v for k, v in records.items() if v["base_group"] in common_group_set})
    all_samples = [
        s for s in (samples1 + samples2 + samples3) if s["base_group"] in common_group_set
    ]
    return all_records, all_samples, common_groups


def make_loader(
    records_by_key: Dict[str, dict],
    sample_records: List[dict],
    batch_size: int,
    is_train: bool,
    num_workers: int,
    input_normalization: str,
    aug_noise_std: float,
    aug_scale_jitter: float,
    aug_frame_dropout: float,
) -> DataLoader:
    base_dataset = CenterLabelSequenceDataset(
        records_by_key,
        sample_records,
        is_train=is_train,
        aug_noise_std=aug_noise_std if is_train else 0.0,
        aug_scale_jitter=aug_scale_jitter if is_train else 0.0,
        aug_frame_dropout=aug_frame_dropout if is_train else 0.0,
        input_normalization=input_normalization,
    )
    dataset = OracleConditionedDataset(base_dataset, sample_records)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(is_train),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def summarize_fold(
    fold_index: int,
    train_groups: Sequence[str],
    val_groups: Sequence[str],
    test_groups: Sequence[str],
    train_samples: Sequence[dict],
    val_samples: Sequence[dict],
    test_samples: Sequence[dict],
) -> FoldStats:
    return FoldStats(
        fold_index=int(fold_index),
        train_groups=int(len(train_groups)),
        val_groups=int(len(val_groups)),
        test_groups=int(len(test_groups)),
        train_samples=int(len(train_samples)),
        val_samples=int(len(val_samples)),
        test_samples=int(len(test_samples)),
        train_positive=int(sum(int(s["label"]) for s in train_samples)),
        val_positive=int(sum(int(s["label"]) for s in val_samples)),
        test_positive=int(sum(int(s["label"]) for s in test_samples)),
    )


def aggregate_fold_metrics(fold_results: Sequence[dict]) -> dict:
    aggregate = {}
    scalar_keys = [
        "test_auc",
        "test_ap",
        "test_f1_at_val_best_f1_threshold",
        "test_recall_at_val_best_f1_threshold",
        "test_precision_at_val_best_f1_threshold",
        "test_f1_at_val_high_sensitivity_threshold",
        "test_recall_at_val_high_sensitivity_threshold",
        "test_precision_at_val_high_sensitivity_threshold",
    ]
    for key in scalar_keys:
        values = [float(item[key]) for item in fold_results]
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }
    return aggregate


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = str(PROJECT_ROOT / "整理好的数据集" / "建表数据")

    all_records, all_samples, common_groups = load_all_common_records_and_samples(
        data_root=data_root,
        label_mode=args.label_mode,
        input_normalization=args.input_normalization,
        dedup_gap=6,
    )
    folds = build_size_balanced_3fold(common_groups)

    protocol = {
        "experiment_type": "oracle_conditioned_detection_ablation",
        "fair_mainline_replacement": False,
        "notes": [
            "Uses known size/depth priors as oracle conditions.",
            "Should not replace the locked mainline detection claim.",
            "High-sensitivity threshold is reported as a separate operating point.",
        ],
        "num_common_groups": int(len(common_groups)),
        "folds": folds,
    }
    with open(output_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        preview = []
        for fold_index in range(3):
            test_groups = folds[fold_index]
            trainval_groups = [g for i, fold in enumerate(folds) if i != fold_index for g in fold]
            train_groups, val_groups = split_base_groups_train_val_balanced(trainval_groups)
            train_group_set = set(train_groups)
            val_group_set = set(val_groups)
            test_group_set = set(test_groups)
            train_samples_all = [s for s in all_samples if s["base_group"] in train_group_set]
            train_samples = downsample_negatives(train_samples_all, args.max_neg_pos_ratio, args.seed + fold_index)
            val_samples = [s for s in all_samples if s["base_group"] in val_group_set]
            test_samples = [s for s in all_samples if s["base_group"] in test_group_set]
            preview.append(
                asdict(
                    summarize_fold(
                        fold_index,
                        train_groups,
                        val_groups,
                        test_groups,
                        train_samples,
                        val_samples,
                        test_samples,
                    )
                )
            )
        with open(output_dir / "dry_run_split_preview.json", "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)
        print(f"Dry run complete. Preview saved to: {output_dir / 'dry_run_split_preview.json'}")
        return

    fold_results = []
    for fold_index in range(3):
        fold_dir = output_dir / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_groups = folds[fold_index]
        trainval_groups = [g for i, fold in enumerate(folds) if i != fold_index for g in fold]
        train_groups, val_groups = split_base_groups_train_val_balanced(trainval_groups)
        train_group_set = set(train_groups)
        val_group_set = set(val_groups)
        test_group_set = set(test_groups)

        train_records = {k: v for k, v in all_records.items() if v["base_group"] in train_group_set}
        val_records = {k: v for k, v in all_records.items() if v["base_group"] in val_group_set}
        test_records = {k: v for k, v in all_records.items() if v["base_group"] in test_group_set}

        train_samples_all = [s for s in all_samples if s["base_group"] in train_group_set]
        train_samples = downsample_negatives(train_samples_all, args.max_neg_pos_ratio, args.seed + fold_index)
        val_samples = [s for s in all_samples if s["base_group"] in val_group_set]
        test_samples = [s for s in all_samples if s["base_group"] in test_group_set]

        fold_manifest = asdict(
            summarize_fold(
                fold_index=fold_index + 1,
                train_groups=train_groups,
                val_groups=val_groups,
                test_groups=test_groups,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
            )
        )
        fold_manifest["train_groups_detail"] = train_groups
        fold_manifest["val_groups_detail"] = val_groups
        fold_manifest["test_groups_detail"] = list(test_groups)
        with open(fold_dir / "fold_manifest.json", "w", encoding="utf-8") as f:
            json.dump(fold_manifest, f, ensure_ascii=False, indent=2)

        model = OracleConditionedDetector(
            frame_feature_dim=args.frame_feature_dim,
            temporal_channels=args.temporal_channels,
            temporal_blocks=args.temporal_blocks,
            dropout=args.dropout,
            cond_embed_dim=args.cond_embed_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
        ).to(device)

        loader_train = make_loader(
            records_by_key=train_records,
            sample_records=train_samples,
            batch_size=args.batch_size,
            is_train=True,
            num_workers=args.num_workers,
            input_normalization=args.input_normalization,
            aug_noise_std=args.aug_noise_std,
            aug_scale_jitter=args.aug_scale_jitter,
            aug_frame_dropout=args.aug_frame_dropout,
        )
        loader_train_eval = make_loader(
            records_by_key=train_records,
            sample_records=train_samples_all,
            batch_size=args.batch_size,
            is_train=False,
            num_workers=args.num_workers,
            input_normalization=args.input_normalization,
            aug_noise_std=0.0,
            aug_scale_jitter=0.0,
            aug_frame_dropout=0.0,
        )
        loader_val = make_loader(
            records_by_key=val_records,
            sample_records=val_samples,
            batch_size=args.batch_size,
            is_train=False,
            num_workers=args.num_workers,
            input_normalization=args.input_normalization,
            aug_noise_std=0.0,
            aug_scale_jitter=0.0,
            aug_frame_dropout=0.0,
        )
        loader_test = make_loader(
            records_by_key=test_records,
            sample_records=test_samples,
            batch_size=args.batch_size,
            is_train=False,
            num_workers=args.num_workers,
            input_normalization=args.input_normalization,
            aug_noise_std=0.0,
            aug_scale_jitter=0.0,
            aug_frame_dropout=0.0,
        )

        y_train = np.array([int(s["label"]) for s in train_samples], dtype=np.int32)
        pos_weight = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
        det_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, pos_weight)], device=device))
        aux_det_criterion = nn.BCEWithLogitsLoss()
        size_criterion = nn.CrossEntropyLoss()
        depth_criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=max(args.lr * 0.05, 1e-6),
        )

        history = []
        best_key = None
        best_state = None
        best_rec = None
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(
                model=model,
                loader=loader_train,
                device=device,
                det_criterion=det_criterion,
                size_criterion=size_criterion,
                depth_criterion=depth_criterion,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                soft_loss_weight=args.soft_loss_weight,
                size_aux_weight=args.size_aux_weight,
                depth_aux_weight=args.depth_aux_weight,
            )
            tr_eval = run_epoch(
                model=model,
                loader=loader_train_eval,
                device=device,
                det_criterion=aux_det_criterion,
                size_criterion=size_criterion,
                depth_criterion=depth_criterion,
                optimizer=None,
                soft_loss_weight=0.0,
                size_aux_weight=args.size_aux_weight,
                depth_aux_weight=args.depth_aux_weight,
            )
            va = run_epoch(
                model=model,
                loader=loader_val,
                device=device,
                det_criterion=aux_det_criterion,
                size_criterion=size_criterion,
                depth_criterion=depth_criterion,
                optimizer=None,
                soft_loss_weight=0.0,
                size_aux_weight=args.size_aux_weight,
                depth_aux_weight=args.depth_aux_weight,
            )

            best_thr_val = select_best_f1_threshold(va["y_true"], va["y_score"])
            auc_val = build_roc(va["y_true"], va["y_score"])
            ap_val = build_pr(va["y_true"], va["y_score"])
            rec = {
                "epoch": int(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(tr["loss"]),
                "train_eval_loss": float(tr_eval["loss"]),
                "val_loss": float(va["loss"]),
                "val_f1": float(best_thr_val["f1"]),
                "val_precision": float(best_thr_val["precision"]),
                "val_recall": float(best_thr_val["recall"]),
                "val_best_threshold": float(best_thr_val["threshold"]),
                "val_auc": float(auc_val),
                "val_ap": float(ap_val),
            }
            history.append(rec)
            scheduler.step(rec["val_auc"])
            cur_key = (rec["val_auc"], rec["val_ap"], rec["val_f1"], -rec["val_loss"])
            if best_key is None or cur_key > best_key:
                best_key = cur_key
                best_rec = rec
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                break

        torch.save(best_state, fold_dir / "best_model.pth")
        with open(fold_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        model.load_state_dict(best_state)
        tr_eval = run_epoch(
            model=model,
            loader=loader_train_eval,
            device=device,
            det_criterion=aux_det_criterion,
            size_criterion=size_criterion,
            depth_criterion=depth_criterion,
            optimizer=None,
            size_aux_weight=args.size_aux_weight,
            depth_aux_weight=args.depth_aux_weight,
        )
        va = run_epoch(
            model=model,
            loader=loader_val,
            device=device,
            det_criterion=aux_det_criterion,
            size_criterion=size_criterion,
            depth_criterion=depth_criterion,
            optimizer=None,
            size_aux_weight=args.size_aux_weight,
            depth_aux_weight=args.depth_aux_weight,
        )
        te = run_epoch(
            model=model,
            loader=loader_test,
            device=device,
            det_criterion=aux_det_criterion,
            size_criterion=size_criterion,
            depth_criterion=depth_criterion,
            optimizer=None,
            size_aux_weight=args.size_aux_weight,
            depth_aux_weight=args.depth_aux_weight,
        )

        val_best = select_best_f1_threshold(va["y_true"], va["y_score"])
        val_high_sens = select_high_sensitivity_threshold(
            va["y_true"], va["y_score"], min_recall=args.high_sensitivity_min_recall
        )
        test_at_val_best = compute_cls_metrics(te["y_true"], te["y_score"], float(val_best["threshold"]))
        test_at_val_high_sens = compute_cls_metrics(te["y_true"], te["y_score"], float(val_high_sens["threshold"]))

        fold_summary = {
            "fold_index": int(fold_index + 1),
            "best_epoch_record": best_rec,
            "train_auc": float(build_roc(tr_eval["y_true"], tr_eval["y_score"])),
            "train_ap": float(build_pr(tr_eval["y_true"], tr_eval["y_score"])),
            "val_auc": float(build_roc(va["y_true"], va["y_score"])),
            "val_ap": float(build_pr(va["y_true"], va["y_score"])),
            "test_auc": float(build_roc(te["y_true"], te["y_score"])),
            "test_ap": float(build_pr(te["y_true"], te["y_score"])),
            "val_best_f1_threshold_metrics": val_best,
            "val_high_sensitivity_threshold_metrics": val_high_sens,
            "test_metrics_at_val_best_f1_threshold": test_at_val_best,
            "test_metrics_at_val_high_sensitivity_threshold": test_at_val_high_sens,
            "test_f1_at_val_best_f1_threshold": float(test_at_val_best["f1"]),
            "test_recall_at_val_best_f1_threshold": float(test_at_val_best["recall"]),
            "test_precision_at_val_best_f1_threshold": float(test_at_val_best["precision"]),
            "test_f1_at_val_high_sensitivity_threshold": float(test_at_val_high_sens["f1"]),
            "test_recall_at_val_high_sensitivity_threshold": float(test_at_val_high_sens["recall"]),
            "test_precision_at_val_high_sensitivity_threshold": float(test_at_val_high_sens["precision"]),
            "config": vars(args),
            "protocol_note": "oracle-conditioned upper-bound exploration; not a fair replacement for locked mainline",
        }
        with open(fold_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, ensure_ascii=False, indent=2)
        fold_results.append(fold_summary)

    overall_summary = {
        "created_at": datetime.now().isoformat(),
        "config": vars(args),
        "protocol": protocol,
        "aggregate": aggregate_fold_metrics(fold_results),
        "folds": fold_results,
    }
    with open(output_dir / "summary_all_folds.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved overall summary to: {output_dir / 'summary_all_folds.json'}")


if __name__ == "__main__":
    main()
