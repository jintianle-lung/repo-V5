import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"

for path in (PROJECT_ROOT, MODELS_DIR, UTILS_DIR, CURRENT_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


from dual_stream_mstcn_detection import DualStreamMSTCNDetector
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM, WINDOW_STRIDE
from task_protocol_v1 import depth_to_coarse_index, size_to_class_index
from train_detection_oracle_conditioned_3fold import (
    CenterLabelSequenceDataset,
    build_detection_samples_for_file,
    downsample_negatives,
    select_high_sensitivity_threshold,
    split_base_groups_train_val_balanced,
)
from train_triplet_repeat_classifier import build_pr, build_roc, compute_cls_metrics, load_json
from train_triplet_repeat_classifier import select_best_f1_threshold, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="File1+File2 train / File3 test shared CNN+MS-TCN cascaded 3-task model."
    )
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1.2e-3)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--frame-feature-dim", type=int, default=32)
    parser.add_argument("--temporal-channels", type=int, default=48)
    parser.add_argument("--temporal-blocks", type=int, default=2)
    parser.add_argument("--temporal-pooling", choices=["attention", "mean", "max", "center", "last"], default="mean")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--soft-loss-weight", type=float, default=0.0)
    parser.add_argument("--size-cls-weight", type=float, default=0.18)
    parser.add_argument("--size-reg-weight", type=float, default=0.06)
    parser.add_argument("--depth-weight", type=float, default=0.12)
    parser.add_argument("--pos-weight-scale", type=float, default=0.75)
    parser.add_argument("--max-neg-pos-ratio", type=float, default=2.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aug-noise-std", type=float, default=0.015)
    parser.add_argument("--aug-scale-jitter", type=float, default=0.10)
    parser.add_argument("--aug-frame-dropout", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--label-mode", choices=["center_frame_positive", "window_overlap_positive"], default="window_overlap_positive")
    parser.add_argument("--input-normalization", choices=["fixed_global_clipped", "window_minmax"], default="window_minmax")
    parser.add_argument("--high-sensitivity-min-recall", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--file1-labels", type=str, default=str(PROJECT_ROOT / "manual_keyframe_labels.json"))
    parser.add_argument("--file2-labels", type=str, default=str(PROJECT_ROOT / "整理好的数据集" / "建表数据" / "manual_keyframe_labels_file2.json"))
    parser.add_argument("--file3-labels", type=str, default=str(PROJECT_ROOT / "整理好的数据集" / "建表数据" / "manual_keyframe_labels_file3.json"))
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "整理好的数据集" / "建表数据"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "latest_algorithm"
            / "runs"
            / f"shared_cnn_mstcn_cascade_file3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    return parser.parse_args()


class CascadeWindowDataset(Dataset):
    def __init__(self, base_dataset: CenterLabelSequenceDataset, sample_records: Sequence[dict]):
        self.base_dataset = base_dataset
        self.sample_records = list(sample_records)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y_hard, y_soft = self.base_dataset[idx]
        sample = self.sample_records[idx]
        size_idx = int(size_to_class_index(float(sample["size_cm"])))
        depth_idx = int(depth_to_coarse_index(float(sample["depth_cm"])))
        positive_mask = 1.0 if int(sample["label"]) == 1 else 0.0
        return (
            x,
            y_hard,
            y_soft,
            torch.tensor(size_idx, dtype=torch.long),
            torch.tensor(depth_idx, dtype=torch.long),
            torch.tensor(float(sample["size_cm"]), dtype=torch.float32),
            torch.tensor(positive_mask, dtype=torch.float32),
        )


class SharedCNNMSTCNCascade(nn.Module):
    def __init__(
        self,
        frame_feature_dim: int,
        temporal_channels: int,
        temporal_blocks: int,
        temporal_pooling: str,
        dropout: float,
        hidden_dim: int,
    ):
        super().__init__()
        self.backbone = DualStreamMSTCNDetector(
            seq_len=INPUT_SEQ_LEN,
            frame_feature_dim=int(frame_feature_dim),
            temporal_channels=int(temporal_channels),
            temporal_blocks=int(temporal_blocks),
            dropout=float(dropout),
            use_delta_branch=False,
            temporal_pooling=temporal_pooling,
        )
        feat_dim = int(self.backbone.feature_dim)
        hidden_dim = max(int(hidden_dim), feat_dim)
        self.shared_norm = nn.LayerNorm(feat_dim)

        self.size_head = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, len(SIZE_VALUES_CM)),
        )
        self.size_reg_head = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.depth_head = nn.Sequential(
            nn.Linear(feat_dim + 1 + len(SIZE_VALUES_CM) + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, len(COARSE_DEPTH_ORDER)),
        )

        self.register_buffer("size_values", torch.tensor(SIZE_VALUES_CM, dtype=torch.float32).view(1, -1))
        self.size_min = float(min(SIZE_VALUES_CM))
        self.size_span = float(max(SIZE_VALUES_CM) - min(SIZE_VALUES_CM))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        det_logit, feats = self.backbone(x, return_features=True)
        shared = self.shared_norm(feats["pooled_features"])
        det_prob = torch.sigmoid(det_logit)

        size_input = torch.cat([shared, det_prob], dim=1)
        size_logits = self.size_head(size_input)
        size_reg_norm = torch.sigmoid(self.size_reg_head(size_input))
        size_reg_cm = self.size_min + size_reg_norm * max(self.size_span, 1e-6)
        size_probs = torch.softmax(size_logits, dim=1)
        size_expected_cm = torch.sum(size_probs * self.size_values.to(size_probs.device), dim=1, keepdim=True)

        depth_input = torch.cat([shared, det_prob, size_probs, size_expected_cm / max(float(max(SIZE_VALUES_CM)), 1e-6)], dim=1)
        depth_logits = self.depth_head(depth_input)

        if return_features:
            return det_logit, size_logits, size_reg_cm, depth_logits, {
                "shared_features": shared,
                "det_prob": det_prob,
                "size_probs": size_probs,
                "size_expected_cm": size_expected_cm,
            }
        return det_logit, size_logits, size_reg_cm, depth_logits


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return torch.sum(values * mask) / torch.clamp(mask.sum(), min=1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    det_criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    grad_clip: float = 0.0,
    soft_loss_weight: float = 0.0,
    size_cls_weight: float = 0.0,
    size_reg_weight: float = 0.0,
    depth_weight: float = 0.0,
) -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum = 0.0
    n = 0
    all_y: List[np.ndarray] = []
    all_score: List[np.ndarray] = []
    positive_count = 0.0
    size_correct = 0.0
    size_top2_correct = 0.0
    size_abs_err = 0.0
    depth_correct = 0.0

    with torch.set_grad_enabled(is_train):
        for x, y_hard, y_soft, size_idx, depth_idx, size_cm, pos_mask in loader:
            x = x.to(device)
            y_hard = y_hard.to(device).unsqueeze(1)
            y_soft = y_soft.to(device).unsqueeze(1)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            size_cm = size_cm.to(device).unsqueeze(1)
            pos_mask = pos_mask.to(device)

            if is_train:
                optimizer.zero_grad()

            det_logit, size_logits, size_reg_cm, depth_logits = model(x)
            det_loss = det_criterion(det_logit, y_hard)
            soft_loss = det_criterion(det_logit, y_soft) if float(soft_loss_weight) > 0.0 else torch.zeros((), device=device)
            size_cls_loss = masked_mean(F.cross_entropy(size_logits, size_idx, reduction="none"), pos_mask)
            size_reg_loss = masked_mean(F.smooth_l1_loss(size_reg_cm, size_cm, reduction="none").squeeze(1), pos_mask)
            depth_loss = masked_mean(F.cross_entropy(depth_logits, depth_idx, reduction="none"), pos_mask)
            loss = (
                det_loss
                + float(soft_loss_weight) * soft_loss
                + float(size_cls_weight) * size_cls_loss
                + float(size_reg_weight) * size_reg_loss
                + float(depth_weight) * depth_loss
            )

            if is_train:
                loss.backward()
                if float(grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

            bs = int(x.size(0))
            n += bs
            loss_sum += float(loss.item()) * bs
            all_y.append(y_hard.detach().cpu().numpy().reshape(-1))
            all_score.append(torch.sigmoid(det_logit).detach().cpu().numpy().reshape(-1))

            pos_bool = pos_mask > 0.5
            pos_seen = int(pos_bool.sum().item())
            positive_count += float(pos_seen)
            if pos_seen > 0:
                size_pred = torch.argmax(size_logits, dim=1)
                depth_pred = torch.argmax(depth_logits, dim=1)
                size_top2 = torch.topk(size_logits, k=min(2, size_logits.size(1)), dim=1).indices
                size_correct += float((size_pred[pos_bool] == size_idx[pos_bool]).sum().item())
                size_top2_correct += float((size_top2[pos_bool] == size_idx[pos_bool].unsqueeze(1)).any(dim=1).sum().item())
                size_abs_err += float(torch.abs(size_reg_cm[pos_bool] - size_cm[pos_bool]).sum().item())
                depth_correct += float((depth_pred[pos_bool] == depth_idx[pos_bool]).sum().item())

    y_true = np.concatenate(all_y, axis=0).astype(np.int32)
    y_score = np.concatenate(all_score, axis=0).astype(np.float64)
    denom = max(positive_count, 1.0)
    return {
        "loss": float(loss_sum / max(n, 1)),
        "y_true": y_true,
        "y_score": y_score,
        "positive_count": int(positive_count),
        "positive_size_cls_acc": float(size_correct / denom),
        "positive_size_top2_acc": float(size_top2_correct / denom),
        "positive_size_reg_mae_cm": float(size_abs_err / denom),
        "positive_depth_acc": float(depth_correct / denom),
    }


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
    base = CenterLabelSequenceDataset(
        records_by_key,
        sample_records,
        is_train=is_train,
        aug_noise_std=aug_noise_std if is_train else 0.0,
        aug_scale_jitter=aug_scale_jitter if is_train else 0.0,
        aug_frame_dropout=aug_frame_dropout if is_train else 0.0,
        input_normalization=input_normalization,
    )
    return DataLoader(
        CascadeWindowDataset(base, sample_records),
        batch_size=int(batch_size),
        shuffle=bool(is_train),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def summarize_split(sample_records: Sequence[dict]) -> dict:
    return {
        "samples": int(len(sample_records)),
        "positive": int(sum(int(s["label"]) == 1 for s in sample_records)),
        "negative": int(sum(int(s["label"]) == 0 for s in sample_records)),
        "groups": int(len(set(str(s["base_group"]) for s in sample_records))),
    }


def load_file_records(args: argparse.Namespace):
    file1_labels = load_json(args.file1_labels)
    file2_labels = load_json(args.file2_labels)
    file3_labels = load_json(args.file3_labels)
    rec1, samples1 = build_detection_samples_for_file(
        file1_labels, "1.CSV", args.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6, args.label_mode, args.input_normalization
    )
    rec2, samples2 = build_detection_samples_for_file(
        file2_labels, "2.CSV", args.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6, args.label_mode, args.input_normalization
    )
    rec3, samples3 = build_detection_samples_for_file(
        file3_labels, "3.CSV", args.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6, args.label_mode, args.input_normalization
    )
    common_groups = sorted(
        set(v["base_group"] for v in rec1.values())
        & set(v["base_group"] for v in rec2.values())
        & set(v["base_group"] for v in rec3.values())
    )
    common = set(common_groups)
    rec1 = {k: v for k, v in rec1.items() if v["base_group"] in common}
    rec2 = {k: v for k, v in rec2.items() if v["base_group"] in common}
    rec3 = {k: v for k, v in rec3.items() if v["base_group"] in common}
    samples1 = [s for s in samples1 if s["base_group"] in common]
    samples2 = [s for s in samples2 if s["base_group"] in common]
    samples3 = [s for s in samples3 if s["base_group"] in common]
    return rec1, samples1, rec2, samples2, rec3, samples3, common_groups


def write_history_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rec1, samples1, rec2, samples2, rec3, samples3, common_groups = load_file_records(args)
    train_groups, val_groups = split_base_groups_train_val_balanced(common_groups)
    train_set = set(train_groups)
    val_set = set(val_groups)
    records_all = {}
    records_all.update(rec1)
    records_all.update(rec2)
    records_all.update(rec3)

    train_records = {k: v for k, v in records_all.items() if v["file_name"] in {"1.CSV", "2.CSV"} and v["base_group"] in train_set}
    val_records = {k: v for k, v in records_all.items() if v["file_name"] in {"1.CSV", "2.CSV"} and v["base_group"] in val_set}
    test_records = rec3
    train_samples_all = [s for s in samples1 + samples2 if s["base_group"] in train_set]
    train_samples = downsample_negatives(train_samples_all, args.max_neg_pos_ratio, args.seed)
    val_samples = [s for s in samples1 + samples2 if s["base_group"] in val_set]
    test_samples = samples3

    manifest = {
        "created_at": datetime.now().isoformat(),
        "experiment_type": "shared_cnn_mstcn_cascaded_detection_size_depth_file3",
        "label_note": "File3 test labels are loaded from args.file3_labels; default is the active best AUC label.",
        "file1_labels": str(Path(args.file1_labels).resolve()),
        "file2_labels": str(Path(args.file2_labels).resolve()),
        "file3_labels": str(Path(args.file3_labels).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "common_groups": int(len(common_groups)),
        "train_groups": train_groups,
        "val_groups": val_groups,
        "split_summary": {
            "train_all": summarize_split(train_samples_all),
            "train_used": summarize_split(train_samples),
            "val": summarize_split(val_samples),
            "test_file3": summarize_split(test_samples),
        },
        "config": vars(args),
        "cascade": [
            "shared CNN frame encoder + MS-TCN temporal blocks",
            "detection head from shared pooled feature",
            "size head conditioned on shared feature and predicted detection probability",
            "depth head conditioned on shared feature, predicted detection probability, and predicted size distribution",
        ],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        print(f"Dry run saved manifest to: {output_dir / 'manifest.json'}")
        return

    loader_train = make_loader(
        train_records, train_samples, args.batch_size, True, args.num_workers, args.input_normalization,
        args.aug_noise_std, args.aug_scale_jitter, args.aug_frame_dropout
    )
    loader_train_eval = make_loader(
        train_records, train_samples_all, args.batch_size, False, args.num_workers, args.input_normalization, 0.0, 0.0, 0.0
    )
    loader_val = make_loader(
        val_records, val_samples, args.batch_size, False, args.num_workers, args.input_normalization, 0.0, 0.0, 0.0
    )
    loader_test = make_loader(
        test_records, test_samples, args.batch_size, False, args.num_workers, args.input_normalization, 0.0, 0.0, 0.0
    )

    model = SharedCNNMSTCNCascade(
        frame_feature_dim=args.frame_feature_dim,
        temporal_channels=args.temporal_channels,
        temporal_blocks=args.temporal_blocks,
        temporal_pooling=args.temporal_pooling,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
    ).to(device)

    y_train = np.array([int(s["label"]) for s in train_samples], dtype=np.int32)
    pos_weight = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)) * float(args.pos_weight_scale)
    det_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, pos_weight)], device=device))
    eval_det_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=max(args.lr * 0.05, 1e-6))

    history = []
    best_key = None
    best_state = None
    best_epoch = None
    no_improve = 0
    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(
            model, loader_train, device, det_criterion, optimizer, args.grad_clip,
            args.soft_loss_weight, args.size_cls_weight, args.size_reg_weight, args.depth_weight
        )
        va = run_epoch(
            model, loader_val, device, eval_det_criterion, None, 0.0,
            0.0, args.size_cls_weight, args.size_reg_weight, args.depth_weight
        )
        val_best = select_best_f1_threshold(va["y_true"], va["y_score"])
        row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(tr["loss"]),
            "val_loss": float(va["loss"]),
            "val_auc": float(build_roc(va["y_true"], va["y_score"])),
            "val_ap": float(build_pr(va["y_true"], va["y_score"])),
            "val_f1": float(val_best["f1"]),
            "val_precision": float(val_best["precision"]),
            "val_recall": float(val_best["recall"]),
            "val_threshold": float(val_best["threshold"]),
            "val_size_acc": float(va["positive_size_cls_acc"]),
            "val_size_top2_acc": float(va["positive_size_top2_acc"]),
            "val_size_mae_cm": float(va["positive_size_reg_mae_cm"]),
            "val_depth_acc": float(va["positive_depth_acc"]),
        }
        history.append(row)
        scheduler.step(row["val_auc"])
        key = (row["val_auc"], row["val_ap"], row["val_f1"], row["val_depth_acc"], row["val_size_acc"], -row["val_loss"])
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"epoch={epoch:03d} val_auc={row['val_auc']:.4f} val_f1={row['val_f1']:.4f} "
            f"size_acc={row['val_size_acc']:.4f} depth_acc={row['val_depth_acc']:.4f}",
            flush=True,
        )
        if no_improve >= int(args.patience):
            break

    write_history_csv(output_dir / "history.csv", history)
    torch.save(best_state, output_dir / "best_model.pth")
    model.load_state_dict(best_state)

    tr_eval = run_epoch(model, loader_train_eval, device, eval_det_criterion, None, 0.0, 0.0, args.size_cls_weight, args.size_reg_weight, args.depth_weight)
    va = run_epoch(model, loader_val, device, eval_det_criterion, None, 0.0, 0.0, args.size_cls_weight, args.size_reg_weight, args.depth_weight)
    te = run_epoch(model, loader_test, device, eval_det_criterion, None, 0.0, 0.0, args.size_cls_weight, args.size_reg_weight, args.depth_weight)

    val_best = select_best_f1_threshold(va["y_true"], va["y_score"])
    val_high = select_high_sensitivity_threshold(va["y_true"], va["y_score"], min_recall=args.high_sensitivity_min_recall)
    test_at_best = compute_cls_metrics(te["y_true"], te["y_score"], float(val_best["threshold"]))
    test_at_high = compute_cls_metrics(te["y_true"], te["y_score"], float(val_high["threshold"]))
    summary = {
        "created_at": datetime.now().isoformat(),
        "best_epoch": best_epoch,
        "best_epoch_record": history[best_epoch - 1] if best_epoch is not None else None,
        "train_auc": float(build_roc(tr_eval["y_true"], tr_eval["y_score"])),
        "train_ap": float(build_pr(tr_eval["y_true"], tr_eval["y_score"])),
        "val_auc": float(build_roc(va["y_true"], va["y_score"])),
        "val_ap": float(build_pr(va["y_true"], va["y_score"])),
        "test_auc": float(build_roc(te["y_true"], te["y_score"])),
        "test_ap": float(build_pr(te["y_true"], te["y_score"])),
        "val_best_f1_threshold_metrics": val_best,
        "val_high_sensitivity_threshold_metrics": val_high,
        "test_metrics_at_val_best_f1_threshold": test_at_best,
        "test_metrics_at_val_high_sensitivity_threshold": test_at_high,
        "test_positive_count": int(te["positive_count"]),
        "test_positive_size_cls_acc": float(te["positive_size_cls_acc"]),
        "test_positive_size_top2_acc": float(te["positive_size_top2_acc"]),
        "test_positive_size_reg_mae_cm": float(te["positive_size_reg_mae_cm"]),
        "test_positive_depth_acc": float(te["positive_depth_acc"]),
        "config": vars(args),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
