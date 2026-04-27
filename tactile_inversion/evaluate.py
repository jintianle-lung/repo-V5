"""Evaluate the released R5 detector and residual inversion checkpoints."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .paths import R5_RUN, chdir_release_root, resolve_release_path
from .task_protocol_v1 import COARSE_DEPTH_ORDER, SIZE_VALUES_CM, depth_to_coarse_index, size_to_class_index
from .train_frozen_detector_residual_inversion import (
    SIZE_COARSE_NAMES,
    SIZE_TO_COARSE_NP,
    FrozenDetectorResidualInversion,
    PositiveWindowDataset,
    build_datasets,
    class_weights,
    load_frozen_detector,
    metrics_from_pred,
    run_epoch,
)


warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")


SUMMARY_KEYS = (
    "loss",
    "gate_coverage",
    "size_acc",
    "size_top2",
    "size_mae_cm",
    "depth_acc",
    "depth_top2",
    "deep_vs_rest_auc",
    "gated_n",
    "gated_size_acc",
    "gated_size_top2",
    "gated_size_mae_cm",
    "gated_depth_acc",
    "gated_depth_top2",
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def summarize_metrics(metrics: dict) -> dict:
    return {key: metrics[key] for key in SUMMARY_KEYS if key in metrics}


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[evaluate] CUDA requested but unavailable; falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def load_release_model(residual_run_dir: str | Path = R5_RUN, device: torch.device | None = None):
    chdir_release_root()
    device = device or choose_device("cpu")
    residual_run = resolve_release_path(residual_run_dir)
    summary = json.loads((residual_run / "summary.json").read_text(encoding="utf-8"))
    detector_run = resolve_release_path(summary["config"]["run_dir"])
    cfg, frozen, threshold = load_frozen_detector(detector_run, device)
    model = FrozenDetectorResidualInversion(
        frozen,
        int(summary["config"].get("morph_dim", 64)),
        int(summary["config"].get("hidden_dim", 160)),
        float(summary["config"].get("dropout", 0.28)),
        str(summary["config"].get("size_reg_mode", "expected_residual")),
        float(summary["config"].get("size_residual_span", 0.35)),
        str(summary["config"].get("depth_conditioning", "size7_coarse")),
    ).to(device)
    model.load_state_dict(torch.load(residual_run / "best_model.pth", map_location=device))
    model.eval()
    model.detector.eval()
    return model, cfg, threshold, summary


def evaluate_release(residual_run_dir: str | Path, device_name: str, batch_size: int, full: bool = False) -> dict:
    device = choose_device(device_name)
    model, cfg, threshold, summary = load_release_model(residual_run_dir, device)
    _rec1, _rec2, _rec3, train_records, _val_records, test_records, train_samples, _val_samples, test_samples = build_datasets(cfg)
    ds_train = PositiveWindowDataset(train_records, train_samples, False)
    ds_test = PositiveWindowDataset(test_records, test_samples, False)
    train_size = np.array([size_to_class_index(float(s["size_cm"])) for s in ds_train.samples], dtype=np.int32)
    train_depth = np.array([depth_to_coarse_index(float(s["depth_cm"])) for s in ds_train.samples], dtype=np.int32)
    weights = {
        "size": class_weights(train_size, len(SIZE_VALUES_CM)).to(device),
        "size_coarse": class_weights(SIZE_TO_COARSE_NP[train_size], len(SIZE_COARSE_NAMES)).to(device),
        "depth": class_weights(train_depth, len(COARSE_DEPTH_ORDER)).to(device),
    }
    loss_weights = {
        "size_cls": float(summary["config"].get("size_cls_weight", 0.9)),
        "size_coarse": float(summary["config"].get("size_coarse_weight", 0.45)),
        "size_reg": float(summary["config"].get("size_reg_weight", 0.65)),
        "depth_cls": float(summary["config"].get("depth_cls_weight", 1.05)),
        "depth_binary": float(summary["config"].get("depth_binary_weight", 0.5)),
    }
    pred = run_epoch(
        model,
        DataLoader(ds_test, batch_size=batch_size, shuffle=False),
        device,
        None,
        loss_weights,
        weights,
        gate_loss_alpha=float(summary["config"].get("gate_loss_alpha", 0.15)),
        gate_loss_min=float(summary["config"].get("gate_loss_min", 0.55)),
    )
    metrics = metrics_from_pred(pred, threshold)
    result_metrics = metrics if full else summarize_metrics(metrics)
    return {
        "threshold": threshold,
        "n_positive_windows": len(ds_test),
        "metrics": result_metrics,
        "released_reference_metrics": summarize_metrics(summary.get("test_positive_metrics", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-run-dir", default=str(R5_RUN.relative_to(R5_RUN.parents[1])))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--full", action="store_true", help="Print full arrays/logits in addition to summary metrics.")
    args = parser.parse_args()
    result = evaluate_release(args.residual_run_dir, args.device, args.batch_size, args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
