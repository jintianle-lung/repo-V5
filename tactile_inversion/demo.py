"""Run a single-sample R5 inference demo and optionally regenerate Score-CAM."""

from __future__ import annotations

import argparse
import json

import torch

from .evaluate import choose_device, load_release_model
from .generate_same_nodule_task_specific_cam import predict_one
from .make_scorecam import run_scorecam
from .paths import R5_RUN, chdir_release_root
from .train_frozen_detector_residual_inversion import PositiveWindowDataset, build_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sample-index", type=int, default=1155)
    parser.add_argument("--no-scorecam", action="store_true")
    args = parser.parse_args()
    chdir_release_root()
    device = choose_device(args.device)
    model, cfg, threshold, _summary = load_release_model(R5_RUN, device)
    _rec1, _rec2, _rec3, _train_records, _val_records, test_records, _train_samples, _val_samples, test_samples = build_datasets(cfg)
    dataset = PositiveWindowDataset(test_records, test_samples, False)
    idx = min(max(int(args.sample_index), 0), len(dataset) - 1)
    x, _size_idx, _depth_idx, _size_cm, _sample_idx = dataset[idx]
    pred = predict_one(model, x, device)
    sample = dataset.samples[idx]
    result = {
        "sample_index": idx,
        "threshold": threshold,
        "sample": {
            "group_key": sample["group_key"],
            "true_size_cm": float(sample["size_cm"]),
            "true_depth_cm": float(sample["depth_cm"]),
        },
        "prediction": pred,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.no_scorecam:
        with torch.no_grad():
            run_scorecam(device=args.device, output_name="cam_scorecam_demo.png", sample_index=idx)
        print("Score-CAM written to results/r5/cam_scorecam_demo.png")


if __name__ == "__main__":
    main()

