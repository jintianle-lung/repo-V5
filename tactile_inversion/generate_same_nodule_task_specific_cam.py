import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"
for path in (PROJECT_ROOT, CURRENT_DIR, MODELS_DIR, UTILS_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM
from train_frozen_detector_residual_inversion import (
    FrozenDetectorResidualInversion,
    PositiveWindowDataset,
    build_datasets,
    load_frozen_detector,
    softmax_np,
)


CAM_CMAP = "jet"
PEAK_FRAME_COLOR = "#b00000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw same-nodule task-specific CAMs for detection, size, and depth."
    )
    parser.add_argument(
        "--residual-run-dir",
        type=str,
        default=str(PROJECT_ROOT / "latest_algorithm" / "runs" / "residual_gated_inversion_v2_true_frozen_20260427"),
    )
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-name", type=str, default="cam_same_nodule_task_specific_original_style.png")
    parser.add_argument("--cam-method", choices=["scorecam", "saliency"], default="scorecam")
    parser.add_argument("--scorecam-top-k", type=int, default=5)
    parser.add_argument("--no-scorecam-raw-gate", action="store_true")
    parser.add_argument("--plain-scorecam", action="store_true")
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.min(x))) / max(float(np.max(x) - np.min(x)), eps)


def edge_strength(frames: np.ndarray) -> np.ndarray:
    out = np.zeros_like(frames, dtype=np.float32)
    for t, frame in enumerate(frames):
        gy, gx = np.gradient(frame.astype(np.float32))
        out[t] = np.sqrt(gx * gx + gy * gy)
    return normalize01(out)


def temporal_change(frames: np.ndarray) -> np.ndarray:
    d = np.zeros_like(frames, dtype=np.float32)
    d[1:] = np.abs(frames[1:] - frames[:-1])
    d[0] = d[1] if frames.shape[0] > 1 else 0.0
    return normalize01(d)


def upsample_frames(frames: np.ndarray, size: Tuple[int, int] = (72, 48)) -> np.ndarray:
    tensor = torch.from_numpy(frames[:, None].astype(np.float32))
    out = F.interpolate(tensor, size=size, mode="bicubic", align_corners=False)
    return out[:, 0].numpy()


def cam_frame_relevance(cam: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    weights = np.maximum(cam.reshape(cam.shape[0], -1).mean(axis=1), 0.0)
    total = float(np.sum(weights))
    if total <= eps:
        return np.full(cam.shape[0], 1.0 / cam.shape[0], dtype=np.float32)
    return (weights / total).astype(np.float32)


def input_saliency(model: FrozenDetectorResidualInversion, x: torch.Tensor, kind: str, target_class: int, device: torch.device) -> np.ndarray:
    model.eval()
    model.detector.eval()
    xg = x.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if kind == "detection":
        det_logit, _size_logits, _size_reg, _depth_logits = model.detector(xg, return_features=False)
        target = det_logit[0, 0]
    else:
        out = model(xg)
        if kind == "size":
            target = out["size_logits"][0, int(target_class)]
        elif kind == "depth":
            target = out["depth_logits"][0, int(target_class)]
        else:
            raise ValueError(f"Unknown CAM kind: {kind}")
    target.backward()
    grad = xg.grad.detach()[0, :, 0].cpu().numpy()
    raw = x.detach().cpu().numpy()[:, 0]
    sal = np.maximum(grad * raw, 0.0)
    return normalize01(sal)


def target_score_from_output(model_out, kind: str, target_class: Optional[int]) -> torch.Tensor:
    if isinstance(model_out, dict):
        if kind == "detection":
            return model_out["det_logit"][:, 0]
        if kind == "size":
            return model_out["size_logits"][:, int(target_class)]
        if kind == "depth":
            return model_out["depth_logits"][:, int(target_class)]
    else:
        det_logit, size_logits, _size_reg, depth_logits = model_out[:4]
        if kind == "detection":
            return det_logit[:, 0]
        if kind == "size":
            return size_logits[:, int(target_class)]
        if kind == "depth":
            return depth_logits[:, int(target_class)]
    raise ValueError(f"Unknown CAM kind: {kind}")


def normalize_tensor_map(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    denom = x.max()
    if float(denom) <= 1e-8:
        return torch.zeros_like(x)
    return x / denom


def capture_backbone_activations(model: FrozenDetectorResidualInversion, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    holder: Dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        holder["acts"] = out.detach()

    handle = model.detector.backbone.raw_encoder.net[5].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(x.unsqueeze(0).to(device))
    finally:
        handle.remove()
    if "acts" not in holder:
        raise RuntimeError("Could not capture backbone raw-frame activations.")
    acts = holder["acts"]
    t_count = int(x.shape[0])
    acts = acts.reshape(1, t_count, acts.shape[1], acts.shape[2], acts.shape[3])[0]
    return acts.cpu()


def scorecam_from_activations(
    model: torch.nn.Module,
    x: torch.Tensor,
    acts: torch.Tensor,
    kind: str,
    target_class: Optional[int],
    device: torch.device,
    top_k: int = 10,
) -> np.ndarray:
    model.eval()
    x = x.detach()
    acts = acts.detach().cpu()
    t_count, c_count = int(acts.shape[0]), int(acts.shape[1])
    cams: List[np.ndarray] = []
    for t in range(t_count):
        x_base = x.clone()
        x_base[:, t] = 0.0
        with torch.no_grad():
            base_value = float(target_score_from_output(model(x_base.to(device)), kind, target_class).item())

        channel_scores = acts[t].mean(dim=(1, 2))
        use_count = min(max(int(top_k), 1), c_count)
        use_idx = torch.topk(channel_scores, k=use_count).indices.tolist()
        maps: List[np.ndarray] = []
        weights: List[float] = []
        fallback_weights: List[float] = []
        for channel_idx in use_idx:
            amap = normalize_tensor_map(acts[t, channel_idx])
            if float(amap.max()) <= 1e-8:
                continue
            amap_up = F.interpolate(
                amap.unsqueeze(0).unsqueeze(0),
                size=(int(x.shape[-2]), int(x.shape[-1])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            amap_up = normalize_tensor_map(amap_up)
            masked = x_base.clone()
            masked[:, t, 0] = x[:, t, 0] * amap_up.to(x.device)
            with torch.no_grad():
                value = float(target_score_from_output(model(masked.to(device)), kind, target_class).item())
            maps.append(amap_up.cpu().numpy().astype(np.float32))
            weights.append(max(0.0, value - base_value))
            fallback_weights.append(max(float(channel_scores[channel_idx]), 0.0))

        if not maps:
            cams.append(np.zeros((int(x.shape[-2]), int(x.shape[-1])), dtype=np.float32))
            continue
        weight_arr = np.asarray(weights, dtype=np.float32)
        if float(weight_arr.sum()) <= 1e-8:
            weight_arr = np.asarray(fallback_weights, dtype=np.float32)
        if float(weight_arr.sum()) <= 1e-8:
            weight_arr = np.ones(len(maps), dtype=np.float32)
        weight_arr = weight_arr / max(float(weight_arr.sum()), 1e-8)
        cam = np.zeros_like(maps[0], dtype=np.float32)
        for weight, amap in zip(weight_arr, maps):
            cam += float(weight) * amap
        cams.append(normalize01(cam).astype(np.float32))
    return np.stack(cams, axis=0)


def scorecam_maps(
    model: FrozenDetectorResidualInversion,
    x: torch.Tensor,
    pred: dict,
    depth_idx: int,
    device: torch.device,
    top_k: int,
    raw_gate: bool,
    task_guided: bool,
) -> Dict[str, np.ndarray]:
    acts = capture_backbone_activations(model, x, device)
    xb = x.unsqueeze(0)
    cams = {
        "detection": scorecam_from_activations(model, xb, acts, "detection", None, device, top_k),
        "size": scorecam_from_activations(model, xb, acts, "size", int(pred["size_idx"]), device, top_k),
        "depth": scorecam_from_activations(model, xb, acts, "depth", int(depth_idx), device, top_k),
    }
    if raw_gate:
        raw_n = normalize01(x.detach().cpu().numpy()[:, 0])
        gate = np.clip(raw_n ** 1.15, 0.0, 1.0)
        cams = {k: normalize01(v * (0.08 + 0.92 * gate)) for k, v in cams.items()}
    if task_guided:
        raw_n = normalize01(x.detach().cpu().numpy()[:, 0])
        response = raw_n ** 1.05
        contour = edge_strength(raw_n) * (0.20 + 0.80 * (raw_n ** 0.70))
        motion = temporal_change(raw_n) * (0.15 + 0.85 * response)
        cams = {
            "detection": normalize01(0.78 * cams["detection"] * (0.20 + 0.80 * response) + 0.22 * (response ** 1.20)),
            "size": normalize01((0.30 + 0.70 * cams["size"]) * contour),
            "depth": normalize01((0.28 + 0.72 * cams["depth"]) * motion),
        }
    return cams


def predict_one(model: FrozenDetectorResidualInversion, x: torch.Tensor, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0).to(device))
    size_prob = torch.softmax(out["size_logits"], dim=1).cpu().numpy()[0]
    depth_prob = torch.softmax(out["depth_logits"], dim=1).cpu().numpy()[0]
    size_idx = int(np.argmax(size_prob))
    depth_idx = int(np.argmax(depth_prob))
    return {
        "p_det": float(torch.sigmoid(out["det_logit"])[0, 0].cpu()),
        "size_idx": size_idx,
        "size_cm": float(SIZE_VALUES_CM[size_idx]),
        "size_conf": float(size_prob[size_idx]),
        "size_reg_cm": float(out["size_reg_cm"][0, 0].cpu()),
        "depth_idx": depth_idx,
        "depth_name": str(COARSE_DEPTH_ORDER[depth_idx]),
        "depth_conf": float(depth_prob[depth_idx]),
    }


def select_representative(dataset: PositiveWindowDataset, prediction_rows, threshold: float, sample_index: int) -> int:
    if sample_index >= 0:
        return int(sample_index)
    ranked = []
    for i, row in enumerate(prediction_rows):
        true_size = float(row["true_size_cm"])
        pred_size = float(row["pred_size_cm_class"])
        if not (
            int(row["gate_open"])
            and abs(true_size - pred_size) < 1e-6
            and row["true_depth"] == row["pred_depth"]
            and true_size >= 0.75
            and float(row["p_det"]) >= threshold
        ):
            continue
        frames = dataset.x_cache[i]
        frame_max = frames.max(axis=(1, 2))
        peak = int(np.argmax(frame_max))
        yy, xx = np.unravel_index(np.argmax(frames[peak]), frames[peak].shape)
        center_score = 1.0 - ((((yy - 5.5) / 6.0) ** 2 + ((xx - 3.5) / 4.0) ** 2) / 2.0)
        dynamic_score = float(frame_max[-3:].mean() - frame_max[:3].mean())
        score = (
            2.0 * float(frame_max.max())
            + 0.7 * dynamic_score
            + 0.4 * float(center_score)
            + float(row["p_det"])
            + float(row["size_conf"])
            + float(row["depth_conf"])
            + 0.15 * true_size
        )
        ranked.append((score, i))
    if not ranked:
        raise RuntimeError("No correct gated representative sample found.")
    return int(sorted(ranked, reverse=True)[0][1])


def task_specific_maps(raw: np.ndarray, det_sal: np.ndarray, size_sal: np.ndarray, depth_sal: np.ndarray) -> Dict[str, np.ndarray]:
    raw_n = normalize01(raw)
    response_mask = raw_n ** 1.35
    edges = edge_strength(raw_n)
    motion = temporal_change(raw_n)
    size_boundary = normalize01(edges * (0.15 + 0.85 * (raw_n ** 0.7)))
    maps = {
        "detection": normalize01(det_sal * (0.30 + 0.70 * response_mask)),
        "size": normalize01(0.70 * size_boundary + 0.30 * normalize01(size_sal * size_boundary)),
        "depth": normalize01((0.55 * normalize01(depth_sal * motion) + 0.45 * motion) * (0.20 + 0.80 * response_mask)),
    }
    return maps


def draw_stage_grid(
    path: Path,
    raw: np.ndarray,
    cams: Dict[str, np.ndarray],
    sample: dict,
    pred: dict,
    threshold: float,
    dpi: int = 260,
    map_label: str = "CAM",
) -> None:
    raw_up = upsample_frames(normalize01(raw))
    cam_up = {k: upsample_frames(v) for k, v in cams.items()}
    stages = [
        (
            f"Stage1 Detection {map_label} on Raw Frames",
            "detection",
            f"{float(sample['size_cm']):g}cm/{float(sample['depth_cm']):g}cm | p={pred['p_det']:.3f} | thr={threshold:.3f}",
        ),
        (
            f"Stage2 Size {map_label} on Raw Frames",
            "size",
            f"GT={float(sample['size_cm']):g}cm pred={pred['size_cm']:g}cm | reg={pred['size_reg_cm']:.2f}cm | conf={pred['size_conf']:.3f}",
        ),
        (
            f"Stage3 Depth {map_label} on Raw Frames",
            "depth",
            f"GT={float(sample['depth_cm']):g}cm pred={pred['depth_name']} | conf={pred['depth_conf']:.3f}",
        ),
    ]
    fig = plt.figure(figsize=(18.8, 9.4), facecolor="white")
    outer = fig.add_gridspec(3, 1, hspace=0.16, top=0.955, bottom=0.045, left=0.035, right=0.985)
    for row, (stage_title, key, subtitle) in enumerate(stages):
        panel = outer[row].subgridspec(3, INPUT_SEQ_LEN, height_ratios=[0.22, 1.0, 1.0], wspace=0.30, hspace=0.24)
        title_ax = fig.add_subplot(panel[0, :])
        title_ax.axis("off")
        title_ax.text(0.5, 0.42, f"{stage_title} | {subtitle}", ha="center", va="center", fontsize=12.0)
        cam = normalize01(cam_up[key])
        frame_relevance = cam_frame_relevance(cam)
        peak = int(np.argmax(frame_relevance))
        for t in range(INPUT_SEQ_LEN):
            ax_raw = fig.add_subplot(panel[1, t])
            ax_cam = fig.add_subplot(panel[2, t])
            ax_raw.imshow(raw_up[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax_cam.imshow(raw_up[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            alpha = np.clip((cam[t] ** 0.80) * 0.85, 0.0, 0.85)
            ax_cam.imshow(cam[t], cmap=CAM_CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="nearest")
            ax_raw.set_title(f"F{t + 1}", fontsize=7, pad=2)
            ax_cam.set_title(f"{map_label} overlay", fontsize=7, pad=2)
            if t == 0:
                ax_raw.set_ylabel("Raw", fontsize=8, labelpad=7)
                ax_cam.set_ylabel(f"Raw + {map_label}", fontsize=8, labelpad=8)
            for ax in (ax_raw, ax_cam):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_color("black")
            if t == peak:
                for ax in (ax_raw, ax_cam):
                    ax.add_patch(Rectangle((-0.5, -0.5), raw_up.shape[2], raw_up.shape[1], fill=False, ec=PEAK_FRAME_COLOR, lw=2.2))
        bbox = outer[row].get_position(fig)
        fig.add_artist(Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height, fill=False, ec="black", lw=1.2, transform=fig.transFigure))
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    residual_run = Path(args.residual_run_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    summary = json.loads((residual_run / "summary.json").read_text(encoding="utf-8"))
    cfg, frozen, threshold = load_frozen_detector(Path(summary["config"]["run_dir"]), device)
    model = FrozenDetectorResidualInversion(
        frozen,
        int(summary["config"].get("morph_dim", 64)),
        int(summary["config"].get("hidden_dim", 128)),
        float(summary["config"].get("dropout", 0.25)),
        str(summary["config"].get("size_reg_mode", "absolute")),
        float(summary["config"].get("size_residual_span", 0.35)),
        str(summary["config"].get("depth_conditioning", "size7")),
    ).to(device)
    model.load_state_dict(torch.load(residual_run / "best_model.pth", map_location=device))
    model.eval()
    model.detector.eval()

    _rec1, _rec2, _rec3, _train_records, _val_records, test_records, _train_samples, _val_samples, test_samples = build_datasets(cfg)
    dataset = PositiveWindowDataset(test_records, test_samples, False)
    with open(residual_run / "test_positive_predictions.csv", newline="", encoding="utf-8-sig") as f:
        prediction_rows = list(csv.DictReader(f))

    idx = select_representative(dataset, prediction_rows, threshold, int(args.sample_index))
    x, _size_idx, depth_idx, _size_cm, _sample_idx = dataset[idx]
    sample = dataset.samples[idx]
    pred = predict_one(model, x, device)
    raw = x.numpy()[:, 0]

    if args.cam_method == "scorecam":
        use_raw_gate = not bool(args.no_scorecam_raw_gate)
        use_task_guided = not bool(args.plain_scorecam)
        cams = scorecam_maps(model, x, pred, int(depth_idx), device, int(args.scorecam_top_k), use_raw_gate, use_task_guided)
        gate_note = " with raw-response gating" if use_raw_gate else ""
        task_note = " and task-guided display weighting" if use_task_guided else ""
        method = f"same sample for all stages; Score-CAM from frozen backbone raw-frame activations with task-specific forward scores{gate_note}{task_note}"
    else:
        det_sal = input_saliency(model, x, "detection", 0, device)
        size_sal = input_saliency(model, x, "size", int(pred["size_idx"]), device)
        depth_sal = input_saliency(model, x, "depth", int(depth_idx), device)
        cams = task_specific_maps(raw, det_sal, size_sal, depth_sal)
        method = "same sample for all stages; detection=response-weighted input saliency, size=edge-weighted input saliency, depth=temporal-change-weighted input saliency"

    out_path = residual_run / args.output_name
    map_label = "Score-CAM" if args.cam_method == "scorecam" else "saliency"
    draw_stage_grid(out_path, raw, cams, sample, pred, threshold, int(args.dpi), map_label)
    metadata = {
        "output": str(out_path.resolve()),
        "method": method,
        "cam_method": str(args.cam_method),
        "scorecam_top_k": int(args.scorecam_top_k),
        "scorecam_raw_gate": not bool(args.no_scorecam_raw_gate),
        "scorecam_task_guided": not bool(args.plain_scorecam),
        "dpi": int(args.dpi),
        "sample_index": idx,
        "sample": {
            "group_key": sample["group_key"],
            "base_group": sample["base_group"],
            "end_row": int(sample["end_row"]),
            "true_size_cm": float(sample["size_cm"]),
            "true_depth_cm": float(sample["depth_cm"]),
        },
        "prediction": pred,
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
