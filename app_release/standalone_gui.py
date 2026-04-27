from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
from matplotlib import patches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tactile_inversion.evaluate import load_release_model
from tactile_inversion.input_normalization_v1 import normalize_raw_frames_window_minmax
from tactile_inversion.paths import R5_RUN, RELEASE_ROOT
from tactile_inversion.task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM


DEFAULT_SAMPLE_PATH = RELEASE_ROOT / "data" / "raw" / "0.25cm大" / "0.5cm深" / "1.CSV"


def _find_csv_columns(df: pd.DataFrame) -> np.ndarray:
    mat_cols = [c for c in df.columns if str(c).strip().startswith("MAT_")]
    if mat_cols:
        try:
            mat_cols.sort(key=lambda x: int(str(x).strip().split("_")[1]))
        except Exception:
            pass
        data = df[mat_cols].values
    else:
        numeric = df.select_dtypes(include=[np.number]).values
        data = numeric if numeric.size else df.values
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D CSV table, got shape {data.shape}")
    if data.shape[1] != 96:
        if data.shape[1] > 96:
            data = data[:, -96:]
        else:
            raise ValueError(f"CSV must have 96 sensor columns, got {data.shape[1]}")
    return data


def _frame_to_matrix(frame: np.ndarray) -> np.ndarray:
    return np.asarray(frame, dtype=np.float32).reshape(12, 8)


def _to_hex(rgb: tuple[float, float, float, float]) -> str:
    return mcolors.to_hex(rgb[:3])


@dataclass
class PredictionBundle:
    det_prob: float
    size_probs: np.ndarray
    size_reg_cm: float
    depth_probs: np.ndarray
    size_idx: int
    depth_idx: int


class StandaloneApp:
    def __init__(self, root: tk.Tk, auto_load: bool = True):
        self.root = root
        self.root.title("High-Performance Pulmonary Nodule Detection System - Optimized")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)

        self.device = torch.device("cpu")
        self.model, self.cfg, self.threshold, self.summary = load_release_model(R5_RUN, self.device)
        self.model.eval()

        self.data: np.ndarray | None = None
        self.current_frame = 0
        self.frame_count = 0
        self.pred_cache: list[PredictionBundle] = []

        self.status_var = tk.StringVar(value="Ready")
        self.summary_var = tk.StringVar(value="No CSV loaded")
        self.prediction_var = tk.StringVar(value="Load a CSV file to start")
        self.frame_var = tk.StringVar(value="0/0")
        self.threshold_var = tk.StringVar(value=f"Threshold: {self.threshold:.3f}")

        self._build_ui()
        self._load_default_if_available(auto_load)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        control = ttk.Frame(outer)
        control.pack(fill="x")

        ttk.Button(control, text="Load CSV", command=self._browse_csv).pack(side="left", padx=(0, 8))
        ttk.Label(control, text="Frame").pack(side="left", padx=(0, 4))
        ttk.Button(control, text="<<", width=4, command=lambda: self.step_frame(-1)).pack(side="left")
        ttk.Button(control, text="<", width=4, command=lambda: self.step_frame(-1)).pack(side="left", padx=(4, 10))
        self.frame_scale = tk.Scale(
            control,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            length=620,
            command=self._on_scale,
        )
        self.frame_scale.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(control, text=">", width=4, command=lambda: self.step_frame(1)).pack(side="left")
        ttk.Button(control, text=">>", width=4, command=lambda: self.step_frame(1)).pack(side="left", padx=(4, 10))
        ttk.Label(control, text="Jump:").pack(side="left", padx=(0, 4))
        self.jump_entry = ttk.Entry(control, width=8)
        self.jump_entry.pack(side="left")
        ttk.Button(control, text="Go", command=self._jump_to_frame).pack(side="left", padx=(4, 10))
        ttk.Label(control, textvariable=self.threshold_var).pack(side="left", padx=(4, 10))
        ttk.Label(control, textvariable=self.frame_var).pack(side="left")

        info = ttk.Frame(outer)
        info.pack(fill="x", pady=(8, 6))
        ttk.Label(info, textvariable=self.summary_var).pack(side="left")
        ttk.Label(info, textvariable=self.prediction_var).pack(side="left", padx=(16, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        grid_box = ttk.LabelFrame(left, text="Sensor Grid Monitor", padding=8)
        grid_box.pack(fill="x")
        self.sensor_labels: list[list[tk.Label]] = []
        grid = tk.Frame(grid_box)
        grid.pack()
        for r in range(12):
            row: list[tk.Label] = []
            for c in range(8):
                lbl = tk.Label(
                    grid,
                    text="0.00",
                    width=6,
                    height=2,
                    relief="solid",
                    borderwidth=1,
                    font=("Consolas", 8),
                    bg="#ffffff",
                )
                lbl.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
                row.append(lbl)
            self.sensor_labels.append(row)

        stats = ttk.LabelFrame(left, text="Value Range", padding=8)
        stats.pack(fill="x", pady=(10, 0))
        self.range_var = tk.StringVar(value="Min: 0.000\nMax: 0.000\nMean: 0.000")
        ttk.Label(stats, textvariable=self.range_var, justify="left").pack(anchor="w")
        self.loaded_var = tk.StringVar(value="Loaded 0 frames")
        ttk.Label(stats, textvariable=self.loaded_var).pack(anchor="w", pady=(8, 0))

        plot_box = ttk.LabelFrame(right, text="Real-Time Visualization (Zoomable)", padding=8)
        plot_box.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(12, 7.2), dpi=110, facecolor="white")
        self.axes = np.array(self.figure.subplots(2, 3), dtype=object)
        self.figure.subplots_adjust(wspace=0.28, hspace=0.35, left=0.05, right=0.98, top=0.94, bottom=0.06)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.fig_labels = {
            "raw": "1. Raw Sensor Map (12x8)",
            "feature": "2. Feature Map (Interpolated)",
            "overlay": "3. AI Detection Overlay",
            "prob": "4. AI Nodule Probability",
            "size": "5. Size Prediction Distribution",
            "depth": "6. Depth Prediction Distribution",
        }

    def _load_default_if_available(self, auto_load: bool) -> None:
        if auto_load and DEFAULT_SAMPLE_PATH.exists():
            self.load_csv(DEFAULT_SAMPLE_PATH, focus_frame=633)

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.load_csv(path)

    def load_csv(self, path: str | Path, focus_frame: int = 0) -> None:
        csv_path = Path(path)
        try:
            df = pd.read_csv(csv_path)
            self.data = _find_csv_columns(df)
            self.frame_count = int(self.data.shape[0])
            self.current_frame = int(np.clip(focus_frame, 0, max(self.frame_count - 1, 0)))
            self.frame_scale.config(to=max(self.frame_count - 1, 0))
            self.frame_scale.set(self.current_frame)
            self._precompute_predictions()
            self.summary_var.set(f"Loaded {csv_path.name} | {self.frame_count} frames | R5 release")
            self.loaded_var.set(f"Loaded {self.frame_count} frames")
            self.status_var.set(f"Loaded {csv_path}")
            self.update_display()
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _precompute_predictions(self) -> None:
        if self.data is None:
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        preds: list[PredictionBundle] = []
        for frame_idx in range(self.frame_count):
            preds.append(self._predict_frame(frame_idx))
        self.pred_cache = preds
        self.root.config(cursor="")

    def _window_for_frame(self, frame_idx: int) -> np.ndarray:
        assert self.data is not None
        seq_len = int(INPUT_SEQ_LEN)
        start = max(0, frame_idx - seq_len + 1)
        window = self.data[start : frame_idx + 1]
        if window.shape[0] < seq_len:
            pad = np.repeat(window[:1], seq_len - window.shape[0], axis=0)
            window = np.vstack([pad, window])
        window = normalize_raw_frames_window_minmax(window)
        return window.astype(np.float32, copy=False)

    def _predict_frame(self, frame_idx: int) -> PredictionBundle:
        window = self._window_for_frame(frame_idx)
        x = torch.from_numpy(window[:, None, :, :]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        det_prob = float(torch.sigmoid(out["det_logit"])[0, 0].item())
        size_probs = torch.softmax(out["size_logits"], dim=1)[0].cpu().numpy().astype(np.float32)
        depth_probs = torch.softmax(out["depth_logits"], dim=1)[0].cpu().numpy().astype(np.float32)
        size_reg_cm = float(out["size_reg_cm"][0, 0].item())
        size_idx = int(np.argmax(size_probs))
        depth_idx = int(np.argmax(depth_probs))
        return PredictionBundle(det_prob, size_probs, size_reg_cm, depth_probs, size_idx, depth_idx)

    def _on_scale(self, value: str) -> None:
        self.current_frame = int(float(value))
        self.update_display()

    def step_frame(self, delta: int) -> None:
        if self.frame_count <= 0:
            return
        self.current_frame = int(np.clip(self.current_frame + delta, 0, self.frame_count - 1))
        self.frame_scale.set(self.current_frame)
        self.update_display()

    def _jump_to_frame(self) -> None:
        try:
            target = int(self.jump_entry.get().strip())
        except Exception:
            messagebox.showwarning("Invalid value", "Please enter a frame number.")
            return
        if self.frame_count <= 0:
            return
        self.current_frame = int(np.clip(target, 0, self.frame_count - 1))
        self.frame_scale.set(self.current_frame)
        self.update_display()

    def _prediction_for_current(self) -> PredictionBundle:
        if not self.pred_cache:
            return PredictionBundle(0.0, np.zeros(len(SIZE_VALUES_CM), dtype=np.float32), 0.0, np.zeros(len(COARSE_DEPTH_ORDER), dtype=np.float32), 0, 0)
        idx = int(np.clip(self.current_frame, 0, len(self.pred_cache) - 1))
        return self.pred_cache[idx]

    def update_display(self) -> None:
        if self.data is None or self.frame_count <= 0:
            return
        self.current_frame = int(np.clip(self.current_frame, 0, self.frame_count - 1))
        self.frame_var.set(f"{self.current_frame + 1}/{self.frame_count}")

        frame = self.data[self.current_frame]
        matrix = _frame_to_matrix(frame)
        pred = self._prediction_for_current()
        det_on = pred.det_prob >= self.threshold
        size_name = f"{SIZE_VALUES_CM[pred.size_idx]:g} cm"
        depth_name = COARSE_DEPTH_ORDER[pred.depth_idx]
        self.prediction_var.set(
            f"Nodule Probability: {pred.det_prob * 100:.2f}% | Estimated Size: {size_name} | Estimated Depth: {depth_name} | Confidence {max(pred.size_probs[pred.size_idx], pred.depth_probs[pred.depth_idx]) * 100:.1f}%"
        )
        self.range_var.set(
            f"Min: {float(matrix.min()):.3f}\nMax: {float(matrix.max()):.3f}\nMean: {float(matrix.mean()):.3f}"
        )
        self._update_sensor_grid(matrix)
        self._update_plots(matrix, pred, det_on)

    def _update_sensor_grid(self, matrix: np.ndarray) -> None:
        mn = float(matrix.min())
        mx = float(matrix.max())
        denom = max(mx - mn, 1e-6)
        plt_cmap = matplotlib.colormaps.get_cmap("turbo")
        for r in range(12):
            for c in range(8):
                value = float(matrix[r, c])
                norm = (value - mn) / denom
                rgba = plt_cmap(norm)
                self.sensor_labels[r][c].config(
                    text=f"{value:.2f}",
                    bg=_to_hex(rgba),
                    fg="#000000" if norm > 0.55 else "#ffffff",
                )

    def _update_plots(self, matrix: np.ndarray, pred: PredictionBundle, det_on: bool) -> None:
        axes = self.axes.flatten()
        for ax in axes:
            ax.clear()

        raw = axes[0]
        raw.imshow(matrix, cmap="turbo", aspect="auto")
        raw.set_title(self.fig_labels["raw"], fontsize=9)
        raw.set_xticks([])
        raw.set_yticks([])

        feat = axes[1]
        tensor = torch.from_numpy(matrix[None, None].astype(np.float32))
        interp = F.interpolate(tensor, size=(48, 36), mode="bicubic", align_corners=False)[0, 0].numpy()
        feat.imshow(interp, cmap="turbo", aspect="auto")
        feat.set_title(self.fig_labels["feature"], fontsize=9)
        feat.set_xticks([])
        feat.set_yticks([])

        overlay = axes[2]
        overlay.imshow(matrix, cmap="turbo", aspect="auto")
        overlay.set_title(
            f"{self.fig_labels['overlay']} [{'DETECTED' if det_on else 'CALM'} P={pred.det_prob:.2f}]",
            fontsize=9,
            color="#d00000" if det_on else "#333333",
        )
        overlay.set_xticks([])
        overlay.set_yticks([])
        y, x = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        h = 1.8 + float(pred.size_reg_cm) * 1.2
        w = 1.6 + float(pred.size_reg_cm) * 1.0
        overlay.add_patch(
            patches.Ellipse(
                (x, y),
                width=w,
                height=h,
                fill=False,
                edgecolor="#ffee33" if det_on else "#ffffff",
                linewidth=2.0,
            )
        )
        if det_on:
            overlay.add_patch(
                patches.Rectangle(
                    (-0.5, -0.5),
                    8.0,
                    12.0,
                    fill=False,
                    edgecolor="#ff0000",
                    linewidth=2.0,
                )
            )

        prob_ax = axes[3]
        probs = np.array([p.det_prob for p in self.pred_cache], dtype=np.float32) if self.pred_cache else np.array([], dtype=np.float32)
        if probs.size:
            prob_ax.plot(probs, color="#1b5cff", linewidth=1.4)
            prob_ax.axhline(self.threshold, color="#ff8844", linestyle="--", linewidth=1.1, label=f"Threshold ({self.threshold:.3f})")
            prob_ax.axvline(self.current_frame, color="#32aa55", linestyle=":", linewidth=1.1, label="Current frame")
            prob_ax.scatter([self.current_frame], [pred.det_prob], color="#cc0000", s=16, zorder=5)
        prob_ax.set_ylim(0, 1.05)
        prob_ax.set_title(self.fig_labels["prob"], fontsize=9)
        prob_ax.set_xlabel("Frame", fontsize=8)
        prob_ax.set_ylabel("Probability", fontsize=8)
        prob_ax.legend(fontsize=7, loc="upper right")
        prob_ax.grid(alpha=0.25)

        size_ax = axes[4]
        size_ax.bar(range(len(SIZE_VALUES_CM)), pred.size_probs, color=["#cfe1ff" if i != pred.size_idx else "#2d7ff9" for i in range(len(SIZE_VALUES_CM))])
        size_ax.set_xticks(range(len(SIZE_VALUES_CM)))
        size_ax.set_xticklabels([f"{v:g}" for v in SIZE_VALUES_CM], fontsize=8)
        size_ax.set_ylim(0, 1.05)
        size_ax.set_title(self.fig_labels["size"], fontsize=9)
        size_ax.text(0.02, 0.95, f"Top1 Confidence: {pred.size_probs[pred.size_idx] * 100:.1f}%\nEst: {pred.size_reg_cm:.2f} cm", transform=size_ax.transAxes, va="top", fontsize=7)
        size_ax.grid(axis="y", alpha=0.18)

        depth_ax = axes[5]
        depth_ax.bar(range(len(COARSE_DEPTH_ORDER)), pred.depth_probs, color=["#d7f0db" if i != pred.depth_idx else "#2f9550" for i in range(len(COARSE_DEPTH_ORDER))])
        depth_ax.set_xticks(range(len(COARSE_DEPTH_ORDER)))
        depth_ax.set_xticklabels([name.title() for name in COARSE_DEPTH_ORDER], fontsize=8)
        depth_ax.set_ylim(0, 1.05)
        depth_ax.set_title(self.fig_labels["depth"], fontsize=9)
        depth_ax.text(0.02, 0.95, f"Top1 Confidence: {pred.depth_probs[pred.depth_idx] * 100:.1f}%\n{COARSE_DEPTH_ORDER[pred.depth_idx]}", transform=depth_ax.transAxes, va="top", fontsize=7)
        depth_ax.grid(axis="y", alpha=0.18)

        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    app = StandaloneApp(root, auto_load=True)
    if app.data is None and DEFAULT_SAMPLE_PATH.exists():
        app.load_csv(DEFAULT_SAMPLE_PATH, focus_frame=633)
    root.mainloop()


if __name__ == "__main__":
    main()
