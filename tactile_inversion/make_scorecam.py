"""Generate the released task-guided Score-CAM visualization."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from .paths import R5_RUN, chdir_release_root


warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")


def run_scorecam(
    device: str = "cpu",
    residual_run_dir: str | Path = R5_RUN,
    output_name: str = "cam_scorecam_demo.png",
    sample_index: int = 1155,
    top_k: int = 3,
    dpi: int = 260,
) -> None:
    chdir_release_root()
    from . import generate_same_nodule_task_specific_cam as scorecam_script

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "make_scorecam",
            "--residual-run-dir",
            str(residual_run_dir),
            "--sample-index",
            str(sample_index),
            "--device",
            device,
            "--scorecam-top-k",
            str(top_k),
            "--dpi",
            str(dpi),
            "--output-name",
            output_name,
        ]
        scorecam_script.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--residual-run-dir", default=str(R5_RUN))
    parser.add_argument("--output-name", default="cam_scorecam_demo.png")
    parser.add_argument("--sample-index", type=int, default=1155)
    parser.add_argument("--scorecam-top-k", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()
    run_scorecam(
        device=args.device,
        residual_run_dir=args.residual_run_dir,
        output_name=args.output_name,
        sample_index=args.sample_index,
        top_k=args.scorecam_top_k,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
