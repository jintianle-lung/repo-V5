"""Retrain the R5 residual inversion branch with release-relative paths."""

from __future__ import annotations

import argparse
import sys

from .paths import FROZEN_DETECTOR_RUN, chdir_release_root


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", default="r5", choices=["r5"])
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    chdir_release_root()
    from . import train_frozen_detector_residual_inversion as residual_script

    defaults = [
        "--run-dir",
        str(FROZEN_DETECTOR_RUN),
        "--output-dir",
        "results/retrained_r5",
        "--lr",
        "7e-4",
        "--weight-decay",
        "8e-4",
        "--dropout",
        "0.28",
        "--morph-dim",
        "64",
        "--hidden-dim",
        "160",
        "--size-reg-mode",
        "expected_residual",
        "--depth-conditioning",
        "size7_coarse",
        "--gate-loss-alpha",
        "0.15",
        "--selection-mode",
        "composite",
        "--size-cls-weight",
        "0.9",
        "--size-coarse-weight",
        "0.45",
        "--size-reg-weight",
        "0.65",
        "--depth-cls-weight",
        "1.05",
        "--depth-binary-weight",
        "0.50",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = ["train_residual", *defaults, *parsed.extra]
        residual_script.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

