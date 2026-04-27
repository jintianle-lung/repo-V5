"""Top-level entry point for the R5 reproducible release."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from tactile_inversion.paths import chdir_release_root


TORCH_ENV = Path(r"C:\Users\SWH\.conda\envs\ai\python.exe")


def resolve_python_executable() -> str:
    if importlib.util.find_spec("torch") is not None:
        return sys.executable

    override = os.environ.get("TACTILE_RELEASE_PYTHON")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return str(override_path)

    if TORCH_ENV.exists():
        return str(TORCH_ENV)

    return sys.executable


def run_module(module: str, *args: str) -> None:
    python = resolve_python_executable()
    cmd = [python, "-m", module, *args]
    print(f"[main] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sample-index", type=int, default=1155)
    parser.add_argument("--no-scorecam", action="store_true")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    chdir_release_root()

    if not args.evaluate_only:
        run_module(
            "tactile_inversion.demo",
            "--device",
            args.device,
            "--sample-index",
            str(args.sample_index),
            *([] if not args.no_scorecam else ["--no-scorecam"]),
        )

    if not args.demo_only:
        run_module("tactile_inversion.evaluate", "--device", args.device)

    if not args.no_scorecam and not args.evaluate_only:
        run_module(
            "tactile_inversion.make_scorecam",
            "--device",
            args.device,
            "--sample-index",
            str(args.sample_index),
            "--output-name",
            "cam_scorecam_main.png",
        )


if __name__ == "__main__":
    main()
