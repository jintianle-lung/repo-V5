"""Retrain the frozen evidence detector with release-relative paths."""

from __future__ import annotations

import argparse
import sys

from .paths import DATA_ROOT, LABEL_DIR, chdir_release_root


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    chdir_release_root()
    from . import train_shared_cnn_mstcn_cascade_file3 as detector_script

    defaults = [
        "--file1-labels",
        str(LABEL_DIR / "manual_keyframe_labels_file1.json"),
        "--file2-labels",
        str(LABEL_DIR / "manual_keyframe_labels_file2.json"),
        "--file3-labels",
        str(LABEL_DIR / "manual_keyframe_labels_file3.json"),
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        "results/retrained_detector",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = ["train_detector", *defaults, *parsed.extra]
        detector_script.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

