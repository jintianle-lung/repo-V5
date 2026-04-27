"""Shared paths for the reproducible R5 release."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = PACKAGE_DIR.parent
DATA_ROOT = RELEASE_ROOT / "data" / "raw"
LABEL_DIR = RELEASE_ROOT / "data" / "labels"
FROZEN_DETECTOR_RUN = RELEASE_ROOT / "results" / "frozen_detector"
R5_RUN = RELEASE_ROOT / "results" / "r5"


def chdir_release_root() -> None:
    """Make legacy script defaults and relative run paths resolve consistently."""

    os.chdir(RELEASE_ROOT)


def resolve_release_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return RELEASE_ROOT / path

