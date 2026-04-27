"""Canonical first-version task protocol for the lung nodule paper project.

This module is the single code-side source of truth for:
1. input tensor definition
2. official output targets
3. size/depth class axes
4. detection-gated runtime display policy

Important note:
- The locked first-version project contract uses the real dataset axes:
  7 size levels x 6 depth levels = 42 conditions.
- Earlier planning notes sometimes described the matrix as 6 x 7. The data
  directories under ``整理好的数据集/建表数据`` confirm the actual axis order is
  7 sizes and 6 depths.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


INPUT_SEQ_LEN = 10
INPUT_CHANNELS = 1
INPUT_HEIGHT = 12
INPUT_WIDTH = 8
WINDOW_STRIDE = 2
INPUT_SHAPE = (INPUT_SEQ_LEN, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

SIZE_VALUES_CM: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
DEPTH_VALUES_CM: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

COARSE_DEPTH_TO_VALUES: Dict[str, Tuple[float, ...]] = {
    "shallow": (0.5, 1.0),
    "middle": (1.5, 2.0),
    "deep": (2.5, 3.0),
}
COARSE_DEPTH_ORDER: Tuple[str, ...] = ("shallow", "middle", "deep")

OUTPUT_DETECTION = "y_det_prob"
OUTPUT_SIZE_CLASS = "y_size_cls"
OUTPUT_SIZE_REGRESSION = "y_size_reg"
OUTPUT_DEPTH_COARSE = "y_depth_cls_coarse"

PRIMARY_TASK = "detection"
MAIN_INVERSION_TASK = "size_inversion"
SECONDARY_TASK = "coarse_depth_training_and_interpretation"

RUNTIME_ALWAYS_VISIBLE_FIELDS: Tuple[str, ...] = (OUTPUT_DETECTION,)
RUNTIME_GATED_FIELDS: Tuple[str, ...] = (
    OUTPUT_SIZE_CLASS,
    OUTPUT_SIZE_REGRESSION,
    OUTPUT_DEPTH_COARSE,
)


@dataclass(frozen=True)
class TaskProtocolV1:
    """Frozen protocol contract for first-version training and deployment."""

    input_shape: Tuple[int, int, int, int] = INPUT_SHAPE
    window_stride: int = WINDOW_STRIDE
    size_values_cm: Tuple[float, ...] = SIZE_VALUES_CM
    depth_values_cm: Tuple[float, ...] = DEPTH_VALUES_CM
    coarse_depth_order: Tuple[str, ...] = COARSE_DEPTH_ORDER
    always_visible_fields: Tuple[str, ...] = RUNTIME_ALWAYS_VISIBLE_FIELDS
    gated_fields: Tuple[str, ...] = RUNTIME_GATED_FIELDS


PROTOCOL_V1 = TaskProtocolV1()


def protocol_summary() -> Dict[str, object]:
    """Return a compact serializable summary of the locked protocol."""

    return {
        "input_shape": list(INPUT_SHAPE),
        "window_stride": WINDOW_STRIDE,
        "size_values_cm": list(SIZE_VALUES_CM),
        "depth_values_cm": list(DEPTH_VALUES_CM),
        "coarse_depth_to_values": {
            key: list(values) for key, values in COARSE_DEPTH_TO_VALUES.items()
        },
        "outputs": [
            OUTPUT_DETECTION,
            OUTPUT_SIZE_CLASS,
            OUTPUT_SIZE_REGRESSION,
            OUTPUT_DEPTH_COARSE,
        ],
        "task_priority": [
            PRIMARY_TASK,
            MAIN_INVERSION_TASK,
            SECONDARY_TASK,
        ],
        "runtime_policy": {
            "always_visible": list(RUNTIME_ALWAYS_VISIBLE_FIELDS),
            "gated": list(RUNTIME_GATED_FIELDS),
        },
    }


def should_display_inversion_outputs(det_prob: float, threshold: float) -> bool:
    """Return True when size/depth outputs may be shown in the GUI."""

    return float(det_prob) >= float(threshold)


def format_runtime_payload(
    det_prob: float,
    threshold: float,
    size_class: Optional[str] = None,
    size_reg_cm: Optional[float] = None,
    depth_coarse: Optional[str] = None,
) -> Dict[str, object]:
    """Build a GUI-friendly runtime payload following the gated policy."""

    gated = should_display_inversion_outputs(det_prob, threshold)
    return {
        "p_det": float(det_prob),
        "gate_open": gated,
        "size_class": size_class if gated else None,
        "size_reg_cm": float(size_reg_cm) if gated and size_reg_cm is not None else None,
        "depth_coarse": depth_coarse if gated else None,
    }


def parse_cm_text(value: str) -> float:
    """Parse a size/depth string such as '1.25cm大' or '2.0cm深'."""

    match = re.search(r"(\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Could not parse numeric cm value from: {value!r}")
    return float(match.group(1))


def parse_group_key(group_key: str) -> Tuple[float, float]:
    """Parse '<size>|<depth>' into numeric cm values."""

    parts = group_key.split("|")
    if len(parts) != 2:
        raise ValueError(f"Group key must be '<size>|<depth>', got: {group_key!r}")
    size_cm = parse_cm_text(parts[0])
    depth_cm = parse_cm_text(parts[1])
    return size_cm, depth_cm


def size_to_class_index(size_cm: float) -> int:
    """Map a numeric size value in cm to the formal class index."""

    return SIZE_VALUES_CM.index(float(size_cm))


def class_index_to_size(index: int) -> float:
    """Map a formal size class index back to the size value in cm."""

    return SIZE_VALUES_CM[index]


def size_to_class_name(size_cm: float) -> str:
    """Return a human-readable size class name."""

    return f"{float(size_cm):g}cm"


def depth_to_coarse_name(depth_cm: float) -> str:
    """Map a fine depth value to the locked first-version coarse depth label."""

    depth_cm = float(depth_cm)
    for coarse_name, values in COARSE_DEPTH_TO_VALUES.items():
        if depth_cm in values:
            return coarse_name
    raise ValueError(f"Depth {depth_cm} cm is outside the formal coarse mapping.")


def depth_to_coarse_index(depth_cm: float) -> int:
    """Map a fine depth value to a coarse depth class index."""

    return COARSE_DEPTH_ORDER.index(depth_to_coarse_name(depth_cm))


def coarse_index_to_name(index: int) -> str:
    """Map a coarse depth class index to its label."""

    return COARSE_DEPTH_ORDER[index]


def coarse_name_to_values(name: str) -> Tuple[float, ...]:
    """Return the fine depth values covered by a coarse depth label."""

    if name not in COARSE_DEPTH_TO_VALUES:
        raise KeyError(f"Unknown coarse depth label: {name!r}")
    return COARSE_DEPTH_TO_VALUES[name]


def infer_size_depth_from_record_parts(
    size_text: str,
    depth_text: str,
) -> Dict[str, object]:
    """Convert raw record names into the canonical protocol labels."""

    size_cm = parse_cm_text(size_text)
    depth_cm = parse_cm_text(depth_text)
    return {
        "size_cm": size_cm,
        "size_class_index": size_to_class_index(size_cm),
        "size_class_name": size_to_class_name(size_cm),
        "depth_cm": depth_cm,
        "depth_coarse_index": depth_to_coarse_index(depth_cm),
        "depth_coarse_name": depth_to_coarse_name(depth_cm),
    }


def scan_dataset_axes(dataset_root: Union[str, Path]) -> Dict[str, List[float]]:
    """Inspect the dataset directory and return observed size/depth axes."""

    dataset_root = Path(dataset_root)
    size_dirs = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.endswith("cm大")]
    if not size_dirs:
        raise FileNotFoundError(f"No size directories found under: {dataset_root}")

    observed_sizes = sorted(parse_cm_text(path.name) for path in size_dirs)
    depth_values = set()
    for size_dir in size_dirs:
        depth_dirs = [
            path for path in size_dir.iterdir() if path.is_dir() and path.name.endswith("cm深")
        ]
        depth_values.update(parse_cm_text(path.name) for path in depth_dirs)

    return {
        "size_values_cm": observed_sizes,
        "depth_values_cm": sorted(depth_values),
    }


def validate_protocol_against_dataset(dataset_root: Union[str, Path]) -> Dict[str, object]:
    """Check whether the real dataset axes match the locked protocol."""

    observed = scan_dataset_axes(dataset_root)
    expected = {
        "size_values_cm": list(SIZE_VALUES_CM),
        "depth_values_cm": list(DEPTH_VALUES_CM),
    }
    return {
        "expected": expected,
        "observed": observed,
        "size_axis_matches": expected["size_values_cm"] == observed["size_values_cm"],
        "depth_axis_matches": expected["depth_values_cm"] == observed["depth_values_cm"],
        "condition_count": len(observed["size_values_cm"]) * len(observed["depth_values_cm"]),
    }


__all__ = [
    "COARSE_DEPTH_ORDER",
    "COARSE_DEPTH_TO_VALUES",
    "DEPTH_VALUES_CM",
    "INPUT_CHANNELS",
    "INPUT_HEIGHT",
    "INPUT_SEQ_LEN",
    "INPUT_SHAPE",
    "INPUT_WIDTH",
    "MAIN_INVERSION_TASK",
    "OUTPUT_DETECTION",
    "OUTPUT_DEPTH_COARSE",
    "OUTPUT_SIZE_CLASS",
    "OUTPUT_SIZE_REGRESSION",
    "PRIMARY_TASK",
    "PROTOCOL_V1",
    "RUNTIME_ALWAYS_VISIBLE_FIELDS",
    "RUNTIME_GATED_FIELDS",
    "SECONDARY_TASK",
    "SIZE_VALUES_CM",
    "TaskProtocolV1",
    "WINDOW_STRIDE",
    "class_index_to_size",
    "coarse_index_to_name",
    "coarse_name_to_values",
    "depth_to_coarse_index",
    "depth_to_coarse_name",
    "format_runtime_payload",
    "infer_size_depth_from_record_parts",
    "parse_cm_text",
    "parse_group_key",
    "protocol_summary",
    "scan_dataset_axes",
    "should_display_inversion_outputs",
    "size_to_class_index",
    "size_to_class_name",
    "validate_protocol_against_dataset",
]
