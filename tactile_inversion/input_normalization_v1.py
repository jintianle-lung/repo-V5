import os
from typing import Optional, Tuple

import numpy as np


def resolve_pressure_conversion(scale: Optional[float] = None, offset: Optional[float] = None) -> Tuple[float, float]:
    if scale is None:
        scale = float(os.environ.get("PAPERV1_PRESSURE_SCALE", "1.0"))
    if offset is None:
        offset = float(os.environ.get("PAPERV1_PRESSURE_OFFSET", "0.0"))
    return float(scale), float(offset)


def convert_sensor_to_pressure(data: np.ndarray, scale: Optional[float] = None, offset: Optional[float] = None) -> np.ndarray:
    scale, offset = resolve_pressure_conversion(scale, offset)
    arr = np.asarray(data, dtype=np.float32)
    return (arr * scale + offset).astype(np.float32, copy=False)


def convert_sensor_to_pressure_maps(
    data: np.ndarray,
    scale: Optional[float] = None,
    offset: Optional[float] = None,
) -> np.ndarray:
    arr = convert_sensor_to_pressure(data, scale=scale, offset=offset)
    if arr.ndim == 2 and arr.shape[1] == 96:
        return arr.reshape(arr.shape[0], 12, 8).astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[1:] == (12, 8):
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Expected (T,96) or (T,12,8), got shape={arr.shape}")


def resolve_raw_norm_bounds(lo: Optional[float] = None, hi: Optional[float] = None) -> Tuple[float, float]:
    if lo is None:
        lo = float(os.environ.get("PAPERV1_RAW_NORM_LO", "0.0"))
    if hi is None:
        hi = float(os.environ.get("PAPERV1_RAW_NORM_HI", "130.0"))
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        raise ValueError(f"Invalid PAPERV1 raw normalization bounds: lo={lo}, hi={hi}")
    return lo, hi


def normalize_raw_frames_global(
    data: np.ndarray,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    out_hi: float = 1.0,
) -> np.ndarray:
    lo, hi = resolve_raw_norm_bounds(lo, hi)
    maps = convert_sensor_to_pressure_maps(data)
    maps = (maps - lo) / max(hi - lo, 1e-6)
    maps = np.clip(maps, 0.0, 1.0) * float(out_hi)
    return maps.astype(np.float32, copy=False)


def normalize_raw_frames_window_minmax(
    data: np.ndarray,
    out_hi: float = 1.0,
) -> np.ndarray:
    maps = convert_sensor_to_pressure_maps(data)
    mn = float(np.min(maps))
    mx = float(np.max(maps))
    if mx <= mn + 1e-6:
        return np.zeros_like(maps, dtype=np.float32)
    maps = (maps - mn) / (mx - mn)
    maps = np.clip(maps, 0.0, 1.0) * float(out_hi)
    return maps.astype(np.float32, copy=False)
