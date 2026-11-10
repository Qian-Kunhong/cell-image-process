"""Image loading and shape conversion without project path assumptions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: Path | str, *, dtype=np.float32) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape={image.shape}: {path}")
    return image.astype(dtype)


def load_rgb_channel(
    path: Path | str,
    channel: str | int,
    *,
    dtype=np.float32,
) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an RGB image, got shape={image.shape}: {path}")
    index = {"red": 0, "green": 1, "blue": 2}.get(channel, channel)
    if not isinstance(index, int) or not 0 <= index < image.shape[2]:
        raise ValueError(f"Unsupported channel {channel!r} for shape={image.shape}")
    return image[..., index].astype(dtype)


def ensure_2d_image(image: np.ndarray, *, channel_index: int = 0) -> np.ndarray:
    """Return a 2D plane from common HW, HWC and CHW arrays.

    Multi-channel conversion is explicit: the selected channel is returned,
    not a luminance mixture. This preserves the historical DAPI behavior.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[-1] == 1:
            return image[..., 0]
        if image.shape[-1] in (3, 4):
            return image[..., channel_index]
        if image.shape[0] == 1:
            return image[0]
        if image.shape[0] in (3, 4):
            return image[channel_index]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_three_channel(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3:
        if image.shape[-1] == 3:
            return image
        if image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)
        if image.shape[0] == 3 and image.shape[-1] != 3:
            return np.moveaxis(image, 0, -1)
        if image.shape[0] == 1 and image.shape[-1] != 1:
            return np.repeat(np.moveaxis(image, 0, -1), 3, axis=-1)
    raise ValueError(f"Unsupported image shape for three-channel conversion: {image.shape}")


def normalize_percentile_uint8(
    image: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    denominator_floor: float = 1e-6,
    denominator_offset: float = 0.0,
    round_values: bool = True,
    finite_only: bool = True,
) -> np.ndarray:
    """Percentile-normalize an image while exposing legacy rounding choices."""
    array = np.asarray(image, dtype=np.float32)
    values = array[np.isfinite(array)] if finite_only else array
    if values.size == 0:
        raise ValueError("Cannot normalize an image without finite pixels")
    if denominator_floor < 0 or denominator_offset < 0:
        raise ValueError("denominator_floor and denominator_offset must be nonnegative")
    p_low, p_high = np.percentile(values, [lower, upper])
    denominator = max(float(p_high - p_low) + denominator_offset, denominator_floor)
    if denominator <= 0:
        raise ValueError("Normalization denominator is zero; set a positive floor or offset")
    scaled = np.clip(
        (array - p_low) / denominator,
        0.0,
        1.0,
    )
    values_255 = scaled * 255.0
    if round_values:
        values_255 = np.round(values_255)
    return values_255.astype(np.uint8)

