"""Segmentation adapters and deterministic label post-processing."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import SegmentationResult
from .io import to_three_channel


def load_cellpose_model(*, use_gpu: bool, pretrained_model: str = "cpsam"):
    """Load Cellpose lazily so non-segmentation utilities remain importable."""
    from cellpose import models

    return models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model)


def segment_cellpose(
    image: np.ndarray,
    *,
    model,
    image_id: str,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    diameter: float | None = None,
    separate_touching: bool = False,
    extra_parameters: dict[str, Any] | None = None,
) -> SegmentationResult:
    parameters: dict[str, Any] = {
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "diameter": diameter,
    }
    overlap = set(parameters) & set(extra_parameters or {})
    if overlap:
        raise ValueError(f"extra_parameters duplicates explicit arguments: {sorted(overlap)}")
    parameters.update(extra_parameters or {})
    cellpose_input = to_three_channel(image)
    masks, _, _ = model.eval(
        cellpose_input,
        **parameters,
    )
    labels = np.asarray(masks, dtype=np.int32)
    if labels.shape != cellpose_input.shape[:2]:
        raise ValueError(f"Mask/image shape mismatch for {image_id}: {labels.shape} vs {cellpose_input.shape}")
    if separate_touching:
        labels = separate_touching_labels(labels)
    return SegmentationResult(
        image_id=image_id,
        labels=labels,
        algorithm="cellpose",
        parameters=parameters,
    )


def separate_touching_labels(mask: np.ndarray) -> np.ndarray:
    """Insert background only where two different labels directly touch."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a 2D integer label mask")
    contacts = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for row_shift, col_shift in ((1, 0), (0, 1), (1, 1), (1, -1)):
        src_rows = slice(max(0, -row_shift), min(height, height - row_shift))
        src_cols = slice(max(0, -col_shift), min(width, width - col_shift))
        dst_rows = slice(max(0, row_shift), min(height, height + row_shift))
        dst_cols = slice(max(0, col_shift), min(width, width + col_shift))
        left = mask[src_rows, src_cols]
        right = mask[dst_rows, dst_cols]
        different = (left > 0) & (right > 0) & (left != right)
        contacts[src_rows, src_cols] |= different
        contacts[dst_rows, dst_cols] |= different
    separated = mask.copy()
    separated[contacts] = 0
    return separated


def label_boundaries_4connected(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D label mask")
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary

