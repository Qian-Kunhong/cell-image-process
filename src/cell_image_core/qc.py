"""Quality-control rules that do not know project labels or output paths."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeQC:
    touches_border: bool
    image_edge_gap_px: int
    edge_buffer_px: int
    keep: bool
    reason: str


def edge_qc_from_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    edge_buffer_px: int = 0,
) -> EdgeQC:
    if edge_buffer_px < 0:
        raise ValueError("edge_buffer_px must be nonnegative")
    min_row, min_col, max_row, max_col = bbox
    height, width = image_shape
    touches = bool(min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width)
    gap = int(min(min_row, min_col, height - max_row, width - max_col))
    keep = gap > edge_buffer_px
    reason = "touches_border" if touches else ("keep" if keep else f"within_{edge_buffer_px}px_image_edge")
    return EdgeQC(touches, gap, edge_buffer_px, keep, reason)

