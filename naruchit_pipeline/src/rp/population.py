"""Naruchit compatibility adapter for the shared population kernel."""
from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORE_SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(CORE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_SOURCE_ROOT))

from cell_image_core.population import (  # noqa: E402
    DEFAULT_QUANTILES,
    summarize_population as _summarize_population,
)


def summarize_population(
    cells: pd.DataFrame,
    group_col: str = "well_id",
    feature_cols: Iterable[str] | None = None,
    cluster_col: str | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Preserve the legacy API while using the project-neutral implementation.

    ``cluster_col`` remains an explicit opt-in compatibility name. The shared
    kernel calls it ``category_col`` so clustering is never implied as a
    required or validated step.
    """
    output = _summarize_population(
        cells,
        group_col=group_col,
        feature_cols=feature_cols,
        category_col=cluster_col,
        quantiles=quantiles,
    )
    if cluster_col:
        output = output.rename(
            columns={f"{cluster_col}__n_categories": f"{cluster_col}__n_clusters"}
        )
    return output
