"""Aggregate cell-level measurements to field/well-level descriptions."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _summarize_one(values: np.ndarray, name: str, quantiles: Sequence[float]) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    output: dict[str, float] = {f"{name}__n": int(valid.size)}
    if valid.size == 0:
        for suffix in ("mean", "median", "std", "iqr", "cv"):
            output[f"{name}__{suffix}"] = np.nan
        for quantile in quantiles:
            output[f"{name}__q{int(round(quantile * 100)):02d}"] = np.nan
        return output
    mean = float(np.mean(valid))
    output[f"{name}__mean"] = mean
    output[f"{name}__median"] = float(np.median(valid))
    output[f"{name}__std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan
    q25, q75 = np.percentile(valid, [25, 75])
    output[f"{name}__iqr"] = float(q75 - q25)
    output[f"{name}__cv"] = float(np.std(valid, ddof=1) / abs(mean)) if valid.size > 1 and mean != 0 else np.nan
    for quantile in quantiles:
        output[f"{name}__q{int(round(quantile * 100)):02d}"] = float(np.percentile(valid, quantile * 100))
    return output


def summarize_population(
    cells: pd.DataFrame,
    group_col: str = "well_id",
    feature_cols: Iterable[str] | None = None,
    category_col: str | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Return one row per group; category summaries are opt-in, never inferred."""
    if group_col not in cells.columns:
        raise KeyError(f"Missing population key: {group_col}")
    if feature_cols is None:
        excluded = {group_col, category_col}
        feature_cols = [
            column for column in cells.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(cells[column])
        ]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("No numeric feature columns to summarize")
    rows: list[dict[str, object]] = []
    for group_value, subset in cells.groupby(group_col, dropna=False):
        row: dict[str, object] = {"group_id": group_value, "n_cells": int(len(subset))}
        for column in feature_cols:
            row.update(_summarize_one(subset[column].to_numpy(dtype=float, na_value=np.nan), column, quantiles))
        if category_col and category_col in subset.columns:
            shares = subset[category_col].value_counts(normalize=True, dropna=False).sort_index()
            for category, proportion in shares.items():
                row[f"{category_col}__share_{category}"] = float(proportion)
            probabilities = shares.to_numpy(dtype=float)
            probabilities = probabilities[probabilities > 0]
            row[f"{category_col}__entropy"] = float(-(probabilities * np.log(probabilities)).sum()) if probabilities.size else np.nan
            row[f"{category_col}__n_categories"] = float(shares.size)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group_id", ignore_index=True)

