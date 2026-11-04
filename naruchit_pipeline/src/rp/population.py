"""单细胞表 -> 群体级特征表。

科研上真正的分析单位是孔/视野，细胞只是技术子样本。这个模块把细胞级测量压成
孔级描述量，保留分布形状（分位数、离散度、亚群占比、熵），而不只是均值——
因为“群体品质由异质性决定”这类假设，用均值表示根本测不出来。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _summarize_one(values: np.ndarray, name: str, quantiles: Sequence[float]) -> dict[str, float]:
    v = values[np.isfinite(values)]
    out: dict[str, float] = {f"{name}__n": int(v.size)}
    if v.size == 0:
        for suffix in ("mean", "median", "std", "iqr", "cv"):
            out[f"{name}__{suffix}"] = np.nan
        for q in quantiles:
            out[f"{name}__q{int(round(q * 100)):02d}"] = np.nan
        return out

    mean = float(np.mean(v))
    out[f"{name}__mean"] = mean
    out[f"{name}__median"] = float(np.median(v))
    out[f"{name}__std"] = float(np.std(v, ddof=1)) if v.size > 1 else np.nan
    q25, q75 = np.percentile(v, [25, 75])
    out[f"{name}__iqr"] = float(q75 - q25)
    out[f"{name}__cv"] = float(np.std(v, ddof=1) / abs(mean)) if v.size > 1 and mean != 0 else np.nan
    for q in quantiles:
        out[f"{name}__q{int(round(q * 100)):02d}"] = float(np.percentile(v, q * 100))
    return out


def summarize_population(
    cells: pd.DataFrame,
    group_col: str = "well_id",
    feature_cols: Iterable[str] | None = None,
    cluster_col: str | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """按 group_col 聚合出一行一个群体的特征表。

    Args:
        cells: 细胞级表，一行一个细胞。
        group_col: 群体键，通常是 well_id 或 field_id。
        feature_cols: 要汇总的数值列；None 表示自动取全部数值列（排除 group/cluster 列）。
        cluster_col: 若有无监督簇标签，额外输出各簇占比与香农熵。
        quantiles: 分位数序列。

    Returns:
        一行一个群体。列名形如 ``<feature>__mean``、``<feature>__q25``、
        ``<cluster>__share_<k>``、``<cluster>__entropy``。
    """
    if group_col not in cells.columns:
        raise KeyError(f"缺少群体列：{group_col}")

    if feature_cols is None:
        drop = {group_col}
        if cluster_col:
            drop.add(cluster_col)
        feature_cols = [
            c for c in cells.columns
            if c not in drop and pd.api.types.is_numeric_dtype(cells[c])
        ]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("没有可汇总的数值特征列")

    rows: list[dict[str, float]] = []
    for group_value, sub in cells.groupby(group_col, dropna=False):
        row: dict[str, float] = {"group_id": group_value, "n_cells": int(len(sub))}
        for col in feature_cols:
            values = sub[col].to_numpy(dtype=float, na_value=np.nan)
            row.update(_summarize_one(values, col, quantiles))

        if cluster_col and cluster_col in sub.columns:
            share = sub[cluster_col].value_counts(normalize=True, dropna=False).sort_index()
            for k, p in share.items():
                row[f"{cluster_col}__share_{k}"] = float(p)
            p = share.to_numpy(dtype=float)
            p = p[p > 0]
            row[f"{cluster_col}__entropy"] = float(-(p * np.log(p)).sum()) if p.size else np.nan
            row[f"{cluster_col}__n_clusters"] = float(share.size)

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("group_id", ignore_index=True)
    return out
