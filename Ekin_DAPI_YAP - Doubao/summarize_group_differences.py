"""Create descriptive between-group tables without treating cells as replicates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ("Ctrl", "HA1", "HA2")
CONTRASTS = (("HA1", "Ctrl"), ("HA2", "Ctrl"), ("HA2", "HA1"))
DEFAULT_METRICS = (
    "area_px", "aspect_ratio", "circularity", "solidity",
    "chromatin_cv_proxy", "chromatin_range_ratio",
    "nearest_spacing_nuclear_units", "local_crowding_area_fraction_proxy",
    "neighbor_size_log_disagreement", "neighbor_shape_disagreement",
    "neighborhood_angular_asymmetry", "gmm_max_posterior",
    "posthoc_yap_log2_nuclear_perinuclear_ratio",
)


def _numeric_valid(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    if metric.startswith("posthoc_yap_") and "posthoc_yap_ratio_valid" in frame:
        valid = frame["posthoc_yap_ratio_valid"].astype(str).str.lower().isin(("true", "1"))
        values = values.where(valid)
    return values.dropna()


def build_descriptive_tables(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"magnification", "experimental_group_label", "seeding_density_cells_per_cm2", "image_id", "dominant_phenotype"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    magnifications = cells["magnification"].astype(str).unique()
    if len(magnifications) != 1:
        raise ValueError("One table must contain exactly one independently fitted magnification")
    metrics = [metric for metric in DEFAULT_METRICS if metric in cells]

    summary_rows = []
    keys = ["magnification", "experimental_group_label", "seeding_density_cells_per_cm2", "image_id"]
    for key, frame in cells.groupby(keys, sort=True):
        for metric in metrics:
            values = _numeric_valid(frame, metric)
            summary_rows.append({
                **dict(zip(keys, key)), "metric": metric, "n_cells_valid": len(values),
                "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75),
                "iqr": values.quantile(.75) - values.quantile(.25), "mean": values.mean(), "sd": values.std(ddof=1),
            })
    field_summary = pd.DataFrame(summary_rows)

    contrast_rows = []
    for (magnification, density, metric), frame in field_summary.groupby(
        ["magnification", "seeding_density_cells_per_cm2", "metric"], sort=True
    ):
        by_group = frame.set_index("experimental_group_label")
        for numerator, reference in CONTRASTS:
            if numerator not in by_group.index or reference not in by_group.index:
                continue
            a, b = by_group.loc[numerator], by_group.loc[reference]
            ref = float(b["median"])
            difference = float(a["median"] - b["median"])
            is_log2 = metric == "posthoc_yap_log2_nuclear_perinuclear_ratio"
            contrast_rows.append({
                "magnification": magnification, "seeding_density_cells_per_cm2": density,
                "contrast": f"{numerator} - {reference}", "metric": metric,
                "numerator_field_median": a["median"], "reference_field_median": b["median"],
                "median_difference": difference,
                "median_percent_change_vs_reference": (
                    np.nan if is_log2 else
                    (float(100 * difference / abs(ref)) if np.isfinite(ref) and ref != 0 else np.nan)
                ),
                "median_fold_change_from_log2_difference": float(2 ** difference) if is_log2 else np.nan,
                "effect_scale": "log2 difference and back-transformed fold change" if is_log2 else "raw median difference",
                "note": "descriptive single-field contrast; no p-value or biological-replicate inference",
            })
    same_density_contrasts = pd.DataFrame(contrast_rows)

    composition = (
        cells.groupby(["magnification", "experimental_group_label", "seeding_density_cells_per_cm2", "image_id", "dominant_phenotype"])
        .size().rename("n_cells").reset_index()
    )
    totals = composition.groupby(["magnification", "experimental_group_label", "seeding_density_cells_per_cm2", "image_id"])["n_cells"].transform("sum")
    composition["cell_fraction"] = composition["n_cells"] / totals
    all_index = pd.MultiIndex.from_product([
        magnifications, GROUPS,
        sorted(cells["seeding_density_cells_per_cm2"].unique()),
        sorted(cells["dominant_phenotype"].unique(), key=lambda x: int(str(x).split()[-1])),
    ], names=["magnification", "experimental_group_label", "seeding_density_cells_per_cm2", "dominant_phenotype"])
    comp = composition.groupby(list(all_index.names))["cell_fraction"].sum().reindex(all_index, fill_value=0).rename("cell_fraction").reset_index()
    comp_contrasts = []
    for (mag, density, phenotype), frame in comp.groupby(["magnification", "seeding_density_cells_per_cm2", "dominant_phenotype"]):
        values = frame.set_index("experimental_group_label")["cell_fraction"]
        for numerator, reference in CONTRASTS:
            comp_contrasts.append({
                "magnification": mag, "seeding_density_cells_per_cm2": density,
                "contrast": f"{numerator} - {reference}", "dominant_phenotype": phenotype,
                "numerator_fraction": values.get(numerator, 0.0), "reference_fraction": values.get(reference, 0.0),
                "fraction_difference": values.get(numerator, 0.0) - values.get(reference, 0.0),
                "percentage_point_difference": 100 * (values.get(numerator, 0.0) - values.get(reference, 0.0)),
                "note": "descriptive single-field composition contrast; phenotype is magnification-specific",
            })
    return {
        "field_descriptive_statistics": field_summary,
        "same_density_group_contrasts": same_density_contrasts,
        "phenotype_composition": composition,
        "same_density_phenotype_composition_contrasts": pd.DataFrame(comp_contrasts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_csv.parent
    output.mkdir(parents=True, exist_ok=True)
    tables = build_descriptive_tables(pd.read_csv(args.input_csv))
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"[written] {name}.csv: {len(table)} rows")


if __name__ == "__main__":
    main()
