from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = (
    Path(__file__).resolve().parent
    / "analysis_outputs"
    / "hepatocyte_confocal_audit"
    / "yap_crowding"
)


def centered(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column] - frame.groupby("field_id")[column].transform("median")


def main() -> None:
    cells = pd.read_csv(ROOT / "yap_cell_level_recomputed.csv")
    cells = cells.loc[cells["edge_distance_um"] >= 5].copy()
    cells["stage"] = cells["sample"].str[1].astype(int)
    cells["log2_ratio"] = np.log2(cells["yap_nucleus_ring_ratio"])
    cells["ratio_centered"] = centered(cells, "log2_ratio")

    metrics = {
        "local_nuclear_fraction_14um": "Higher = more nuclear area nearby",
        "neighbors_within_14um": "Higher = more neighboring nuclei",
        "nearest_neighbor_um": "Lower = closer nearest nucleus",
        "nucleus_area_um2": "Nuclear area",
        "nucleus_eccentricity": "Nuclear elongation",
    }
    rows = []
    for metric, interpretation in metrics.items():
        cells[f"{metric}_centered"] = centered(cells, metric)
        pooled = spearmanr(cells[f"{metric}_centered"], cells["ratio_centered"])
        field_rhos = []
        for _, group in cells.groupby("field_id"):
            if len(group) >= 12 and group[metric].nunique() >= 3:
                field_rhos.append(spearmanr(group[metric], group["log2_ratio"]).statistic)
        rows.append(
            {
                "metric": metric,
                "interpretation": interpretation,
                "n_cells": len(cells),
                "pooled_within_field_rho": pooled.statistic,
                "pooled_p": pooled.pvalue,
                "n_fields": len(field_rhos),
                "median_field_rho": float(np.nanmedian(field_rhos)),
                "fraction_fields_same_direction_as_pooled": float(
                    np.mean(np.sign(field_rhos) == np.sign(pooled.statistic))
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "local_metric_sensitivity_summary.csv", index=False)
    print(output.to_string(index=False))

    stage_rows = []
    for stage, group in cells.groupby("stage"):
        result = spearmanr(
            centered(group, "local_nuclear_fraction_14um"),
            centered(group, "log2_ratio"),
        )
        stage_rows.append(
            {
                "stage_code": stage,
                "n_cells": len(group),
                "n_fields": group["field_id"].nunique(),
                "within_field_rho": result.statistic,
                "p": result.pvalue,
            }
        )
    stage_output = pd.DataFrame(stage_rows)
    stage_output.to_csv(ROOT / "local_crowding_by_stage.csv", index=False)
    print("\n", stage_output.to_string(index=False))


if __name__ == "__main__":
    main()
