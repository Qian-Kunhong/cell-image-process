"""Audit candidate, intensity-free feature panels before choosing a primary GMM.

This script reads an existing single-cell table and does not modify pipeline
outputs.  It reports dynamic K, posterior confidence, seed stability, cluster
sizes and silhouette for each predeclared feature panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import dapi_model_a as core


PANELS = {
    "raw_no_intensity": [
        name for name in core.MODEL_FEATURES
        if "intensity" not in name.lower() and not name.startswith("dapi_")
    ],
    "curated_morphology_9": [
        "area_px", "aspect_ratio", "circularity", "solidity",
        "nearest_spacing_nuclear_units", "local_crowding_area_fraction_proxy",
        "neighbor_size_log_disagreement", "neighbor_shape_disagreement",
        "neighborhood_angular_asymmetry",
    ],
    "compact_morphology_6": [
        "area_px", "aspect_ratio", "circularity",
        "nearest_spacing_nuclear_units", "local_crowding_area_fraction_proxy",
        "neighbor_shape_disagreement",
    ],
    "nuclear_shape_5": [
        "area_px", "aspect_ratio", "circularity", "solidity", "extent",
    ],
}


def evaluate(cells: pd.DataFrame) -> list[dict]:
    reports: list[dict] = []
    for panel_name, enabled in PANELS.items():
        switches = {name: name in enabled for name in core.ALL_SWITCHABLE_MODEL_FEATURES}
        scaled, columns, preprocessing = core.preprocess_features(
            cells, feature_set="augmented", feature_switches=switches)
        _, labels, probabilities, _, selection, selected_k = core.fit_gmm_with_dynamic_k(scaled)
        alternate_ari = []
        for seed in (7, 19, 73):
            _, alternate, _, _, _, _ = core.fit_gmm_with_dynamic_k(scaled, random_state=seed)
            alternate_ari.append(float(adjusted_rand_score(labels, alternate)))
        counts = np.bincount(labels, minlength=selected_k)
        selected = selection[selection["n_clusters"].eq(selected_k)].iloc[0]
        reports.append({
            "panel": panel_name,
            "features": columns,
            "n_features": len(columns),
            "selected_k": selected_k,
            "mean_max_posterior": float(probabilities.max(axis=1).mean()),
            "fraction_max_posterior_below_0_8": float((probabilities.max(axis=1) < .8).mean()),
            "seed_stability_ari_mean": float(np.mean(alternate_ari)),
            "seed_stability_ari_values": alternate_ari,
            "silhouette": float(selected["silhouette"]),
            "minimum_cluster_fraction": float(counts.min() / counts.sum()),
            "cluster_counts": sorted((int(value) for value in counts), reverse=True),
            "preprocessing": preprocessing,
        })
    return reports


def main(argv: list[str] | None = None) -> None:
    default = (Path(__file__).resolve().parent / "outputs" / "composite_no_dapi_intensity" /
               "40x" / "all_fields" / "tables" / "model_a_single_cell_results.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=default)
    args = parser.parse_args(argv)
    cells = pd.read_csv(args.cells)
    print(json.dumps(evaluate(cells), indent=2))


if __name__ == "__main__":
    main()
