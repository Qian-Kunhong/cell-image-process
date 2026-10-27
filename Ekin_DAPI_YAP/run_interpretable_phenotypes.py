"""Primary PyCharm entry for a more parsimonious, interpretable phenotype model.

The model uses only direct DAPI nuclear morphology and neighborhood features.
All DAPI intensity and engineered composite features are disabled.  K remains
data-selected; after preserving background gaps between touching nuclear labels,
the current 40x dataset selects three phenotypes.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ekin_dapi_yap.model_a_pipeline import parse_args, run

MAGNIFICATION = "40x"
OUTPUT_PROFILE = "interpretable_no_dapi_intensity"

# Neutral morphology descriptions, not inferred cell-cycle or quality labels.
PHENOTYPE_LABELS = [
    {"dominant_phenotype": "Phenotype 1", "short_label": "Compact / dense",
     "heatmap_label": "P1 · Compact / dense"},
    {"dominant_phenotype": "Phenotype 2", "short_label": "Large / sparse",
     "heatmap_label": "P2 · Large / sparse"},
    {"dominant_phenotype": "Phenotype 3", "short_label": "Elongated / irregular",
     "heatmap_label": "P3 · Elongated / irregular"},
]

# True = enter preprocessing/PCA/GMM; False = excluded from all three.
FEATURE_SWITCHES = {
    # Direct nuclear size and shape
    "area_px": True,
    "perimeter_px": True,
    "equivalent_diameter_px": True,
    "major_axis_length_px": True,
    "minor_axis_length_px": True,
    "aspect_ratio": True,
    "eccentricity": True,
    "circularity": True,
    "solidity": True,
    "extent": True,
    # DAPI intensity: excluded
    "dapi_mean_intensity": False,
    "dapi_std_intensity": False,
    "dapi_min_intensity": False,
    "dapi_max_intensity": False,
    "dapi_intensity_range": False,
    # Direct spatial and neighboring morphology
    "nn1_distance_px": True,
    "knn6_distance_mean_px": True,
    "knn6_distance_std_px": True,
    "local_density_per_px2": True,
    "adaptive_neighbor_count": True,
    "nb_area_mean_px2": True,
    "nb_area_std_px2": True,
    "nb_circularity_mean": True,
    "nb_circularity_std": True,
    "nb_eccentricity_mean": True,
    "nb_eccentricity_std": True,
    "nb_aspect_ratio_mean": True,
    "nb_aspect_ratio_std": True,
    "nb_dapi_mean_intensity_mean": False,
    "nb_dapi_mean_intensity_std": False,
    "fixed_neighbor_count": True,
    # Engineered composite features: excluded from the primary phenotype model
    "chromatin_cv_proxy": False,
    "chromatin_range_ratio": False,
    "nearest_spacing_nuclear_units": False,
    "local_crowding_area_fraction_proxy": False,
    "neighbor_size_log_disagreement": False,
    "neighbor_shape_disagreement": False,
    "neighborhood_angular_asymmetry": False,
}


if __name__ == "__main__":
    args = parse_args()
    args.fit_magnification = MAGNIFICATION
    args.feature_set = "baseline"
    if not any(item == "--output-root" or item.startswith("--output-root=") for item in sys.argv[1:]):
        args.output_root = (Path(__file__).resolve().parent / "outputs" / OUTPUT_PROFILE /
                            args.fit_magnification / "all_fields")
    args.feature_switches = FEATURE_SWITCHES.copy()
    run(args)
    label_path = args.output_root / "tables" / "phenotype_labels.csv"
    pd.DataFrame(PHENOTYPE_LABELS).to_csv(label_path, index=False, encoding="utf-8-sig")
    # Reuse the centralized presentation plotter after model fitting.  It reads
    # phenotype_labels.csv, so the selected components are shown with neutral,
    # morphology-based descriptions instead of legacy class notes.
    from make_ppt_figures import main as make_ppt_figures
    make_ppt_figures(["--output-root", str(args.output_root)])
    # figures_ppt is the single official summary set. Keep only non-duplicated
    # field-level/QC subdirectories below figures; remove legacy top-level charts.
    for legacy_chart in (args.output_root / "figures").glob("*.png"):
        legacy_chart.unlink()
