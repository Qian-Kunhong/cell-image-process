"""Composite Model A with one visible switch for every model feature.

Current profile excludes all direct and derived DAPI-intensity features to test
whether illumination/exposure nonuniformity drives the morphology phenotypes.
Edit FEATURE_SWITCHES below, then run this file in PyCharm.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.entrypoint import configured_args
from yap_40x_trial import parse_args, run

MAGNIFICATION = "40x"
OUTPUT_PROFILE = "composite_no_dapi_intensity"

# True = enter preprocessing/PCA/GMM; False = excluded from all three.
FEATURE_SWITCHES = {
    # Nuclear size and shape
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
    # Direct nuclear DAPI intensity (disabled for this trial)
    "dapi_mean_intensity": False,
    "dapi_std_intensity": False,
    "dapi_min_intensity": False,
    "dapi_max_intensity": False,
    "dapi_intensity_range": False,
    # Spatial context and neighboring nuclear morphology
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
    # Neighbor DAPI intensity (disabled for this trial)
    "nb_dapi_mean_intensity_mean": False,
    "nb_dapi_mean_intensity_std": False,
    "fixed_neighbor_count": True,
    # Composite features
    "chromatin_cv_proxy": False,
    "chromatin_range_ratio": False,
    "nearest_spacing_nuclear_units": True,
    "local_crowding_area_fraction_proxy": True,
    "neighbor_size_log_disagreement": True,
    "neighbor_shape_disagreement": True,
    "neighborhood_angular_asymmetry": True,
}


if __name__ == "__main__":
    args = configured_args(parse_args, Path(__file__).resolve().parent, "composite", MAGNIFICATION)
    if not any(item == "--output-root" or item.startswith("--output-root=") for item in sys.argv[1:]):
        args.output_root = (Path(__file__).resolve().parent / "outputs" / OUTPUT_PROFILE
                            / args.fit_magnification / "all_fields")
    args.feature_switches = FEATURE_SWITCHES.copy()
    run(args)
