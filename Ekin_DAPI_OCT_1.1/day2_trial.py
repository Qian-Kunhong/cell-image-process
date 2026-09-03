from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


DEFAULT_DATA_ROOT = Path(
    r"E:\Kino-oka Lab\Immunostaining Data_Ekin\Immunostaining Data_Ekin\Day 2 Data"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "day2_trial"

RANDOM_STATE = 42
FIT_MAGNIFICATION = "40x"
K_MIN = 2
K_MAX = 12
GMM_COVARIANCE_TYPE = "diag"
GMM_N_INIT = 5
PCA_COMPONENTS = 5
K_NEIGHBORS = 6
UMAP_N_EPOCHS = 50  # Single-image trial visualization; does not affect GMM fitting.

# Existing repository method from 04-b_adaptive_clustering.py.
PREFERRED_MAX_CLUSTERS = 5
SMALL_CLUSTER_FRACTION = 0.08
TINY_CLUSTER_FRACTION = 0.05
EXCESS_CLUSTER_PENALTY = 12.0
SMALL_CLUSTER_DEFICIT_WEIGHT = 250.0
TINY_CLUSTER_DEFICIT_WEIGHT = 500.0

BASC_IMPLEMENTATION = "Python port of CRAN Binarize 1.3.1 BASC A threshold core"
BASC_SOURCE = "https://cran.r-project.org/src/contrib/Archive/Binarize/Binarize_1.3.1.tar.gz"


MODEL_FEATURES = [
    # Nuclear morphology.
    "area_px",
    "perimeter_px",
    "equivalent_diameter_px",
    "major_axis_length_px",
    "minor_axis_length_px",
    "aspect_ratio",
    "eccentricity",
    "circularity",
    "solidity",
    "extent",
    # DAPI/DNA-channel intensity. The available trial data are DAPI, not Hoechst.
    "dapi_mean_intensity",
    "dapi_std_intensity",
    "dapi_min_intensity",
    "dapi_max_intensity",
    "dapi_intensity_range",
    # Existing kNN / spatial-organization feature families.
    "nn1_distance_px",
    "knn6_distance_mean_px",
    "knn6_distance_std_px",
    "local_density_per_px2",
    "adaptive_neighbor_count",
    "nb_area_mean_px2",
    "nb_area_std_px2",
    "nb_circularity_mean",
    "nb_circularity_std",
    "nb_eccentricity_mean",
    "nb_eccentricity_std",
    "nb_aspect_ratio_mean",
    "nb_aspect_ratio_std",
    "nb_dapi_mean_intensity_mean",
    "nb_dapi_mean_intensity_std",
    "fixed_neighbor_count",
]

# Dimensionless, DAPI-only morphology/spatial proxies.  These names deliberately
# describe measurements rather than unverified biological states.
COMPOSITE_FEATURES = [
    "nuclear_size_log_ratio",
    "nuclear_elongation_log",
    "boundary_irregularity",
    "convexity_deficit",
    "chromatin_cv_proxy",
    "chromatin_range_ratio",
    "nearest_spacing_nuclear_units",
    "local_crowding_area_fraction_proxy",
    "neighbor_size_log_disagreement",
    "neighbor_shape_disagreement",
    "neighborhood_angular_asymmetry",
]

# Algebraic restatements of an existing single-cell column remain useful in
# exported interpretation tables, but including them would implicitly double
# weight area/aspect-ratio/circularity/solidity.  Only the relative-intensity
# and spatial-context composites below enter the augmented model.
COMPOSITE_MODEL_FEATURES = [
    "chromatin_cv_proxy",
    "chromatin_range_ratio",
    "nearest_spacing_nuclear_units",
    "local_crowding_area_fraction_proxy",
    "neighbor_size_log_disagreement",
    "neighbor_shape_disagreement",
    "neighborhood_angular_asymmetry",
]

FORBIDDEN_MODEL_TOKENS = (
    "oct4", "oct_4", "oct-4", "af488", "yap", "experimental_group",
    "ha_exposure", "ha_concentration", "seeding_density", "culture_day",
    "treatment", "time", "dose", "replicate", "sample",
)

META_COLUMNS = [
    "cell_id",
    "culture_day",
    "replicate",
    "sample",
    "magnification",
    "imaging_condition",
    "image_id",
    "source_snap_id",
    "label",
    "centroid_row_px",
    "centroid_col_px",
    "dapi_image_path",
    "oct4_image_path",
    "merge_image_path",
]


@dataclass(frozen=True)
class ImagePair:
    culture_day: int
    replicate: str
    sample: str
    magnification: str
    image_id: str
    source_snap_id: str
    dapi_path: Path
    oct4_path: Path
    merge_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feeder-free Model A single-image trial using DAPI/DNA features only for GMM."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fit-magnification", default=FIT_MAGNIFICATION)
    parser.add_argument("--culture-day", type=int, default=2)
    parser.add_argument("--sample", default="not_provided")
    parser.add_argument("--replicate", default="not_provided")
    parser.add_argument("--cpu", action="store_true", help="Run Cellpose on CPU.")
    parser.add_argument(
        "--reuse-masks",
        action="store_true",
        help="Reuse existing masks under the selected output directory.",
    )
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)


def normalize_snap_id(stem: str) -> str:
    base = re.sub(r"_(DAPI|AF488)$", "", stem, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    return base


def display_trial_label(culture_day: int, sample: str) -> str:
    suffix = "" if sample == "not_provided" else f" / {sample}"
    return f"Day {culture_day}{suffix}"


def trial_file_prefix(culture_day: int, sample: str) -> str:
    prefix = f"day{culture_day}"
    if sample != "not_provided":
        sample_slug = re.sub(r"[^a-z0-9]+", "_", sample.lower()).strip("_")
        prefix = f"{prefix}_{sample_slug}"
    return prefix


def discover_image_pairs(
    data_root: Path,
    culture_day: int,
    sample: str,
    replicate: str,
) -> tuple[list[ImagePair], pd.DataFrame]:
    if not data_root.exists():
        raise FileNotFoundError(f"Trial data root not found: {data_root}")

    dapi_root = data_root / "DAPI"
    oct_root = data_root / "OCT"
    merge_root = data_root / "Merge"
    if not merge_root.exists():
        merge_root = data_root / "Merged"
    dapi_files = sorted(dapi_root.glob("*/*.png"))
    oct_files = sorted(oct_root.glob("*/*.png"))
    merge_files = sorted(merge_root.glob("*/*.png"))
    if not dapi_files or not oct_files:
        raise RuntimeError("DAPI or OCT PNG files were not found under the trial directory.")

    oct_index: dict[tuple[str, str], Path] = {}
    for path in oct_files:
        key = (path.parent.name.lower(), normalize_snap_id(path.stem).lower())
        if key in oct_index:
            raise ValueError(f"Duplicated OCT pair key: {key}")
        oct_index[key] = path

    merge_index: dict[tuple[str, str], Path] = {}
    for path in merge_files:
        key = (path.parent.name.lower(), normalize_snap_id(path.stem).lower())
        if key in merge_index:
            raise ValueError(f"Duplicated Merge pair key: {key}")
        merge_index[key] = path

    pairs: list[ImagePair] = []
    report: list[dict] = []
    for dapi_path in dapi_files:
        magnification = dapi_path.parent.name
        snap_id = normalize_snap_id(dapi_path.stem)
        key = (magnification.lower(), snap_id.lower())
        oct_path = oct_index.get(key)
        merge_path = merge_index.get(key)
        status = "ok" if oct_path is not None else "missing_oct4"
        report.append(
            {
                "culture_day": culture_day,
                "replicate": replicate,
                "sample": sample,
                "magnification": magnification,
                "source_snap_id": snap_id,
                "dapi_image_path": str(dapi_path),
                "oct4_image_path": str(oct_path) if oct_path else None,
                "merge_image_path": str(merge_path) if merge_path else None,
                "pairing_status": status,
            }
        )
        if oct_path is None:
            continue
        sample_id = "" if sample == "not_provided" else f"_{normalize_snap_id(sample)}"
        image_id = f"D{culture_day}{sample_id}_{magnification}_{snap_id}"
        pairs.append(
            ImagePair(
                culture_day=culture_day,
                replicate=replicate,
                sample=sample,
                magnification=magnification,
                image_id=image_id,
                source_snap_id=snap_id,
                dapi_path=dapi_path,
                oct4_path=oct_path,
                merge_path=merge_path,
            )
        )

    if not pairs:
        raise RuntimeError("No valid DAPI/OCT pairs were found for the requested trial.")
    return pairs, pd.DataFrame(report)


def load_effective_channel(path: Path, channel: str) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB PNG, got shape={image.shape}: {path}")
    channel_index = {"red": 0, "green": 1, "blue": 2}[channel]
    return image[..., channel_index].astype(np.float32)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    p1, p99 = np.percentile(arr[np.isfinite(arr)], [1, 99])
    scaled = np.clip((arr - p1) / max(float(p99 - p1), 1e-6), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def save_mask_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    base = normalize_to_uint8(image)
    rgb = np.stack([base, base, base], axis=-1)
    boundary = find_boundaries(mask, mode="outer")
    rgb[boundary] = np.array([255, 40, 40], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def load_cellpose_model(use_gpu: bool):
    from cellpose import models

    return models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")


def segment_pairs(
    pairs: list[ImagePair],
    output_root: Path,
    use_gpu: bool,
    reuse_masks: bool,
) -> dict[str, Path]:
    mask_dir = output_root / "segmentation" / "masks"
    overlay_dir = output_root / "segmentation" / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = {pair.image_id: mask_dir / f"{pair.image_id}_mask.npy" for pair in pairs}
    pending = [pair for pair in pairs if not (reuse_masks and mask_paths[pair.image_id].exists())]
    model = load_cellpose_model(use_gpu=use_gpu) if pending else None

    for pair in pairs:
        mask_path = mask_paths[pair.image_id]
        dapi = load_effective_channel(pair.dapi_path, "blue")
        if mask_path.exists() and reuse_masks:
            mask = np.load(mask_path)
        else:
            cellpose_input = np.stack([dapi, dapi, dapi], axis=-1)
            masks, _, _ = model.eval(
                cellpose_input,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                diameter=None,
            )
            mask = np.asarray(masks, dtype=np.int32)
            np.save(mask_path, mask)
        if mask.shape != dapi.shape:
            raise ValueError(f"Mask/image shape mismatch for {pair.image_id}: {mask.shape} vs {dapi.shape}")
        save_mask_overlay(dapi, mask, overlay_dir / f"{pair.image_id}_overlay.png")
        print(f"[segmentation] {pair.image_id}: nuclei={int(mask.max())}")
    return mask_paths


def safe_div(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(b) and b > 0 else np.nan


def circularity(area: float, perimeter: float) -> float:
    return safe_div(4.0 * math.pi * area, perimeter * perimeter)


def fixed_radius_neighbor_count(coords: np.ndarray, radius: float) -> np.ndarray:
    if len(coords) == 0:
        return np.array([], dtype=int)
    tree = cKDTree(coords)
    return np.asarray(
        [len(tree.query_ball_point(coords[i], r=radius)) - 1 for i in range(len(coords))],
        dtype=int,
    )


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    spatial_columns = [
        "nn1_distance_px",
        "knn6_distance_mean_px",
        "knn6_distance_std_px",
        "local_density_per_px2",
        "adaptive_neighbor_count",
        "nb_area_mean_px2",
        "nb_area_std_px2",
        "nb_circularity_mean",
        "nb_circularity_std",
        "nb_eccentricity_mean",
        "nb_eccentricity_std",
        "nb_aspect_ratio_mean",
        "nb_aspect_ratio_std",
        "nb_dapi_mean_intensity_mean",
        "nb_dapi_mean_intensity_std",
        "fixed_neighbor_count",
        "local_crowding_area_fraction_proxy",
        "neighbor_size_log_disagreement",
        "neighbor_shape_disagreement",
        "neighborhood_angular_asymmetry",
    ]
    for col in spatial_columns:
        out[col] = np.nan
    if n <= 1:
        out["adaptive_neighbor_count"] = 0
        out["fixed_neighbor_count"] = 0
        return out

    coords = out[["centroid_row_px", "centroid_col_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    k_eff = min(K_NEIGHBORS, n - 1)
    distances, indices = tree.query(coords, k=k_eff + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    area = out["area_px"].to_numpy(dtype=float)
    circ = out["circularity"].to_numpy(dtype=float)
    ecc = out["eccentricity"].to_numpy(dtype=float)
    aspect = out["aspect_ratio"].to_numpy(dtype=float)
    dapi_mean = out["dapi_mean_intensity"].to_numpy(dtype=float)

    for i in range(n):
        d = np.asarray(distances[i][1:], dtype=float)
        idx = np.asarray(indices[i][1:], dtype=int)
        if len(d) == 0:
            continue
        radius = float(d[-1])
        neighbors = np.asarray(
            [j for j in tree.query_ball_point(coords[i], r=radius + 1e-8) if j != i],
            dtype=int,
        )
        out.at[i, "nn1_distance_px"] = float(d[0])
        out.at[i, "knn6_distance_mean_px"] = float(np.mean(d))
        out.at[i, "knn6_distance_std_px"] = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        out.at[i, "adaptive_neighbor_count"] = int(len(neighbors))
        out.at[i, "local_density_per_px2"] = safe_div(len(neighbors), math.pi * radius * radius)
        if len(neighbors) == 0:
            continue

        vectors = coords[neighbors] - coords[i]
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors[norms > 0] / norms[norms > 0, None]
        out.at[i, "neighborhood_angular_asymmetry"] = (
            float(np.linalg.norm(np.mean(unit, axis=0))) if len(unit) else np.nan
        )
        out.at[i, "neighbor_size_log_disagreement"] = float(
            np.nanmedian(np.abs(np.log(np.maximum(area[neighbors], 1e-12) / max(area[i], 1e-12))))
        )
        out.at[i, "neighbor_shape_disagreement"] = float(
            np.nanmedian(np.sqrt((circ[neighbors] - circ[i]) ** 2 + (ecc[neighbors] - ecc[i]) ** 2))
        )
        out.at[i, "local_crowding_area_fraction_proxy"] = float(
            np.nansum(area[neighbors]) / (math.pi * radius * radius)
        )

        for values, mean_col, std_col in [
            (area, "nb_area_mean_px2", "nb_area_std_px2"),
            (circ, "nb_circularity_mean", "nb_circularity_std"),
            (ecc, "nb_eccentricity_mean", "nb_eccentricity_std"),
            (aspect, "nb_aspect_ratio_mean", "nb_aspect_ratio_std"),
            (dapi_mean, "nb_dapi_mean_intensity_mean", "nb_dapi_mean_intensity_std"),
        ]:
            vals = values[neighbors]
            out.at[i, mean_col] = float(np.nanmean(vals))
            out.at[i, std_col] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

    median_eqdiam = float(np.nanmedian(out["equivalent_diameter_px"]))
    radius = 2.5 * median_eqdiam
    out["fixed_neighbor_count"] = fixed_radius_neighbor_count(coords, radius)
    return out


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add reproducible, dimensionless DAPI morphology/spatial proxies.

    Size is normalized within an image, so image identifiers determine only the
    reference median and are never numerical model inputs.  No marker channel or
    experimental metadata is read here.
    """
    required = set(MODEL_FEATURES) | {
        "image_id", "local_crowding_area_fraction_proxy",
        "neighbor_size_log_disagreement", "neighbor_shape_disagreement",
        "neighborhood_angular_asymmetry",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing inputs for composite features: {missing}")
    out = df.copy()
    area = pd.to_numeric(out["area_px"], errors="coerce").clip(lower=1e-12)
    image_median_area = area.groupby(out["image_id"], sort=False).transform("median").clip(lower=1e-12)
    mean_i = pd.to_numeric(out["dapi_mean_intensity"], errors="coerce").abs().clip(lower=1e-12)
    out["nuclear_size_log_ratio"] = np.log(area / image_median_area)
    out["nuclear_elongation_log"] = np.log(pd.to_numeric(out["aspect_ratio"], errors="coerce").clip(lower=1.0))
    out["boundary_irregularity"] = (
        pd.to_numeric(out["perimeter_px"], errors="coerce") /
        (2.0 * np.sqrt(math.pi * area)) - 1.0
    )
    out["convexity_deficit"] = 1.0 - pd.to_numeric(out["solidity"], errors="coerce")
    out["chromatin_cv_proxy"] = pd.to_numeric(out["dapi_std_intensity"], errors="coerce") / mean_i
    out["chromatin_range_ratio"] = pd.to_numeric(out["dapi_intensity_range"], errors="coerce") / mean_i
    out["nearest_spacing_nuclear_units"] = (
        pd.to_numeric(out["nn1_distance_px"], errors="coerce") /
        pd.to_numeric(out["equivalent_diameter_px"], errors="coerce").clip(lower=1e-12)
    )
    validate_model_feature_names(COMPOSITE_FEATURES)
    return out


def extract_pair_features(pair: ImagePair, mask_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dapi = load_effective_channel(pair.dapi_path, "blue")
    oct4 = load_effective_channel(pair.oct4_path, "green")
    mask = np.load(mask_path)
    if dapi.shape != oct4.shape or dapi.shape != mask.shape:
        raise ValueError(f"Channel/mask shape mismatch for {pair.image_id}")

    height, width = dapi.shape
    feature_rows: list[dict] = []
    oct4_rows: list[dict] = []
    for prop in regionprops(mask, intensity_image=dapi):
        coords = prop.coords
        dapi_values = dapi[coords[:, 0], coords[:, 1]].astype(float)
        oct4_values = oct4[coords[:, 0], coords[:, 1]].astype(float)
        min_row, min_col, max_row, max_col = prop.bbox
        area = float(prop.area)
        perimeter = float(prop.perimeter)
        major = float(prop.major_axis_length)
        minor = float(prop.minor_axis_length)
        touches_border = bool(min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width)
        cell_id = f"{pair.image_id}__cell_{int(prop.label):05d}"
        metadata = {
            "cell_id": cell_id,
            "culture_day": pair.culture_day,
            "replicate": pair.replicate,
            "sample": pair.sample,
            "magnification": pair.magnification,
            "imaging_condition": f"magnification_{pair.magnification}",
            "image_id": pair.image_id,
            "source_snap_id": pair.source_snap_id,
            "label": int(prop.label),
            "centroid_row_px": float(prop.centroid[0]),
            "centroid_col_px": float(prop.centroid[1]),
            "dapi_image_path": str(pair.dapi_path),
            "oct4_image_path": str(pair.oct4_path),
            "merge_image_path": str(pair.merge_path) if pair.merge_path else "not_provided",
        }
        feature_rows.append(
            {
                **metadata,
                "touches_border": touches_border,
                "area_px": area,
                "perimeter_px": perimeter,
                "equivalent_diameter_px": float(prop.equivalent_diameter_area),
                "major_axis_length_px": major,
                "minor_axis_length_px": minor,
                "aspect_ratio": safe_div(major, minor),
                "eccentricity": float(prop.eccentricity),
                "circularity": circularity(area, perimeter),
                "solidity": float(prop.solidity),
                "extent": float(prop.extent),
                "dapi_mean_intensity": float(np.mean(dapi_values)),
                "dapi_std_intensity": float(np.std(dapi_values, ddof=1)) if len(dapi_values) > 1 else 0.0,
                "dapi_min_intensity": float(np.min(dapi_values)),
                "dapi_max_intensity": float(np.max(dapi_values)),
                "dapi_intensity_range": float(np.max(dapi_values) - np.min(dapi_values)),
            }
        )
        oct4_rows.append(
            {
                "cell_id": cell_id,
                "image_id": pair.image_id,
                "culture_day": pair.culture_day,
                "magnification": pair.magnification,
                "label": int(prop.label),
                "oct4_mean_intensity_raw_png_green": float(np.mean(oct4_values)),
                "oct4_std_intensity_raw_png_green": float(np.std(oct4_values, ddof=1)) if len(oct4_values) > 1 else 0.0,
                "oct4_min_intensity_raw_png_green": float(np.min(oct4_values)),
                "oct4_max_intensity_raw_png_green": float(np.max(oct4_values)),
                "oct4_image_path": str(pair.oct4_path),
            }
        )

    features = pd.DataFrame(feature_rows)
    oct4_internal = pd.DataFrame(oct4_rows)
    if len(features) == 0:
        return features, oct4_internal
    features = add_spatial_features(features)
    features = add_composite_features(features)
    features["qc_keep"] = (~features["touches_border"]).astype(bool)
    features["qc_reason"] = np.where(features["touches_border"], "touches_border", "keep")
    return features, oct4_internal


def extract_all_features(
    pairs: list[ImagePair], mask_paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_tables = []
    oct4_tables = []
    for pair in pairs:
        features, oct4_internal = extract_pair_features(pair, mask_paths[pair.image_id])
        feature_tables.append(features)
        oct4_tables.append(oct4_internal)
        print(
            f"[features] {pair.image_id}: total={len(features)}, "
            f"qc_keep={int(features['qc_keep'].sum()) if len(features) else 0}"
        )
    return pd.concat(feature_tables, ignore_index=True), pd.concat(oct4_tables, ignore_index=True)


def validate_model_feature_names(feature_columns: Iterable[str]) -> None:
    violations = [col for col in feature_columns if any(token in col.lower() for token in FORBIDDEN_MODEL_TOKENS)]
    if violations:
        raise AssertionError(f"OCT4 leakage detected in Model A feature columns: {violations}")


def preprocess_features(df: pd.DataFrame, feature_set: str = "augmented") -> tuple[np.ndarray, list[str], dict]:
    requested = MODEL_FEATURES if feature_set == "raw" else MODEL_FEATURES + COMPOSITE_MODEL_FEATURES
    if feature_set not in {"raw", "augmented"}:
        raise ValueError("feature_set must be 'raw' or 'augmented'")
    missing = [col for col in requested if col not in df.columns]
    if missing:
        raise KeyError(f"Missing Model A feature columns: {missing}")
    validate_model_feature_names(requested)

    numeric = df[requested].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    all_missing = [col for col in numeric.columns if numeric[col].notna().sum() == 0]
    numeric = numeric.drop(columns=all_missing)
    zero_variance = [
        col for col in numeric.columns if numeric[col].dropna().nunique() <= 1
    ]
    numeric = numeric.drop(columns=zero_variance)
    feature_columns = numeric.columns.tolist()
    validate_model_feature_names(feature_columns)
    if len(feature_columns) < 2:
        raise RuntimeError("Fewer than two usable Model A features remain after validation.")

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(imputed)
    if not np.isfinite(scaled).all():
        raise AssertionError("Non-finite values remain after Model A preprocessing.")
    info = {
        "all_missing_features_removed": all_missing,
        "zero_variance_features_removed": zero_variance,
        "imputer": "SimpleImputer(strategy='median')",
        "scaler": "RobustScaler",
        "feature_selection_rule": "explicit existing feature families; no correlation cutoff",
        "feature_set": feature_set,
    }
    return scaled, feature_columns, info


def fragmentation_penalty(cluster_counts: np.ndarray) -> dict:
    counts = np.asarray(cluster_counts, dtype=float)
    fractions = counts / counts.sum()
    small_deficit = np.clip(SMALL_CLUSTER_FRACTION - fractions, 0.0, None)
    tiny_deficit = np.clip(TINY_CLUSTER_FRACTION - fractions, 0.0, None)
    excess = EXCESS_CLUSTER_PENALTY * max(0, len(counts) - PREFERRED_MAX_CLUSTERS)
    small = SMALL_CLUSTER_DEFICIT_WEIGHT * float(small_deficit.sum())
    tiny = TINY_CLUSTER_DEFICIT_WEIGHT * float(tiny_deficit.sum())
    return {
        "cluster_size_min_fraction": float(fractions.min()),
        "small_cluster_count": int((fractions < SMALL_CLUSTER_FRACTION).sum()),
        "tiny_cluster_count": int((fractions < TINY_CLUSTER_FRACTION).sum()),
        "fragmentation_penalty": float(excess + small + tiny),
    }


def fit_gmm_with_dynamic_k(
    scaled: np.ndarray, random_state: int = RANDOM_STATE,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray, PCA, pd.DataFrame, int]:
    n_components = min(PCA_COMPONENTS, scaled.shape[1], scaled.shape[0] - 1)
    if n_components < 2:
        raise RuntimeError("Not enough cells/features for PCA and GMM.")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(scaled)

    max_k = min(K_MAX, len(pca_values) - 1)
    candidate_ks = list(range(K_MIN, max_k + 1))
    if not candidate_ks:
        raise RuntimeError("Not enough QC-kept cells for dynamic K search.")

    rows = []
    models: dict[int, GaussianMixture] = {}
    for k in candidate_ks:
        model = GaussianMixture(
            n_components=k,
            covariance_type=GMM_COVARIANCE_TYPE,
            random_state=random_state,
            n_init=GMM_N_INIT,
            reg_covar=1e-6,
        )
        labels = model.fit_predict(pca_values)
        probabilities = model.predict_proba(pca_values)
        counts = np.bincount(labels, minlength=k)
        penalty = fragmentation_penalty(counts)
        silhouette = (
            float(silhouette_score(pca_values, labels))
            if len(np.unique(labels)) > 1
            else np.nan
        )
        bic = float(model.bic(pca_values))
        aic = float(model.aic(pca_values))
        rows.append(
            {
                "n_clusters": k,
                "bic": bic,
                "aic": aic,
                "silhouette": silhouette,
                "fit_log_likelihood_mean": float(model.score(pca_values)),
                "mean_max_posterior": float(probabilities.max(axis=1).mean()),
                "fraction_max_posterior_below_0_8": float((probabilities.max(axis=1) < 0.8).mean()),
                **penalty,
                "bic_penalized": bic + penalty["fragmentation_penalty"],
            }
        )
        models[k] = model

    selection = pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)
    best_row = selection.sort_values(
        ["bic_penalized", "bic", "n_clusters"], ascending=[True, True, True]
    ).iloc[0]
    best_k = int(best_row["n_clusters"])
    best_model = models[best_k]
    raw_labels = best_model.predict(pca_values)
    raw_probabilities = best_model.predict_proba(pca_values)
    return best_model, raw_labels, raw_probabilities, pca, selection, best_k


def compare_feature_models(df: pd.DataFrame) -> tuple[dict, dict]:
    """Compare raw and augmented models without using post-hoc markers/metadata."""
    reports, artifacts = {}, {}
    for feature_set in ("raw", "augmented"):
        scaled, columns, prep = preprocess_features(df, feature_set=feature_set)
        model, labels, probs, pca, selection, selected_k = fit_gmm_with_dynamic_k(scaled)
        seed_aris = []
        for seed in (7, 19, 73):
            _, alternate, _, _, _, _ = fit_gmm_with_dynamic_k(scaled, random_state=seed)
            seed_aris.append(float(adjusted_rand_score(labels, alternate)))
        reports[feature_set] = {
            "n_features": len(columns), "selected_k": selected_k,
            "selected_k_at_search_upper_bound": selected_k == int(selection["n_clusters"].max()),
            "mean_max_posterior": float(probs.max(axis=1).mean()),
            "fraction_max_posterior_below_0_8": float((probs.max(axis=1) < 0.8).mean()),
            "seed_stability_ari_mean": float(np.mean(seed_aris)),
            "seed_stability_ari_values": seed_aris,
            "selected_bic": float(selection.loc[selection.n_clusters.eq(selected_k), "bic"].iloc[0]),
            "preprocessing": prep,
        }
        artifacts[feature_set] = (scaled, columns, model, labels, probs, pca, selection, selected_k)
    corr = df[MODEL_FEATURES + COMPOSITE_FEATURES].corr(method="spearman").abs()
    reports["augmented"]["composite_max_abs_spearman_with_raw"] = {
        col: float(corr.loc[col, MODEL_FEATURES].max()) for col in COMPOSITE_FEATURES
    }
    delta = reports["augmented"]["mean_max_posterior"] - reports["raw"]["mean_max_posterior"]
    reports["conclusion"] = {
        "posterior_confidence_delta_augmented_minus_raw": float(delta),
        "confidence_improved": bool(delta > 0),
        "note": "Composite features are retained for interpretability; improvement is not assumed.",
    }
    return reports, artifacts


def reorder_phenotypes_by_size(
    raw_labels: np.ndarray, raw_probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    counts = pd.Series(raw_labels).value_counts().sort_values(ascending=False)
    ordered_raw_ids = [int(value) for value in counts.index]
    raw_to_rank = {raw_id: rank for rank, raw_id in enumerate(ordered_raw_ids)}
    ranks = np.asarray([raw_to_rank[int(label)] for label in raw_labels], dtype=int)
    labels = np.asarray([f"Phenotype {rank + 1}" for rank in ranks], dtype=object)
    probabilities = raw_probabilities[:, ordered_raw_ids]
    return ranks, labels, ordered_raw_ids, probabilities


def _segment_sse(prefix_sum: np.ndarray, prefix_sq: np.ndarray, start: int, end: int) -> float:
    count = end - start + 1
    total = prefix_sum[end + 1] - prefix_sum[start]
    total_sq = prefix_sq[end + 1] - prefix_sq[start]
    return max(float(total_sq - (total * total) / count), 0.0)


def _segment_mean(prefix_sum: np.ndarray, start: int, end: int) -> float:
    return float((prefix_sum[end + 1] - prefix_sum[start]) / (end - start + 1))


def basc_a_threshold(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Port the threshold core of CRAN Binarize 1.3.1 BASC A.

    This follows binarizeBASCA.c/common.c: sorted measurements, optimal
    step functions for every scale, strongest discontinuity at each scale,
    and the median discontinuity index as the final threshold. The bootstrap
    p-value is not needed for status assignment and is intentionally omitted.
    """
    original = np.asarray(values, dtype=float)
    if original.ndim != 1:
        raise ValueError("BASC input must be one-dimensional.")
    if len(original) < 3:
        raise ValueError("BASC input must contain at least three values.")
    if not np.isfinite(original).all():
        raise ValueError("BASC input contains missing or infinite values.")
    if np.unique(original).size == 1:
        raise ValueError("BASC input is constant.")

    x = np.sort(original)
    n = len(x)
    prefix_sum = np.concatenate([[0.0], np.cumsum(x)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(x * x)])

    cc = np.zeros((n - 1, n - 1), dtype=float)
    for start in range(n - 1):
        cc[0, start] = _segment_sse(prefix_sum, prefix_sq, start, n - 1)

    ind = np.zeros((n - 2, n - 1), dtype=int)
    for j in range(n - 2):
        for start in range(n - j - 1):
            best_cost = np.inf
            best_boundary = -1
            for end in range(start, n - j - 1):
                cost = _segment_sse(prefix_sum, prefix_sq, start, end)
                if end + 1 < n - 1:
                    cost += cc[j, end + 1]
                if cost < best_cost:
                    best_cost = cost
                    best_boundary = end + 1
            cc[j + 1, start] = best_cost
            ind[j, start] = best_boundary

    p_matrix = np.zeros((n - 2, n - 2), dtype=int)
    for j in range(n - 2):
        z = j
        p_matrix[j, 0] = ind[z, 0]
        z -= 1
        for i in range(1, j + 1):
            p_matrix[j, i] = ind[z, p_matrix[j, i - 1]]
            z -= 1

    strongest_boundaries = np.zeros(n - 2, dtype=int)
    total_sum = prefix_sum[-1]
    total_sq = prefix_sq[-1]
    for j in range(n - 2):
        best_score = -np.inf
        best_boundary = -1
        for i in range(j + 1):
            boundary = int(p_matrix[j, i])
            left_start = 0 if i == 0 else int(p_matrix[j, i - 1])
            left_end = boundary - 1
            right_start = boundary
            right_end = n - 1 if i == j else int(p_matrix[j, i + 1]) - 1
            jump = _segment_mean(prefix_sum, right_start, right_end) - _segment_mean(
                prefix_sum, left_start, left_end
            )
            midpoint = 0.5 * (x[boundary - 1] + x[boundary])
            approximation_error = total_sq - 2.0 * midpoint * total_sum + n * midpoint * midpoint
            score = jump / max(float(approximation_error), np.finfo(float).tiny)
            if score > best_score:
                best_score = score
                best_boundary = boundary
        strongest_boundaries[j] = best_boundary

    median_boundary = float(np.median(strongest_boundaries))
    threshold_index = int(math.floor(median_boundary))
    threshold = 0.5 * (x[threshold_index] + x[threshold_index - 1])
    binary = (original > threshold).astype(np.int8)
    return float(threshold), binary


def compute_umap(scaled: np.ndarray) -> np.ndarray:
    started = time.perf_counter()
    if len(scaled) >= 4096:
        raise RuntimeError(
            "The lightweight single-image UMAP path is limited to fewer than 4096 cells; "
            "use the full pynndescent-backed import for a larger multi-image run."
        )
    import numba

    # This is a tiny single-image trial. Running the ordinary UMAP kernels as
    # Python functions is much faster here than this Windows environment's
    # first-time JIT compilation, and does not change the UMAP objective.
    numba.config.DISABLE_JIT = True
    # umap.__init__ eagerly imports optional ParametricUMAP, AlignedUMAP,
    # TensorFlow/Keras and Torch helpers. Load the ordinary UMAP implementation
    # directly so this non-parametric analysis does not pay that unrelated
    # import cost. The installed umap-learn source itself is left untouched.
    package_dir = Path(importlib.metadata.distribution("umap-learn").locate_file("umap"))
    package = types.ModuleType("umap")
    package.__path__ = [str(package_dir)]
    package.__package__ = "umap"
    sys.modules["umap"] = package

    # For fewer than 4096 observations umap-learn uses its exact small-data
    # path and never calls NNDescent. Stub only the eagerly imported symbols so
    # pynndescent's expensive import-time compilation is not triggered.
    pynndescent_package = types.ModuleType("pynndescent")
    pynndescent_distances = types.ModuleType("pynndescent.distances")
    pynndescent_sparse = types.ModuleType("pynndescent.sparse")

    class _UnusedNNDescent:
        pass

    pynndescent_package.NNDescent = _UnusedNNDescent
    pynndescent_distances.named_distances = {}
    pynndescent_sparse.sparse_named_distances = {}
    sys.modules["pynndescent"] = pynndescent_package
    sys.modules["pynndescent.distances"] = pynndescent_distances
    sys.modules["pynndescent.sparse"] = pynndescent_sparse

    spec = importlib.util.spec_from_file_location("umap.umap_", package_dir / "umap_.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the installed umap.umap_ module.")
    umap_module = importlib.util.module_from_spec(spec)
    sys.modules["umap.umap_"] = umap_module
    spec.loader.exec_module(umap_module)
    UMAP = umap_module.UMAP

    print(f"[umap +{time.perf_counter() - started:7.1f}s] import complete", flush=True)

    n_neighbors = min(30, max(2, len(scaled) - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=RANDOM_STATE,
        init="random",
        n_epochs=UMAP_N_EPOCHS,
    )
    print(f"[umap +{time.perf_counter() - started:7.1f}s] fitting embedding", flush=True)
    embedding = reducer.fit_transform(scaled)
    print(f"[umap +{time.perf_counter() - started:7.1f}s] fit complete", flush=True)
    return embedding


def save_selection_plot(
    selection: pd.DataFrame,
    selected_k: int,
    trial_label: str,
    magnification: str,
    path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    ax1.plot(selection["n_clusters"], selection["bic"], marker="o", label="BIC")
    ax1.plot(
        selection["n_clusters"],
        selection["bic_penalized"],
        marker="D",
        linestyle="--",
        label="Penalized BIC (selection)",
    )
    ax1.axvline(selected_k, color="black", linestyle=":", label=f"Selected K={selected_k}")
    ax1.set_xlabel("Number of GMM components (K)")
    ax1.set_ylabel("Information criterion (lower is better)")
    ax2 = ax1.twinx()
    ax2.plot(
        selection["n_clusters"],
        selection["silhouette"],
        color="purple",
        marker="s",
        alpha=0.75,
        label="Silhouette",
    )
    ax2.set_ylabel("Silhouette (higher is better)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"{trial_label} {magnification} GMM model selection")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def phenotype_colors(labels: Iterable[str]) -> dict[str, tuple]:
    unique = sorted(set(str(label) for label in labels), key=lambda x: int(x.split()[-1]))
    cmap = plt.get_cmap("tab20")
    # The color is a pure function of the global phenotype rank.  A phenotype
    # therefore keeps its color even when an individual image lacks lower ranks.
    return {label: cmap((int(label.split()[-1]) - 1) % cmap.N) for label in unique}


def save_phenotype_overlay_on_merge(
    pair: ImagePair,
    mask_path: Path,
    df: pd.DataFrame,
    path: Path,
    low_confidence_threshold: float = 0.80,
) -> None:
    """Overlay dominant GMM phenotypes on the colored microscopy image.

    The Merge image is display-only: its OCT4 channel is never converted into
    a model feature. Cells excluded by QC retain an unfilled gray boundary.
    """
    mask = np.load(mask_path)
    if pair.merge_path is not None and pair.merge_path.exists():
        base = np.asarray(Image.open(pair.merge_path).convert("RGB"), dtype=np.uint8)
        background_label = "original Merge image"
    else:
        dapi = normalize_to_uint8(load_effective_channel(pair.dapi_path, "blue"))
        oct4 = normalize_to_uint8(load_effective_channel(pair.oct4_path, "green"))
        base = np.stack([np.zeros_like(dapi), oct4, dapi], axis=-1)
        background_label = "reconstructed DAPI/OCT4 display composite"
    if base.shape[:2] != mask.shape:
        raise ValueError(
            f"Merge/mask shape mismatch for {pair.image_id}: {base.shape[:2]} vs {mask.shape}"
        )

    image_df = df.loc[df["image_id"].eq(pair.image_id)].copy()
    colors = phenotype_colors(image_df["dominant_phenotype"])
    label_to_rank = np.zeros(int(mask.max()) + 1, dtype=np.int16)
    rank_to_name: dict[int, str] = {}
    for row in image_df.itertuples(index=False):
        cell_label = int(row.label)
        rank = int(row.dominant_phenotype_rank)
        if 0 < cell_label < len(label_to_rank):
            label_to_rank[cell_label] = rank
            rank_to_name[rank] = str(row.dominant_phenotype)

    phenotype_map = label_to_rank[mask]
    overlay = base.astype(np.float32)
    alpha = 0.42
    for rank, phenotype in sorted(rank_to_name.items()):
        region = phenotype_map == rank
        color = np.asarray(colors[phenotype][:3], dtype=float) * 255.0
        overlay[region] = (1.0 - alpha) * overlay[region] + alpha * color

    inner_boundaries = find_boundaries(mask, mode="inner")
    for rank, phenotype in sorted(rank_to_name.items()):
        boundary = inner_boundaries & (phenotype_map == rank)
        overlay[boundary] = np.asarray(colors[phenotype][:3], dtype=float) * 255.0
    excluded_boundary = inner_boundaries & (mask > 0) & (phenotype_map == 0)
    overlay[excluded_boundary] = np.array([220.0, 220.0, 220.0])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    ax.imshow(overlay)
    low_confidence = image_df.loc[
        image_df["gmm_max_posterior"] < low_confidence_threshold
    ]
    if len(low_confidence):
        ax.scatter(
            low_confidence["centroid_col_px"],
            low_confidence["centroid_row_px"],
            marker="x",
            s=24,
            linewidths=1.1,
            color="white",
        )

    handles = []
    for rank, phenotype in sorted(rank_to_name.items()):
        count = int((image_df["dominant_phenotype_rank"] == rank).sum())
        handles.append(
            Patch(
                facecolor=colors[phenotype],
                edgecolor=colors[phenotype],
                label=f"{phenotype} (n={count})",
            )
        )
    if len(low_confidence):
        handles.append(
            Line2D(
                [],
                [],
                color="white",
                marker="x",
                linestyle="None",
                markeredgewidth=1.2,
                label=f"GMM max posterior < {low_confidence_threshold:.2f} (n={len(low_confidence)})",
            )
        )
    if np.any(excluded_boundary):
        handles.append(
            Patch(
                facecolor="none",
                edgecolor="#dcdcdc",
                label="QC excluded (no phenotype assigned)",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=1.0,
    )
    ax.set_title(
        f"{display_trial_label(pair.culture_day, pair.sample)} {pair.magnification}: "
        f"dominant GMM phenotype on {background_label}\n"
        "Phenotype colors use DAPI/DNA features only; Merge/OCT4 is display-only"
    )
    ax.axis("off")
    fig.subplots_adjust(right=0.74, top=0.90)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_umap_phenotypes(df: pd.DataFrame, path: Path) -> None:
    colors = phenotype_colors(df["dominant_phenotype"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for label, sub in df.groupby("dominant_phenotype"):
        ax.scatter(sub["umap_1"], sub["umap_2"], s=18, alpha=0.75, color=colors[label], label=label)
    magnification = str(df["magnification"].iloc[0])
    trial_label = display_trial_label(int(df["culture_day"].iloc[0]), str(df["sample"].iloc[0]))
    ax.set_title(f"{trial_label} {magnification}: dominant GMM phenotype\nDAPI/DNA features only; OCT4 was not used for UMAP or GMM")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_umap_memberships(df: pd.DataFrame, probability_columns: list[str], path: Path) -> None:
    n = len(probability_columns)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows), squeeze=False)
    membership_blue = (31 / 255, 119 / 255, 180 / 255)
    probability_cmap = LinearSegmentedColormap.from_list(
        "membership_probability_blue",
        [(*membership_blue, 0.06), (*membership_blue, 1.0)],
    )
    probability_norm = Normalize(vmin=0.0, vmax=1.0)
    for index, col in enumerate(probability_columns):
        ax = axes[index // ncols][index % ncols]
        sc = ax.scatter(
            df["umap_1"],
            df["umap_2"],
            c=df[col],
            s=18,
            cmap=probability_cmap,
            norm=probability_norm,
            edgecolors="none",
        )
        ax.set_title(col.replace("P_phenotype_", "P(Phenotype ") + ")")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(sc, ax=ax, label="GMM posterior probability")
    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    magnification = str(df["magnification"].iloc[0])
    trial_label = display_trial_label(int(df["culture_day"].iloc[0]), str(df["sample"].iloc[0]))
    fig.suptitle(
        f"{trial_label} {magnification} phenotype memberships\n"
        "Single hue: higher posterior probability is less transparent; OCT4 excluded from UMAP/GMM",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_umap_oct4_binary(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    colors = {0: "#8c8c8c", 1: "#22a447"}
    names = {0: "BASC OCT4-negative", 1: "BASC OCT4-positive"}
    for status, sub in df.groupby("BASC_OCT4_status"):
        ax.scatter(sub["umap_1"], sub["umap_2"], s=18, alpha=0.75, color=colors[int(status)], label=names[int(status)])
    magnification = str(df["magnification"].iloc[0])
    trial_label = display_trial_label(int(df["culture_day"].iloc[0]), str(df["sample"].iloc[0]))
    ax.set_title(f"{trial_label} {magnification}: BASC-defined binary OCT4 status\nOCT4 was not used for UMAP or GMM calculations")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_basc_threshold_diagnostic(
    values: np.ndarray,
    threshold: float,
    trial_label: str,
    magnification: str,
    path: Path,
) -> None:
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(values, bins="auto", color="#7f8c8d", edgecolor="white", alpha=0.9)
    ax.axvline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"BASC threshold = {threshold:.3f}",
    )
    ax.set_xlabel("Raw OCT4 mean intensity (PNG green channel; thresholding only)")
    ax.set_ylabel("Cell count")
    ax.set_title(
        f"{trial_label} {magnification}: BASC threshold diagnostic\n"
        "Raw OCT4 shown only to audit binarization; not used by UMAP or GMM"
    )
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def characterize_phenotypes(
    result_df: pd.DataFrame, feature_columns: list[str], output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for phenotype, sub in result_df.groupby("dominant_phenotype"):
        row = {
            "dominant_phenotype": phenotype,
            "n_cells": int(len(sub)),
            "BASC_OCT4_positive_fraction": float(sub["BASC_OCT4_status"].mean()),
            "mean_max_gmm_posterior": float(sub["gmm_max_posterior"].mean()),
        }
        for col in feature_columns:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}__mean"] = float(values.mean())
            row[f"{col}__median"] = float(values.median())
            row[f"{col}__std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("dominant_phenotype")

    overall_mean = result_df[feature_columns].mean(numeric_only=True)
    overall_std = result_df[feature_columns].std(numeric_only=True, ddof=0).replace(0, np.nan)
    heatmap = (
        result_df.groupby("dominant_phenotype")[feature_columns].mean() - overall_mean
    ) / overall_std
    heatmap = heatmap.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(12, len(feature_columns) * 0.42), max(3.5, len(heatmap) * 0.7)))
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(heatmap.index)), labels=heatmap.index)
    ax.set_xticks(range(len(feature_columns)), labels=feature_columns, rotation=75, ha="right")
    ax.set_title("Phenotype feature characterization (standardized mean differences)")
    fig.colorbar(image, ax=ax, label="Mean difference / overall SD")
    fig.tight_layout()
    fig.savefig(output_root / "figures" / "phenotype_feature_heatmap.png", dpi=220)
    plt.close(fig)

    oct4_summary = (
        result_df.groupby("dominant_phenotype", as_index=False)
        .agg(n_cells=("cell_id", "size"), BASC_OCT4_positive_fraction=("BASC_OCT4_status", "mean"))
        .sort_values("dominant_phenotype")
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(oct4_summary["dominant_phenotype"], oct4_summary["BASC_OCT4_positive_fraction"], color="#22a447")
    ax.set_ylim(0, 1)
    ax.set_ylabel("BASC OCT4-positive fraction")
    ax.set_title("Post-hoc OCT4 characterization by phenotype\nOCT4 did not affect GMM fitting")
    fig.tight_layout()
    fig.savefig(output_root / "figures" / "basc_oct4_fraction_by_phenotype.png", dpi=220)
    plt.close(fig)
    return summary, oct4_summary


def run_trial(args: argparse.Namespace) -> None:
    trial_started = time.perf_counter()

    def stage(message: str) -> None:
        elapsed = time.perf_counter() - trial_started
        print(f"[stage +{elapsed:7.1f}s] {message}", flush=True)

    output_root = args.output_root.resolve()
    tables_dir = output_root / "tables"
    internal_dir = output_root / "internal"
    figures_dir = output_root / "figures"
    for directory in [tables_dir, internal_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    trial_label = display_trial_label(args.culture_day, args.sample)
    file_prefix = trial_file_prefix(args.culture_day, args.sample)
    pairs, pairing_report = discover_image_pairs(
        args.data_root,
        culture_day=args.culture_day,
        sample=args.sample,
        replicate=args.replicate,
    )
    pairing_report.to_csv(
        tables_dir / f"{file_prefix}_image_manifest.csv", index=False, encoding="utf-8-sig"
    )
    trial_pairs = [
        pair
        for pair in pairs
        if pair.magnification.lower() == args.fit_magnification.lower()
    ]
    if len(trial_pairs) != 1:
        raise RuntimeError(
            f"Expected exactly one {trial_label} {args.fit_magnification} pair, "
            f"got {len(trial_pairs)}."
        )
    mask_paths = segment_pairs(
        trial_pairs,
        output_root=output_root,
        use_gpu=not args.cpu,
        reuse_masks=args.reuse_masks,
    )
    stage("segmentation complete")
    all_features, all_oct4_internal = extract_all_features(trial_pairs, mask_paths)
    all_features.to_csv(
        tables_dir / f"{file_prefix}_trial_dapi_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qc_summary = (
        all_features.groupby(["culture_day", "magnification", "image_id"], as_index=False)
        .agg(n_segmented=("cell_id", "size"), n_qc_keep=("qc_keep", "sum"))
    )
    qc_summary["qc_keep_fraction"] = qc_summary["n_qc_keep"] / qc_summary["n_segmented"]
    qc_summary.to_csv(
        tables_dir / f"{file_prefix}_qc_summary.csv", index=False, encoding="utf-8-sig"
    )

    fit_df = all_features.loc[
        all_features["magnification"].astype(str).str.lower().eq(args.fit_magnification.lower())
        & all_features["qc_keep"]
    ].reset_index(drop=True)
    if len(fit_df) < 20:
        raise RuntimeError(
            f"Only {len(fit_df)} QC-kept cells are available for {args.fit_magnification}; trial requires at least 20."
        )

    # Model A preprocessing and GMM occur before any OCT4 table is merged.
    stage(f"preprocessing {len(fit_df)} QC-kept cells using DAPI/DNA features only")
    comparison, comparison_artifacts = compare_feature_models(fit_df)
    scaled, feature_columns, model, raw_labels, raw_probabilities, pca, selection, selected_k = comparison_artifacts["augmented"]
    preprocess_info = comparison["augmented"]["preprocessing"]
    pd.DataFrame([
        {"feature_set": name, **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}}
        for name, report in comparison.items() if name in {"raw", "augmented"}
    ]).to_csv(tables_dir / "raw_vs_composite_model_comparison.csv", index=False, encoding="utf-8-sig")
    save_json(comparison, tables_dir / "raw_vs_composite_model_comparison.json")
    stage("starting dynamic GMM search")
    stage(f"dynamic GMM search complete; selected K={selected_k}")
    ranks, phenotype_labels, ordered_raw_ids, probabilities = reorder_phenotypes_by_size(
        raw_labels, raw_probabilities
    )
    if probabilities.shape[1] != selected_k:
        raise AssertionError("Posterior probability column count does not equal selected K.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("GMM posterior probabilities do not sum to one.")

    stage("starting UMAP")
    umap_values = compute_umap(scaled)
    stage("UMAP complete")
    result_df = fit_df.copy()
    result_df["pca_1"] = pca.transform(scaled)[:, 0]
    result_df["pca_2"] = pca.transform(scaled)[:, 1]
    result_df["umap_1"] = umap_values[:, 0]
    result_df["umap_2"] = umap_values[:, 1]
    result_df["gmm_component_raw"] = raw_labels
    result_df["dominant_phenotype_rank"] = ranks + 1
    result_df["dominant_phenotype"] = phenotype_labels
    result_df["gmm_max_posterior"] = probabilities.max(axis=1)
    probability_columns = []
    for index in range(selected_k):
        col = f"P_phenotype_{index + 1}"
        result_df[col] = probabilities[:, index]
        probability_columns.append(col)

    # Only now use raw OCT4 intensities for BASC and binary post-hoc characterization.
    oct4_fit = all_oct4_internal.loc[
        all_oct4_internal["cell_id"].isin(result_df["cell_id"])
    ].copy()
    oct4_fit = oct4_fit.set_index("cell_id").loc[result_df["cell_id"]].reset_index()
    if not np.array_equal(oct4_fit["cell_id"].to_numpy(), result_df["cell_id"].to_numpy()):
        raise AssertionError("OCT4/internal and Model A cell order do not align.")
    oct4_col = "oct4_mean_intensity_raw_png_green"
    stage("starting BASC-A threshold calculation on internal OCT4 values")
    basc_threshold, basc_binary = basc_a_threshold(oct4_fit[oct4_col].to_numpy(dtype=float))
    stage(f"BASC-A complete; threshold={basc_threshold:.6g}")
    oct4_fit["BASC_threshold"] = basc_threshold
    oct4_fit["BASC_OCT4_status"] = basc_binary
    oct4_fit.to_csv(internal_dir / "oct4_basc_input_internal.csv", index=False, encoding="utf-8-sig")
    result_df["BASC_OCT4_status"] = basc_binary
    save_basc_threshold_diagnostic(
        oct4_fit[oct4_col].to_numpy(dtype=float),
        basc_threshold,
        trial_label,
        args.fit_magnification,
        figures_dir / "basc_threshold_diagnostic.png",
    )

    forbidden_final = [col for col in result_df.columns if "oct4" in col.lower() and col != "BASC_OCT4_status" and col != "oct4_image_path"]
    if forbidden_final:
        raise AssertionError(f"Continuous OCT4 columns leaked into the final table: {forbidden_final}")

    selection.to_csv(tables_dir / "gmm_model_selection.csv", index=False, encoding="utf-8-sig")
    save_selection_plot(
        selection,
        selected_k,
        trial_label,
        args.fit_magnification,
        figures_dir / "gmm_model_selection.png",
    )
    save_phenotype_overlay_on_merge(
        trial_pairs[0],
        mask_paths[trial_pairs[0].image_id],
        result_df,
        figures_dir / "phenotype_overlay_on_merge.png",
    )
    save_umap_phenotypes(result_df, figures_dir / "umap_dominant_phenotype.png")
    save_umap_memberships(result_df, probability_columns, figures_dir / "umap_membership_probabilities.png")
    save_umap_oct4_binary(result_df, figures_dir / "umap_basc_oct4_binary.png")

    phenotype_summary, oct4_summary = characterize_phenotypes(result_df, feature_columns, output_root)
    phenotype_summary.to_csv(tables_dir / "phenotype_feature_summary.csv", index=False, encoding="utf-8-sig")
    oct4_summary.to_csv(tables_dir / "phenotype_basc_oct4_summary.csv", index=False, encoding="utf-8-sig")

    final_columns = META_COLUMNS + feature_columns + [
        "qc_keep",
        "pca_1",
        "pca_2",
        "umap_1",
        "umap_2",
        "gmm_component_raw",
        "dominant_phenotype_rank",
        "dominant_phenotype",
        "gmm_max_posterior",
        *probability_columns,
        "BASC_OCT4_status",
    ]
    final_columns = [col for col in final_columns if col in result_df.columns]
    final_df = result_df[final_columns].copy()
    final_df.to_csv(tables_dir / "model_a_single_cell_results.csv", index=False, encoding="utf-8-sig")
    stage("tables and figures written")

    verification = {
        "oct4_excluded_from_model_features": not any("oct4" in col.lower() or "af488" in col.lower() for col in feature_columns),
        "posterior_rows_sum_to_one": bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)),
        "posterior_column_count_equals_selected_k": len(probability_columns) == selected_k,
        "final_continuous_oct4_columns": [
            col for col in final_df.columns if "oct4" in col.lower() and col != "BASC_OCT4_status" and col != "oct4_image_path"
        ],
        "culture_day_values": sorted(final_df["culture_day"].unique().tolist()),
        "magnification_values_in_fit": sorted(final_df["magnification"].astype(str).unique().tolist()),
        "replicate_value": sorted(final_df["replicate"].astype(str).unique().tolist()),
    }
    if not all(
        [
            verification["oct4_excluded_from_model_features"],
            verification["posterior_rows_sum_to_one"],
            verification["posterior_column_count_equals_selected_k"],
            not verification["final_continuous_oct4_columns"],
        ]
    ):
        raise AssertionError(f"Model A verification failed: {verification}")

    run_info = {
        "trial_scope": f"{trial_label} only; {args.fit_magnification} is the only image segmented, characterized, and used for GMM/BASC",
        "available_dna_stain": "DAPI (not Hoechst)",
        "input_limitation": "8-bit pseudo-colored PNG exports; no biological micrometre-per-pixel calibration",
        "data_root": args.data_root,
        "output_root": output_root,
        "culture_day": args.culture_day,
        "sample": args.sample,
        "replicate": args.replicate,
        "fit_magnification": args.fit_magnification,
        "replicate_metadata": args.replicate,
        "n_image_pairs_discovered": len(pairs),
        "n_trial_image_pairs_used": len(trial_pairs),
        "n_fit_cells": len(final_df),
        "model_feature_columns": feature_columns,
        "preprocessing": preprocess_info,
        "gmm": {
            "candidate_k": selection["n_clusters"].astype(int).tolist(),
            "selected_k": selected_k,
            "selection_metric": "bic_penalized (existing repository method)",
            "covariance_type": GMM_COVARIANCE_TYPE,
            "n_init": GMM_N_INIT,
            "random_state": RANDOM_STATE,
            "ordered_raw_component_ids_by_size": ordered_raw_ids,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            "selected_k_at_search_upper_bound": selected_k == int(selection["n_clusters"].max()),
            "fraction_max_posterior_below_0_8": float((probabilities.max(axis=1) < 0.8).mean()),
        },
        "raw_vs_composite_model_comparison": comparison,
        "umap": {
            "n_neighbors": min(30, max(2, len(scaled) - 1)),
            "min_dist": 0.1,
            "init": "random",
            "n_epochs": UMAP_N_EPOCHS,
            "numba_jit": False,
            "small_data_exact_neighbors": True,
            "role": "visualization only; not used for GMM fitting or K selection",
        },
        "phenotype_image_overlay": {
            "file": figures_dir / "phenotype_overlay_on_merge.png",
            "background": str(trial_pairs[0].merge_path) if trial_pairs[0].merge_path else "reconstructed DAPI/OCT4 composite",
            "background_used_for_model": False,
            "color_encodes": "dominant GMM phenotype",
            "white_x_encodes": "gmm_max_posterior < 0.80",
        },
        "basc": {
            "implementation": BASC_IMPLEMENTATION,
            "source": BASC_SOURCE,
            "method": "BASC A",
            "diagnostic_file": figures_dir / "basc_threshold_diagnostic.png",
            "input_internal_column": oct4_col,
            "threshold": basc_threshold,
            "positive_rule": "raw OCT4 mean intensity > BASC threshold",
            "positive_count": int(basc_binary.sum()),
            "negative_count": int((basc_binary == 0).sum()),
            "used_for_gmm_or_umap": False,
        },
        "technical_and_cell_cycle_wording": "evaluated/described only; no technical or cell-cycle correction was applied",
        "verification": verification,
    }
    save_json(run_info, output_root / "run_info.json")
    print(json.dumps(json_ready(run_info), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        run_trial(parse_args())
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
