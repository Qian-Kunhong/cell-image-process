"""Reusable 2D nuclear morphology and neighborhood calculations."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from .qc import edge_qc_from_bbox


def safe_div(numerator: float, denominator: float | None) -> float:
    if denominator is None:
        return np.nan
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan


def circularity(area: float, perimeter: float | None) -> float:
    if perimeter is None:
        return np.nan
    return safe_div(4.0 * math.pi * area, perimeter * perimeter)


def extract_morphology_2d(
    mask: np.ndarray,
    intensity: np.ndarray,
    *,
    image_id: str,
    edge_buffer_px: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Extract the repository's common 2D DAPI feature definition."""
    mask = np.asarray(mask)
    intensity = np.asarray(intensity)
    if mask.ndim != 2 or intensity.ndim != 2 or mask.shape != intensity.shape:
        raise ValueError("mask and intensity must have identical 2D shapes")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("mask must contain integer labels")
    common_metadata = dict(metadata or {})
    rows: list[dict[str, object]] = []
    for prop in regionprops(mask, intensity_image=intensity):
        coords = prop.coords
        values = intensity[coords[:, 0], coords[:, 1]].astype(float)
        area = float(prop.area)
        perimeter = float(prop.perimeter)
        major = float(prop.major_axis_length)
        minor = float(prop.minor_axis_length)
        edge = edge_qc_from_bbox(prop.bbox, mask.shape, edge_buffer_px=edge_buffer_px)
        rows.append(
            {
                **common_metadata,
                "cell_id": f"{image_id}__cell_{int(prop.label):05d}",
                "image_id": image_id,
                "label": int(prop.label),
                "centroid_row_px": float(prop.centroid[0]),
                "centroid_col_px": float(prop.centroid[1]),
                "touches_border": edge.touches_border,
                "image_edge_gap_px": edge.image_edge_gap_px,
                "edge_buffer_px": edge.edge_buffer_px,
                "qc_keep": edge.keep,
                "qc_reason": edge.reason,
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
                "dapi_mean_intensity": float(np.mean(values)),
                "dapi_std_intensity": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "dapi_min_intensity": float(np.min(values)),
                "dapi_max_intensity": float(np.max(values)),
                "dapi_intensity_range": float(np.max(values) - np.min(values)),
            }
        )
    return pd.DataFrame(rows)


def fixed_radius_neighbor_count(coords: np.ndarray, radius: float) -> np.ndarray:
    if len(coords) == 0:
        return np.array([], dtype=int)
    tree = cKDTree(coords)
    return np.asarray(
        [len(tree.query_ball_point(coords[i], r=radius)) - 1 for i in range(len(coords))],
        dtype=int,
    )


def add_spatial_features(frame: pd.DataFrame, *, k_neighbors: int = 6) -> pd.DataFrame:
    """Add the historical adaptive-kNN fields without project metadata."""
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer")
    original_index = frame.index.copy()
    out = frame.reset_index(drop=True).copy()
    spatial_columns = [
        "nn1_distance_px", "knn6_distance_mean_px", "knn6_distance_std_px",
        "local_density_per_px2", "adaptive_neighbor_count", "nb_area_mean_px2",
        "nb_area_std_px2", "nb_circularity_mean", "nb_circularity_std",
        "nb_eccentricity_mean", "nb_eccentricity_std", "nb_aspect_ratio_mean",
        "nb_aspect_ratio_std", "nb_dapi_mean_intensity_mean",
        "nb_dapi_mean_intensity_std", "fixed_neighbor_count",
        "local_crowding_area_fraction_proxy", "neighbor_size_log_disagreement",
        "neighbor_shape_disagreement", "neighborhood_angular_asymmetry",
    ]
    for column in spatial_columns:
        out[column] = np.nan
    n = len(out)
    if n <= 1:
        out["adaptive_neighbor_count"] = 0
        out["fixed_neighbor_count"] = 0
        out.index = original_index
        return out

    coords = out[["centroid_row_px", "centroid_col_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    k_eff = min(k_neighbors, n - 1)
    distances, indices = tree.query(coords, k=k_eff + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    area = out["area_px"].to_numpy(dtype=float)
    circ = out["circularity"].to_numpy(dtype=float)
    ecc = out["eccentricity"].to_numpy(dtype=float)
    aspect = out["aspect_ratio"].to_numpy(dtype=float)
    dapi_mean = out["dapi_mean_intensity"].to_numpy(dtype=float)

    for i in range(n):
        distances_i = np.asarray(distances[i][1:], dtype=float)
        if len(distances_i) == 0:
            continue
        radius = float(distances_i[-1])
        neighbors = np.asarray(
            [j for j in tree.query_ball_point(coords[i], r=radius + 1e-8) if j != i],
            dtype=int,
        )
        out.at[i, "nn1_distance_px"] = float(distances_i[0])
        out.at[i, "knn6_distance_mean_px"] = float(np.mean(distances_i))
        out.at[i, "knn6_distance_std_px"] = float(np.std(distances_i, ddof=1)) if len(distances_i) > 1 else 0.0
        out.at[i, "adaptive_neighbor_count"] = int(len(neighbors))
        out.at[i, "local_density_per_px2"] = safe_div(len(neighbors), math.pi * radius * radius)
        if len(neighbors) == 0:
            continue
        vectors = coords[neighbors] - coords[i]
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors[norms > 0] / norms[norms > 0, None]
        out.at[i, "neighborhood_angular_asymmetry"] = float(np.linalg.norm(np.mean(unit, axis=0))) if len(unit) else np.nan
        out.at[i, "neighbor_size_log_disagreement"] = float(
            np.nanmedian(np.abs(np.log(np.maximum(area[neighbors], 1e-12) / max(area[i], 1e-12))))
        )
        out.at[i, "neighbor_shape_disagreement"] = float(
            np.nanmedian(np.sqrt((circ[neighbors] - circ[i]) ** 2 + (ecc[neighbors] - ecc[i]) ** 2))
        )
        out.at[i, "local_crowding_area_fraction_proxy"] = float(
            np.nansum(area[neighbors]) / (math.pi * radius * radius)
        )
        for values, mean_column, std_column in [
            (area, "nb_area_mean_px2", "nb_area_std_px2"),
            (circ, "nb_circularity_mean", "nb_circularity_std"),
            (ecc, "nb_eccentricity_mean", "nb_eccentricity_std"),
            (aspect, "nb_aspect_ratio_mean", "nb_aspect_ratio_std"),
            (dapi_mean, "nb_dapi_mean_intensity_mean", "nb_dapi_mean_intensity_std"),
        ]:
            values_i = values[neighbors]
            out.at[i, mean_column] = float(np.nanmean(values_i))
            out.at[i, std_column] = float(np.nanstd(values_i, ddof=1)) if len(values_i) > 1 else 0.0

    median_eqdiam = float(np.nanmedian(out["equivalent_diameter_px"]))
    out["fixed_neighbor_count"] = fixed_radius_neighbor_count(coords, 2.5 * median_eqdiam)
    out.index = original_index
    return out


def add_composite_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dimensionless morphology/spatial proxies used by Ekin adapters."""
    out = frame.copy()
    area = pd.to_numeric(out["area_px"], errors="coerce").clip(lower=1e-12)
    image_median_area = area.groupby(out["image_id"], sort=False).transform("median").clip(lower=1e-12)
    mean_intensity = pd.to_numeric(out["dapi_mean_intensity"], errors="coerce").abs().clip(lower=1e-12)
    out["nuclear_size_log_ratio"] = np.log(area / image_median_area)
    out["nuclear_elongation_log"] = np.log(pd.to_numeric(out["aspect_ratio"], errors="coerce").clip(lower=1.0))
    out["boundary_irregularity"] = pd.to_numeric(out["perimeter_px"], errors="coerce") / (2.0 * np.sqrt(math.pi * area)) - 1.0
    out["convexity_deficit"] = 1.0 - pd.to_numeric(out["solidity"], errors="coerce")
    out["chromatin_cv_proxy"] = pd.to_numeric(out["dapi_std_intensity"], errors="coerce") / mean_intensity
    out["chromatin_range_ratio"] = pd.to_numeric(out["dapi_intensity_range"], errors="coerce") / mean_intensity
    out["nearest_spacing_nuclear_units"] = pd.to_numeric(out["nn1_distance_px"], errors="coerce") / pd.to_numeric(out["equivalent_diameter_px"], errors="coerce").clip(lower=1e-12)
    return out

