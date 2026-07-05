from pathlib import Path
import json
import math
import os
import re
import shutil
import warnings
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning

from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)
from skimage.segmentation import find_boundaries

# =========================
# Optional UMAP
# =========================
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    warnings.warn("umap-learn is not installed. UMAP embedding will be skipped.")

# Keep logs clean during hyper-parameter search / UMAP fitting.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn\.gaussian_process\.kernels")
warnings.filterwarnings("ignore", category=UserWarning, module=r"umap\.umap_")


# =========================================================
# 04_unsupervised_learning.py
# ---------------------------------------------------------
# Neighbor-focused unsupervised clustering
# + inner-colony-only fitting for main model
# + single final label output
#
# Core logic / 核心逻辑:
# 1) edge metrics are computed first
# 2) main GMM is fit on inner-colony nuclei, then predicts all nuclei
# 3) global pseudo-labels are decided ONLY by main GMM
# 4) morphology model is used ONLY to veto suspicious edge-red
# 5) edge green is kept by default to avoid collapsing all red
#
# Final overlay colors / 最终颜色统一:
#   deviated         = red
#   undifferentiated = green
#   uncertain        = yellow
# =========================================================


# =========================
# Labeling mode / 标签模式
# =========================
# options: "conservative", "balanced", "aggressive"
LABELING_MODE = "conservative"

LABELING_THRESHOLDS = {
    "conservative": {
        "deviated_prob_min": 0.80,
        "undiff_prob_min": 0.65,
    },
    "balanced": {
        "deviated_prob_min": 0.82,
        "undiff_prob_min": 0.55,
    },
    "aggressive": {
        "deviated_prob_min": 0.75,
        "undiff_prob_min": 0.45,
    },
}

if LABELING_MODE not in LABELING_THRESHOLDS:
    raise ValueError(f"Invalid LABELING_MODE: {LABELING_MODE}")

DEVIATED_PROB_MIN = LABELING_THRESHOLDS[LABELING_MODE]["deviated_prob_min"]
UNDIFF_PROB_MIN = LABELING_THRESHOLDS[LABELING_MODE]["undiff_prob_min"]


# =========================
# Paths / 路径
# =========================
DEFAULT_SUZUI_ROOT = Path(r"F:\Suzui")
SUZUI_ROOT = Path(os.environ.get("SUZUI_ROOT", str(DEFAULT_SUZUI_ROOT)))
ANALYSIS_ROOT = SUZUI_ROOT / "analysis_out"
TRAINING_ROOT = SUZUI_ROOT / "training data"
TRAINING_SET_NAME = os.environ.get("TRAINING_SET_NAME", "SNL")

INPUT_CSV = Path(os.environ["INPUT_CSV"]) if "INPUT_CSV" in os.environ else (
    ANALYSIS_ROOT / "features_training" / TRAINING_SET_NAME / "nucleus_features.csv"
)
from scipy.optimize import differential_evolution
INTENSITY_CSV = Path(os.environ["INTENSITY_CSV"]) if "INTENSITY_CSV" in os.environ else (
    INPUT_CSV.parent / "nucleus_intensity_features.csv"
)

IMAGE_DIR = Path(os.environ["IMAGE_DIR"]) if "IMAGE_DIR" in os.environ else (TRAINING_ROOT / TRAINING_SET_NAME)
ORIG_MASK_DIR = Path(os.environ["ORIG_MASK_DIR"]) if "ORIG_MASK_DIR" in os.environ else (
    ANALYSIS_ROOT / "masks_training" / TRAINING_SET_NAME
)
QC_KEEP_MASK_DIR = Path(os.environ["QC_KEEP_MASK_DIR"]) if "QC_KEEP_MASK_DIR" in os.environ else ORIG_MASK_DIR

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"INPUT_CSV not found: {INPUT_CSV}\n"
        "Set env var INPUT_CSV to your local nucleus_features.csv path."
    )

OUT_DIR = INPUT_CSV.parent / f"cluster_neighbor_innerfit_{LABELING_MODE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MINIMAL_OUTPUT_MODE = True
KEEP_EXTENDED_OUTPUT_DIR = True
FINAL_CLUSTERED_CSV = INPUT_CSV
FINAL_CLUSTER_FIG = INPUT_CSV.parent / "cluster_umap.png"

OUT_CLUSTERED_CSV = OUT_DIR / "nucleus_features_qc_clustered.csv"
OUT_FEATURE_SUMMARY_RAW = OUT_DIR / "cluster_feature_summary_raw_gmm.csv"
OUT_FEATURE_SUMMARY = OUT_DIR / "cluster_feature_summary.csv"
OUT_REVIEW_SAMPLES = OUT_DIR / "cluster_review_samples.csv"
OUT_PCA_EMBEDDING = OUT_DIR / "pca_embedding.csv"
OUT_UMAP_EMBEDDING = OUT_DIR / "umap_embedding.csv"
OUT_UMAP2_EMBEDDING = OUT_DIR / "umap2d_embedding.csv"
OUT_UMAP3_EMBEDDING = OUT_DIR / "umap3d_embedding.csv"
OUT_FIG_PCA_RAW = OUT_DIR / "pca_clusters_raw_gmm.png"
OUT_FIG_PCA = OUT_DIR / "pca_clusters.png"
OUT_FIG_UMAP_RAW = OUT_DIR / "umap_clusters_raw_gmm.png"
OUT_FIG_UMAP = OUT_DIR / "umap_clusters.png"
OUT_FIG_UMAP2_RAW = OUT_DIR / "umap2d_clusters_raw_gmm.png"
OUT_FIG_UMAP2 = OUT_DIR / "umap2d_clusters.png"
OUT_FIG_UMAP3_RAW = OUT_DIR / "umap3d_clusters_raw_gmm.png"
OUT_FIG_UMAP3 = OUT_DIR / "umap3d_clusters.png"
OUT_RUN_INFO = OUT_DIR / "run_info.txt"
OUT_RUN_INFO_JSON = OUT_DIR / "run_info.json"
OUT_FEATURE_INFO = OUT_DIR / "selected_features.txt"
OUT_DEVIATION_INTENSITY_CSV = INPUT_CSV.parent / "deviated_score_vs_mean_intensity.csv"
OUT_DEVIATION_INTENSITY_FIG = INPUT_CSV.parent / "deviated_score_vs_mean_intensity.png"
OUT_DEVIATION_INTENSITY_BOXPLOT = INPUT_CSV.parent / "deviation_score_vs_oct4_intensity_boxplot.png"
COMPACT_OUTPUT_ONLY = True

REVIEW_OVERLAY_DIR = OUT_DIR / "review_overlay_tif"
REVIEW_CLUSTER_MASK_DIR = OUT_DIR / "review_cluster_masks"

for p in [
    REVIEW_OVERLAY_DIR,
    REVIEW_CLUSTER_MASK_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Main model features / 主模型特征
# =========================
FEATURE_SPECS = [
    {"candidates": ["nn1_distance_um", "nn1_distance"], "weight": 0.80, "log1p": True, "required": True},
    {"candidates": ["knn6_distance_mean_um", "knn6_distance_mean"], "weight": 0.85, "log1p": True, "required": True},
    {"candidates": ["knn6_distance_std_um", "knn6_distance_std"], "weight": 0.90, "log1p": True, "required": True},
    {"candidates": ["local_density_per_um2", "local_density"], "weight": 0.65, "log1p": True, "required": True},
    {"candidates": ["adaptive_nb_area_mean_um2", "nb_area_mean_um2", "adaptive_nb_area_mean", "nb_area_mean"], "weight": 0.95, "log1p": True, "required": True},
    {"candidates": ["adaptive_nb_circularity_mean", "nb_circularity_mean"], "weight": 1.25, "log1p": False, "required": True},
    {"candidates": ["adaptive_nb_eccentricity_mean", "nb_eccentricity_mean"], "weight": 1.25, "log1p": False, "required": True},
    {"candidates": ["adaptive_nb_aspect_ratio_mean", "nb_aspect_ratio_mean"], "weight": 1.25, "log1p": False, "required": True},
    {"candidates": ["fixed_neighbor_count"], "weight": 0.65, "log1p": True, "required": True},
    {"candidates": ["fixed_nb_area_mean_um2", "fixed_nb_area_mean"], "weight": 0.95, "log1p": True, "required": False},
    {"candidates": ["fixed_nb_circularity_mean"], "weight": 1.30, "log1p": False, "required": False},
    {"candidates": ["fixed_nb_eccentricity_mean"], "weight": 1.30, "log1p": False, "required": False},
    {"candidates": ["fixed_nb_aspect_ratio_mean"], "weight": 1.30, "log1p": False, "required": False},
    {"candidates": ["fixed_nb_distance_mean_um", "fixed_nb_distance_mean"], "weight": 0.85, "log1p": True, "required": False},
    # Upgraded morphology-only descriptors (DAPI geometry), optional by availability.
    {"candidates": ["solidity"], "weight": 1.10, "log1p": False, "required": False},
    {"candidates": ["extent"], "weight": 0.95, "log1p": False, "required": False},
    {"candidates": ["circularity"], "weight": 1.20, "log1p": False, "required": False},
    {"candidates": ["eccentricity"], "weight": 1.20, "log1p": False, "required": False},
    {"candidates": ["aspect_ratio"], "weight": 1.20, "log1p": False, "required": False},
    {"candidates": ["area_um2", "area"], "weight": 0.80, "log1p": True, "required": False},
    {"candidates": ["perimeter_um", "perimeter"], "weight": 0.80, "log1p": True, "required": False},
    {"candidates": ["convex_area_um2", "convex_area"], "weight": 0.75, "log1p": True, "required": False},
    {"candidates": ["filled_area_um2", "filled_area"], "weight": 0.75, "log1p": True, "required": False},
    {"candidates": ["major_axis_length_um", "major_axis_length"], "weight": 0.85, "log1p": True, "required": False},
    {"candidates": ["minor_axis_length_um", "minor_axis_length"], "weight": 0.85, "log1p": True, "required": False},
    {"candidates": ["equivalent_diameter_um", "equivalent_diameter"], "weight": 0.85, "log1p": True, "required": False},
]


# =========================
# Main clustering settings / 主聚类参数
# =========================
N_CLUSTERS = 2
RANDOM_STATE = 42
GMM_COVARIANCE_TYPE = "diag"
GMM_COVARIANCE_CANDIDATES = ["diag", "full", "tied"]
GMM_RANDOM_SEED_CANDIDATES = [42, 77, 101, 131, 197]
N_PCA_COMPONENTS_FOR_CLUSTERING = 5
PCA_COMPONENT_CANDIDATES = [4, 5, 6, 7]
MAIN_MODEL_SELECTION_OBJECTIVE = "spearman_abs"  # options: "spearman_abs", "silhouette"
USE_BAYESIAN_TUNING = True
BAYES_N_INIT = 20
BAYES_N_ITER = 100
USE_EDGE_ADJUSTED_FEATURES = False
USE_DE_REFINEMENT = True
DE_MAXITER = 60
DE_POPSIZE = 20

# IMPORTANT:
# If the biological meaning flips in a future run,
# swap this mapping manually.
CLUSTER_TO_STATE = {
    0: "deviated",
    1: "undifferentiated",
}
AUTO_RESOLVE_CLUSTER_TO_STATE = True

N_REVIEW_PER_CLUSTER = 80

MAKE_UMAP = True
MAKE_UMAP_2D = True
MAKE_UMAP_3D = True
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10
UMAP_RANDOM_STATE = RANDOM_STATE
SAVE_PCA_FIGURES = False

SAVE_REVIEW_MASKS = True
SAVE_REVIEW_OVERLAYS = True
SHOW_PREVIEW_AFTER_RUN = False
PREVIEW_MAX_IMAGES = 4

ORIG_MASK_SUFFIX = "_mask.npy"
QC_KEEP_MASK_SUFFIX = "_qc_keep_mask.npy"
IMAGE_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

PREFERRED_META_COLS = [
    "image_name",
    "label",
    "centroid_row",
    "centroid_col",
    "qc_keep",
]


# =========================
# Edge / colony geometry settings
# =========================
EDGE_AWARE_ENABLED = True

# Build colony-support from qc_keep binary region
COLONY_SUPPORT_DILATE_EQDIAM_FACTOR = 0.45
COLONY_SUPPORT_MIN_DILATE_UM = 3.0
COLONY_SUPPORT_ERODE_RATIO = 0.15

# Edge-band width
EDGE_BAND_EQDIAM_FACTOR = 2.40
EDGE_BAND_MIN_UM = 16.0

# Use median-ish object depth instead of center-like depth
EDGE_DISTANCE_OBJECT_PERCENTILE = 50

# Inner-fit region for main model fitting
INNER_FIT_EQDIAM_FACTOR = 2.80
INNER_FIT_MIN_UM = 18.0
MIN_INNER_FIT_RATIO = 0.15
MIN_INNER_FIT_ABS = 300


# =========================
# Morphology-only auxiliary model
# -------------------------
# 目的:
# 用 morphology evidence 判断:
# 当前 edge red 到底是 biology-like deviation,
# 还是只是 edge/sparsity 推出来的假红
# =========================
MORPH_FEATURE_SPECS = [
    {"candidates": ["adaptive_nb_area_mean_um2", "nb_area_mean_um2", "adaptive_nb_area_mean", "nb_area_mean"], "log1p": True, "required": True},
    {"candidates": ["adaptive_nb_circularity_mean", "nb_circularity_mean"], "log1p": False, "required": True},
    {"candidates": ["adaptive_nb_eccentricity_mean", "nb_eccentricity_mean"], "log1p": False, "required": True},
    {"candidates": ["adaptive_nb_aspect_ratio_mean", "nb_aspect_ratio_mean"], "log1p": False, "required": True},
    {"candidates": ["fixed_nb_area_mean_um2", "fixed_nb_area_mean"], "log1p": True, "required": False},
    {"candidates": ["fixed_nb_circularity_mean"], "log1p": False, "required": False},
    {"candidates": ["fixed_nb_eccentricity_mean"], "log1p": False, "required": False},
    {"candidates": ["fixed_nb_aspect_ratio_mean"], "log1p": False, "required": False},
]

MORPH_N_PCA = 4
MORPH_GMM_COVARIANCE_TYPE = "diag"

# Higher irregularity score => more likely deviated-like morphology
MORPH_IRREGULARITY_POS = [
    "adaptive_nb_eccentricity_mean",
    "adaptive_nb_aspect_ratio_mean",
    "fixed_nb_eccentricity_mean",
    "fixed_nb_aspect_ratio_mean",
]
MORPH_IRREGULARITY_NEG = [
    "adaptive_nb_circularity_mean",
    "fixed_nb_circularity_mean",
]


# =========================
# Single final-label pass
# 新逻辑:
# 1) 全局标签先由 main GMM 决定
# 2) morphology 只在 edge red 上做 veto
# 3) 暂时不主动把 edge green 打回 uncertain
# =========================
FULL_DEV_MIN = DEVIATED_PROB_MIN
FULL_UNDIFF_MIN = UNDIFF_PROB_MIN
UNCERTAIN_TO_DEV_MAIN_MIN = 0.62
UNCERTAIN_TO_DEV_MORPH_MIN = 0.52
UNCERTAIN_TO_DEV_MARGIN_MIN = 0.08

# Optional intensity augmentation for clustering.
# Uses image-relative robust normalization to reduce batch/image-level shift.
USE_INTENSITY_AUGMENTATION = False

# Edge correction tuning: stricter for edge-red false positives
EDGE_DEV_KEEP_MAIN_MIN = 0.80
EDGE_DEV_KEEP_MORPH_MIN = 0.52
EDGE_DEV_DOWNGRADE_MAIN_MAX = 0.74
EDGE_DEV_DOWNGRADE_MORPH_MAX = 0.35
EDGE_UNDIFF_PROMOTE_MAIN_MIN = 0.72
EDGE_UNDIFF_PROMOTE_MORPH_MAX = 0.40

# Only when main red is not strong enough AND morphology also does not support it
# do we downgrade edge red to uncertain
# =========================
# Optional debug output
# =========================
SAVE_EDGE_DEBUG_COLONY_SUPPORT = True
EDGE_DEBUG_DIR = OUT_DIR / "edge_debug"
if SAVE_EDGE_DEBUG_COLONY_SUPPORT:
    EDGE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Utilities
# =========================
def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def safe_log1p(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.where(x < 0, np.nan, x)
    return pd.Series(np.log1p(x), index=series.index)


def parse_float_safe(value) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) and x > 0 else None


def parse_pixel_size_from_text(text: str) -> tuple[float | None, float | None]:
    if not text:
        return None, None

    for px_pat, py_pat in [
        (r"PhysicalSizeX\s*=\s*['\"]?([0-9.eE+-]+)", r"PhysicalSizeY\s*=\s*['\"]?([0-9.eE+-]+)"),
        (r"pixel[_\s-]*size[_\s-]*x\s*[:=]\s*([0-9.eE+-]+)", r"pixel[_\s-]*size[_\s-]*y\s*[:=]\s*([0-9.eE+-]+)"),
    ]:
        mx = re.search(px_pat, text, flags=re.IGNORECASE)
        my = re.search(py_pat, text, flags=re.IGNORECASE)
        if mx and my:
            sx = parse_float_safe(mx.group(1))
            sy = parse_float_safe(my.group(1))
            if sx is not None and sy is not None:
                return sy, sx

    for pat in [r"PhysicalSizeX\s*=\s*['\"]?([0-9.eE+-]+)", r"pixel[_\s-]*size\s*[:=]\s*([0-9.eE+-]+)"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            s = parse_float_safe(m.group(1))
            if s is not None:
                return s, s

    return None, None


def get_image_pixel_size_um(img_path: Path) -> tuple[float, float]:
    with tiff.TiffFile(img_path) as tif:
        page = tif.pages[0]
        row_um, col_um = parse_pixel_size_from_text(page.description or "")
        if row_um is not None and col_um is not None:
            return row_um, col_um

        if tif.ome_metadata:
            row_um, col_um = parse_pixel_size_from_text(tif.ome_metadata)
            if row_um is not None and col_um is not None:
                return row_um, col_um

        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        if xres_tag is not None and yres_tag is not None and unit_tag is not None:
            xres = xres_tag.value
            yres = yres_tag.value
            unit = unit_tag.value
            xpp = float(xres[0]) / float(xres[1]) if isinstance(xres, tuple) else float(xres)
            ypp = float(yres[0]) / float(yres[1]) if isinstance(yres, tuple) else float(yres)
            if xpp > 0 and ypp > 0:
                if unit == 2:
                    return 25400.0 / ypp, 25400.0 / xpp
                if unit == 3:
                    return 10000.0 / ypp, 10000.0 / xpp

    raise RuntimeError(f"Cannot determine pixel size for image: {img_path}")


def resolve_feature_specs(df: pd.DataFrame, specs: list[dict], group_name: str) -> tuple[list[str], set[str], dict[str, float]]:
    selected = []
    log1p_cols = set()
    weights = {}
    missing = []

    for spec in specs:
        chosen = next((c for c in spec["candidates"] if c in df.columns), None)
        if chosen is None:
            if spec.get("required", True):
                missing.append(spec["candidates"][0])
            continue
        selected.append(chosen)
        if spec.get("log1p"):
            log1p_cols.add(chosen)
        if "weight" in spec:
            weights[chosen] = spec["weight"]

    if missing:
        raise KeyError(f"{group_name} missing required columns: {missing}")

    return selected, log1p_cols, weights


def add_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    alias_map = {
        "adaptive_nb_area_mean_um2": "nb_area_mean_um2",
        "adaptive_nb_area_mean": "nb_area_mean",
        "adaptive_nb_circularity_mean": "nb_circularity_mean",
        "adaptive_nb_eccentricity_mean": "nb_eccentricity_mean",
        "adaptive_nb_aspect_ratio_mean": "nb_aspect_ratio_mean",
    }
    for dst, src in alias_map.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]
    return df


def augment_with_intensity_features(df: pd.DataFrame, intensity_csv: Path) -> pd.DataFrame:
    if not USE_INTENSITY_AUGMENTATION:
        return df

    required_join_cols = ["image_name", "label"]
    if any(c not in df.columns for c in required_join_cols):
        print("[skip] Intensity augmentation skipped: INPUT_CSV lacks image_name/label.")
        return df
    if not intensity_csv.exists():
        print(f"[skip] Intensity augmentation skipped: not found {intensity_csv}")
        return df

    intensity_df = pd.read_csv(intensity_csv)
    needed_intensity_cols = ["image_name", "label", "mean_intensity"]
    missing = [c for c in needed_intensity_cols if c not in intensity_df.columns]
    if missing:
        print(f"[skip] Intensity augmentation skipped: intensity CSV missing columns {missing}")
        return df

    dup_mask = intensity_df.duplicated(subset=["image_name", "label"], keep=False)
    if dup_mask.any():
        raise ValueError("Intensity CSV has duplicated (image_name, label) rows.")

    merged = df.merge(
        intensity_df[needed_intensity_cols],
        on=["image_name", "label"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_from_intensity"),
    )

    def _as_series(frame: pd.DataFrame, col: str) -> pd.Series | None:
        if col not in frame.columns:
            return None
        obj = frame[col]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    base_series = _as_series(merged, "mean_intensity")
    aux_series = _as_series(merged, "mean_intensity_from_intensity")

    if base_series is None and aux_series is not None:
        merged["mean_intensity"] = aux_series
    elif base_series is not None and aux_series is not None:
        base_vals = pd.to_numeric(base_series, errors="coerce")
        aux_vals = pd.to_numeric(aux_series, errors="coerce")
        merged["mean_intensity"] = base_vals.where(base_vals.notna(), aux_vals)

    if "mean_intensity" not in merged.columns:
        raise KeyError("mean_intensity column missing after intensity augmentation merge.")

    merged["mean_intensity"] = pd.to_numeric(merged["mean_intensity"], errors="coerce")
    merged["mean_intensity_img_rel"] = merged.groupby("image_name")["mean_intensity"].transform(
        lambda s: (s - s.median()) / max(float(s.quantile(0.75) - s.quantile(0.25)), 1e-9)
    )
    ok = int(merged["mean_intensity_img_rel"].notna().sum())
    print(f"[info] Intensity augmentation enabled: mean_intensity_img_rel valid rows = {ok}/{len(merged)}")
    return merged


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim != 2:
        raise ValueError(f"normalize_to_u8 expects 2D image, got shape={x.shape}")

    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)

    vals = x[finite]
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)

    if hi <= lo:
        hi = vals.max()
        lo = vals.min()
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def ensure_2d_image(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    raise ValueError(f"Expected 2D grayscale image, got shape={x.shape}")


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    return find_boundaries(mask, mode="outer")


def relabel_compact(mask: np.ndarray) -> np.ndarray:
    labels = np.unique(mask)
    labels = labels[labels > 0]
    out = np.zeros_like(mask, dtype=np.int32)
    for new_label, old_label in enumerate(labels, start=1):
        out[mask == old_label] = new_label
    return out


_IMAGE_FILE_CACHE: list[Path] | None = None


def _normalize_name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _list_image_files_cached() -> list[Path]:
    global _IMAGE_FILE_CACHE
    if _IMAGE_FILE_CACHE is not None:
        return _IMAGE_FILE_CACHE

    if not IMAGE_DIR.exists():
        _IMAGE_FILE_CACHE = []
        return _IMAGE_FILE_CACHE

    ext_set = {e.lower() for e in IMAGE_EXTS}
    files: list[Path] = []
    for p in IMAGE_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_set:
            files.append(p)
    _IMAGE_FILE_CACHE = files
    return _IMAGE_FILE_CACHE


def find_matching_image(image_name: str) -> Path | None:
    stem = Path(image_name).stem
    if not IMAGE_DIR.exists():
        return None

    # 1) Fast direct check in root directory
    for ext in IMAGE_EXTS:
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p

    # 2) Recursive + case-insensitive matching with normalized keys
    files = _list_image_files_cached()
    if not files:
        return None

    stem_key = _normalize_name_key(stem)
    if not stem_key:
        return None

    exact_hits = [p for p in files if _normalize_name_key(p.stem) == stem_key]
    if exact_hits:
        return sorted(exact_hits, key=lambda x: len(x.name))[0]

    prefix_hits = [p for p in files if _normalize_name_key(p.stem).startswith(stem_key)]
    if prefix_hits:
        return sorted(prefix_hits, key=lambda x: len(x.name))[0]

    contains_hits = [p for p in files if stem_key in _normalize_name_key(p.stem)]
    if contains_hits:
        return sorted(contains_hits, key=lambda x: len(x.name))[0]

    # 3) Fallback mapping for generic image names in CSV (e.g., "Oct4")
    # Prefer DAPI image for geometry/mask overlay if available.
    if stem_key in {"oct4"}:
        dapi_hits = [p for p in files if "dapi" in p.name.lower()]
        if dapi_hits:
            return sorted(dapi_hits, key=lambda x: len(x.name))[0]
        if files:
            return sorted(files, key=lambda x: len(x.name))[0]

    return None


def find_matching_orig_mask(image_name: str) -> Path | None:
    stem = Path(image_name).stem
    candidates = [
        ORIG_MASK_DIR / f"{image_name}{ORIG_MASK_SUFFIX}",
        ORIG_MASK_DIR / f"{stem}{ORIG_MASK_SUFFIX}",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(ORIG_MASK_DIR.glob(f"{stem}*{ORIG_MASK_SUFFIX}"))
    return hits[0] if hits else None


def find_matching_qc_keep_mask(image_name: str) -> Path | None:
    stem = Path(image_name).stem
    candidates = [
        QC_KEEP_MASK_DIR / f"{image_name}{QC_KEEP_MASK_SUFFIX}",
        QC_KEEP_MASK_DIR / f"{stem}{QC_KEEP_MASK_SUFFIX}",
        QC_KEEP_MASK_DIR / f"{image_name}{ORIG_MASK_SUFFIX}",
        QC_KEEP_MASK_DIR / f"{stem}{ORIG_MASK_SUFFIX}",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(QC_KEEP_MASK_DIR.glob(f"{stem}*{QC_KEEP_MASK_SUFFIX}"))
    if hits:
        return hits[0]
    hits = list(QC_KEEP_MASK_DIR.glob(f"{stem}*{ORIG_MASK_SUFFIX}"))
    return hits[0] if hits else None


def make_label_overlay(
    img_gray: np.ndarray,
    deviated_mask: np.ndarray,
    undiff_mask: np.ndarray,
    uncertain_mask: np.ndarray,
) -> np.ndarray:
    base = normalize_to_u8(img_gray)
    rgb = np.stack([base, base, base], axis=-1)

    bd = boundary_from_mask(deviated_mask)
    bu = boundary_from_mask(undiff_mask)
    bx = boundary_from_mask(uncertain_mask)

    rgb[bd] = np.array([255, 0, 0], dtype=np.uint8)      # red
    rgb[bu] = np.array([0, 255, 0], dtype=np.uint8)      # green
    rgb[bx] = np.array([255, 255, 0], dtype=np.uint8)    # yellow
    return rgb


def make_group_summary(df_out: pd.DataFrame, feature_cols: list[str], group_col: str) -> pd.DataFrame:
    rows = []
    for group_name, sub in df_out.groupby(group_col, dropna=False):
        row = {group_col: group_name, "n": len(sub)}
        for col in feature_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by="n", ascending=False)


def sample_for_review(df_out: pd.DataFrame, group_col: str, n_each: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for _, sub in df_out.groupby(group_col, dropna=False):
        if len(sub) <= n_each:
            samp = sub.copy()
        else:
            samp = sub.sample(n=n_each, random_state=seed)
        parts.append(samp)
    return pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()


def save_scatter_figure(df_embed: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 8))
    color_map = {
        "deviated": "red",
        "undifferentiated": "green",
        "uncertain": "yellow",
    }
    for name, sub in df_embed.groupby(group_col, dropna=False):
        plt.scatter(sub[x_col], sub[y_col], s=8, alpha=0.7, label=str(name),color=color_map.get(str(name), "gray"))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_scatter_figure_3d(
    df_embed: pd.DataFrame,
    xyz_cols: list[str],
    group_col: str,
    title: str,
    out_path: Path,
):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    color_map = {
        "deviated": "red",
        "undifferentiated": "green",
        "uncertain": "yellow",
    }
    x_col, y_col, z_col = xyz_cols
    for name, sub in df_embed.groupby(group_col, dropna=False):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            sub[z_col],
            s=8,
            alpha=0.6,
            color=color_map.get(str(name), "gray"),
            label=str(name),
            depthshade=False,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.view_init(elev=22, azim=45)
    ax.legend(markerscale=2, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_deviation_intensity_relation(
    df_out: pd.DataFrame,
    intensity_csv: Path,
    out_csv: Path,
    out_fig: Path,
) -> dict | None:
    if not intensity_csv.exists():
        print(f"[skip] Intensity CSV not found: {intensity_csv}")
        return None

    intensity_df = pd.read_csv(intensity_csv)
    required_cols = ["image_name", "label", "mean_intensity"]
    missing = [c for c in required_cols if c not in intensity_df.columns]
    if missing:
        print(f"[skip] Intensity CSV missing columns: {missing}")
        return None

    dup_mask = intensity_df.duplicated(subset=["image_name", "label"], keep=False)
    if dup_mask.any():
        raise ValueError("Intensity CSV has duplicated (image_name, label) rows.")

    # Always use intensity CSV as the single source of truth for mean_intensity,
    # to avoid stale/duplicated columns carried in df_out.
    base_cols = [c for c in df_out.columns if c != "mean_intensity"]
    merged = df_out[base_cols].merge(
        intensity_df[["image_name", "label", "mean_intensity"]],
        on=["image_name", "label"],
        how="left",
        validate="one_to_one",
    )

    merged["deviated_score"] = pd.to_numeric(merged["gmm_prob_deviated_raw"], errors="coerce")
    if "mean_intensity" not in merged.columns:
        raise KeyError("mean_intensity column missing after merge in save_deviation_intensity_relation.")
    merged["mean_intensity"] = pd.to_numeric(merged["mean_intensity"], errors="coerce")
    print(
        "[info] Intensity source fixed to CSV: "
        f"{intensity_csv} | min={float(np.nanmin(merged['mean_intensity'])):.3f} "
        f"median={float(np.nanmedian(merged['mean_intensity'])):.3f} "
        f"max={float(np.nanmax(merged['mean_intensity'])):.3f}"
    )

    valid = merged["deviated_score"].notna() & merged["mean_intensity"].notna()
    rel_df = merged.loc[
        valid,
        [
            "image_name",
            "label",
            "final_state_label",
            "deviated_score",
            "gmm_prob_margin_dev_minus_undiff_raw",
            "mean_intensity",
        ],
    ].copy()

    if len(rel_df) == 0:
        print("[skip] No valid rows for deviation-vs-intensity relation.")
        return None

    rel_df["mean_intensity_img_rel"] = rel_df.groupby("image_name")["mean_intensity"].transform(
        lambda s: (s - s.median()) / max(float(s.quantile(0.75) - s.quantile(0.25)), 1e-9)
    )
    rel_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_csv}")

    pearson_r = float(rel_df["deviated_score"].corr(rel_df["mean_intensity"], method="pearson"))
    spearman_r = float(rel_df["deviated_score"].corr(rel_df["mean_intensity"], method="spearman"))

    plt.figure(figsize=(7.5, 6.0))
    color_map = {
        "deviated": "red",
        "undifferentiated": "green",
        "uncertain": "gold",
    }
    for state_name, sub in rel_df.groupby("final_state_label", dropna=False):
        plt.scatter(
            sub["mean_intensity_plot"],
            sub["deviated_score"],
            s=9,
            alpha=0.35,
            color=color_map.get(str(state_name), "gray"),
            label=str(state_name),
        )
    plt.xlabel("mean_intensity")
    plt.ylabel("deviated_score")
    plt.title(
        "Deviation score vs mean intensity\n"
        f"Pearson r = {pearson_r:.3f}, Spearman rho = {spearman_r:.3f}"
    )
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=220)
    plt.close()
    print(f"[saved] {out_fig}")

    return {
        "n_valid_rows": int(len(rel_df)),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "csv": str(out_csv),
        "figure": str(out_fig),
    }


def save_deviation_intensity_boxplot(rel_df: pd.DataFrame, out_fig: Path) -> None:
    work = rel_df.copy()
    work = work.loc[work["final_state_label"].isin(["undifferentiated", "uncertain", "deviated"])].copy()
    if len(work) == 0:
        return
    order = [x for x in ["undifferentiated", "uncertain", "deviated"] if x in set(work["final_state_label"].astype(str))]
    label_map = {"undifferentiated": "cluster1", "uncertain": "uncertain", "deviated": "cluster2"}
    color_map = {"undifferentiated": "#67c587", "uncertain": "#d8b365", "deviated": "#d87a7a"}
    data = []
    counts = []
    for c in order:
        vals = pd.to_numeric(work.loc[work["final_state_label"] == c, "mean_intensity"], errors="coerce").dropna().to_numpy()
        data.append(vals)
        counts.append(int(len(vals)))
    if not any(len(x) > 0 for x in data):
        return

    np.random.seed(RANDOM_STATE)
    plt.figure(figsize=(8.4, 6.4))
    ax = plt.gca()
    ax.set_facecolor("none")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)

    x_pos = np.arange(1, len(order) + 1, dtype=float)
    for i, c in enumerate(order):
        vals = data[i]
        if len(vals) == 0:
            continue
        jitter = np.random.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos[i]) + jitter,
            vals,
            s=10,
            alpha=0.5,
            color=color_map.get(c, "#999999"),
            edgecolors="none",
        )

    bp = ax.boxplot(
        data,
        tick_labels=[f"{label_map.get(c, c)}\n(n={n})" for c, n in zip(order, counts)],
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )
    for box in bp["boxes"]:
        box.set_facecolor("#ffffff")
        box.set_alpha(0.75)
        box.set_linewidth(1.2)
    for med in bp["medians"]:
        med.set_color("#333333")
        med.set_linewidth(1.5)

    ax.set_ylabel("Oct-4 mean intensity")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=220, transparent=True)
    plt.close()
    print(f"[saved] {out_fig}")


def get_cluster_prob_columns(cluster_to_state: dict[int, str]) -> tuple[int, int]:
    dev_clusters = [k for k, v in cluster_to_state.items() if v == "deviated"]
    undiff_clusters = [k for k, v in cluster_to_state.items() if v == "undifferentiated"]
    if len(dev_clusters) != 1 or len(undiff_clusters) != 1:
        raise ValueError("CLUSTER_TO_STATE must map exactly one cluster to deviated and one to undifferentiated.")
    return dev_clusters[0], undiff_clusters[0]


def get_kept_label_areas_from_masks(orig_mask: np.ndarray, keep_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lbls, counts = np.unique(orig_mask[keep_binary], return_counts=True)
    valid = lbls > 0
    return lbls[valid].astype(int), counts[valid].astype(float)


def resolve_label_in_keep_mask(row: pd.Series, orig_mask: np.ndarray, keep_binary: np.ndarray) -> int | None:
    label_val = pd.to_numeric(pd.Series([row.get("label")]), errors="coerce").iloc[0]
    if pd.notna(label_val):
        label_int = int(label_val)
        if ((orig_mask == label_int) & keep_binary).any():
            return label_int

    rr = pd.to_numeric(pd.Series([row.get("centroid_row")]), errors="coerce").iloc[0]
    cc = pd.to_numeric(pd.Series([row.get("centroid_col")]), errors="coerce").iloc[0]
    if pd.isna(rr) or pd.isna(cc):
        return None

    r = int(np.clip(round(float(rr)), 0, orig_mask.shape[0] - 1))
    c = int(np.clip(round(float(cc)), 0, orig_mask.shape[1] - 1))
    if keep_binary[r, c] and orig_mask[r, c] > 0:
        return int(orig_mask[r, c])

    for radius in (2, 4, 6):
        r0 = max(0, r - radius)
        r1 = min(orig_mask.shape[0], r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(orig_mask.shape[1], c + radius + 1)
        win_keep = keep_binary[r0:r1, c0:c1]
        win_mask = orig_mask[r0:r1, c0:c1]
        valid = win_keep & (win_mask > 0)
        if valid.any():
            labels, counts = np.unique(win_mask[valid], return_counts=True)
            pos = labels > 0
            labels = labels[pos]
            counts = counts[pos]
            if len(labels):
                best_label = labels[np.argmax(counts)]
                return int(best_label)

    return None


def compute_median_equiv_diameter_um(areas_px: np.ndarray, pixel_area_um2: float) -> float:
    if areas_px is None or len(areas_px) == 0:
        return 12.0
    eqd = 2.0 * np.sqrt((areas_px * pixel_area_um2) / np.pi)
    eqd = eqd[np.isfinite(eqd)]
    return float(np.median(eqd)) if len(eqd) else 12.0


def build_colony_support_mask(
    qc_keep_mask: np.ndarray,
    median_eqdiam_um: float,
    pixel_size_mean_um: float,
) -> tuple[np.ndarray, int, float]:
    nuclei_binary = qc_keep_mask > 0
    if not nuclei_binary.any():
        return nuclei_binary.astype(bool), 0, 0.0

    dilate_um = float(max(COLONY_SUPPORT_MIN_DILATE_UM, median_eqdiam_um * COLONY_SUPPORT_DILATE_EQDIAM_FACTOR))
    dilate_px = int(round(max(1.0, dilate_um / max(pixel_size_mean_um, 1e-12))))
    erode_px = int(round(max(0, dilate_px * COLONY_SUPPORT_ERODE_RATIO)))

    support = binary_dilation(nuclei_binary, iterations=dilate_px)
    support = binary_fill_holes(support)
    if erode_px > 0:
        support = binary_erosion(support, iterations=erode_px)
    support = np.logical_or(support, nuclei_binary)
    support = binary_fill_holes(support)
    return support.astype(bool), dilate_px, dilate_um


def save_colony_support_debug(image_name: str, img_gray: np.ndarray, support_mask: np.ndarray):
    if not SAVE_EDGE_DEBUG_COLONY_SUPPORT:
        return
    base = normalize_to_u8(img_gray)
    rgb = np.stack([base, base, base], axis=-1)
    bd = find_boundaries(support_mask.astype(np.uint8), mode="outer")
    rgb[bd] = np.array([0, 255, 255], dtype=np.uint8)
    tiff.imwrite(str(EDGE_DEBUG_DIR / f"{image_name}_colony_support_overlay.tif"), rgb)


def compute_edge_metrics_for_all_rows(df_meta: pd.DataFrame) -> pd.DataFrame:
    require_columns(df_meta, ["image_name", "label"], "df_meta")
    rows = []

    for image_name, sub in df_meta.groupby("image_name"):
        orig_mask_path = find_matching_orig_mask(image_name)
        qc_keep_mask_path = find_matching_qc_keep_mask(image_name)
        img_path = find_matching_image(image_name)

        if orig_mask_path is None or qc_keep_mask_path is None:
            for idx in sub.index:
                rows.append({
                    "row_index": idx,
                    "pixel_size_row_um": np.nan,
                    "pixel_size_col_um": np.nan,
                    "pixel_size_mean_um": np.nan,
                    "boundary_distance_px": np.nan,
                    "boundary_distance_um": np.nan,
                    "boundary_distance_min_px": np.nan,
                    "boundary_distance_min_um": np.nan,
                    "edge_band_px": np.nan,
                    "edge_band_um": np.nan,
                    "inner_fit_cutoff_px": np.nan,
                    "inner_fit_cutoff_um": np.nan,
                    "median_eqdiam_px": np.nan,
                    "median_eqdiam_um": np.nan,
                    "colony_support_dilate_px": np.nan,
                    "colony_support_dilate_um": np.nan,
                    "is_edge_band": False,
                    "is_inner_fit": False,
                    "edge_metric_status": "mask_missing",
                })
            continue

        try:
            orig_mask = np.load(orig_mask_path)
            qc_keep_mask = np.load(qc_keep_mask_path)

            if orig_mask.ndim != 2 or qc_keep_mask.ndim != 2:
                raise ValueError("orig_mask or qc_keep_mask is not 2D")
            if orig_mask.shape != qc_keep_mask.shape:
                raise ValueError(f"shape mismatch: orig={orig_mask.shape}, keep={qc_keep_mask.shape}")

            if img_path is None:
                raise RuntimeError("matching image not found for pixel size lookup")

            pixel_size_row_um, pixel_size_col_um = get_image_pixel_size_um(img_path)
            pixel_size_mean_um = math.sqrt(pixel_size_row_um * pixel_size_col_um)
            pixel_area_um2 = pixel_size_row_um * pixel_size_col_um

            keep_binary = qc_keep_mask > 0
            kept_labels, kept_areas = get_kept_label_areas_from_masks(orig_mask, keep_binary)
            median_eqdiam_um = compute_median_equiv_diameter_um(kept_areas, pixel_area_um2)
            median_eqdiam_px = median_eqdiam_um / max(pixel_size_mean_um, 1e-12)
            support_mask, dilate_px, dilate_um = build_colony_support_mask(
                qc_keep_mask,
                median_eqdiam_um,
                pixel_size_mean_um,
            )
            dist_map_um = distance_transform_edt(support_mask, sampling=(pixel_size_row_um, pixel_size_col_um))
            dist_map_px = distance_transform_edt(support_mask)
            edge_band_um = float(max(EDGE_BAND_MIN_UM, median_eqdiam_um * EDGE_BAND_EQDIAM_FACTOR))
            inner_fit_cutoff_um = float(max(INNER_FIT_MIN_UM, median_eqdiam_um * INNER_FIT_EQDIAM_FACTOR))
            edge_band_px = edge_band_um / max(pixel_size_mean_um, 1e-12)
            inner_fit_cutoff_px = inner_fit_cutoff_um / max(pixel_size_mean_um, 1e-12)

            try:
                img_gray = ensure_2d_image(tiff.imread(img_path))
                if img_gray.shape == support_mask.shape:
                    save_colony_support_debug(image_name, img_gray, support_mask)
            except Exception:
                pass

            for idx, row in sub.iterrows():
                label_int = resolve_label_in_keep_mask(row, orig_mask, keep_binary)
                if label_int is None:
                    rows.append({
                        "row_index": idx,
                        "pixel_size_row_um": pixel_size_row_um,
                        "pixel_size_col_um": pixel_size_col_um,
                        "pixel_size_mean_um": pixel_size_mean_um,
                        "boundary_distance_px": np.nan,
                        "boundary_distance_um": np.nan,
                        "boundary_distance_min_px": np.nan,
                        "boundary_distance_min_um": np.nan,
                        "edge_band_px": edge_band_px,
                        "edge_band_um": edge_band_um,
                        "inner_fit_cutoff_px": inner_fit_cutoff_px,
                        "inner_fit_cutoff_um": inner_fit_cutoff_um,
                        "median_eqdiam_px": median_eqdiam_px,
                        "median_eqdiam_um": median_eqdiam_um,
                        "colony_support_dilate_px": dilate_px,
                        "colony_support_dilate_um": dilate_um,
                        "is_edge_band": False,
                        "is_inner_fit": False,
                        "edge_metric_status": "label_unresolved",
                    })
                    continue

                obj_mask = (orig_mask == label_int) & keep_binary
                if not obj_mask.any():
                    rows.append({
                        "row_index": idx,
                        "pixel_size_row_um": pixel_size_row_um,
                        "pixel_size_col_um": pixel_size_col_um,
                        "pixel_size_mean_um": pixel_size_mean_um,
                        "boundary_distance_px": np.nan,
                        "boundary_distance_um": np.nan,
                        "boundary_distance_min_px": np.nan,
                        "boundary_distance_min_um": np.nan,
                        "edge_band_px": edge_band_px,
                        "edge_band_um": edge_band_um,
                        "inner_fit_cutoff_px": inner_fit_cutoff_px,
                        "inner_fit_cutoff_um": inner_fit_cutoff_um,
                        "median_eqdiam_px": median_eqdiam_px,
                        "median_eqdiam_um": median_eqdiam_um,
                        "colony_support_dilate_px": dilate_px,
                        "colony_support_dilate_um": dilate_um,
                        "is_edge_band": False,
                        "is_inner_fit": False,
                        "edge_metric_status": "label_not_in_keep_region",
                    })
                    continue

                dvals_um = dist_map_um[obj_mask]
                dvals_um = dvals_um[np.isfinite(dvals_um)]
                dvals_px = dist_map_px[obj_mask]
                dvals_px = dvals_px[np.isfinite(dvals_px)]
                if len(dvals_um) == 0:
                    rows.append({
                        "row_index": idx,
                        "pixel_size_row_um": pixel_size_row_um,
                        "pixel_size_col_um": pixel_size_col_um,
                        "pixel_size_mean_um": pixel_size_mean_um,
                        "boundary_distance_px": np.nan,
                        "boundary_distance_um": np.nan,
                        "boundary_distance_min_px": np.nan,
                        "boundary_distance_min_um": np.nan,
                        "edge_band_px": edge_band_px,
                        "edge_band_um": edge_band_um,
                        "inner_fit_cutoff_px": inner_fit_cutoff_px,
                        "inner_fit_cutoff_um": inner_fit_cutoff_um,
                        "median_eqdiam_px": median_eqdiam_px,
                        "median_eqdiam_um": median_eqdiam_um,
                        "colony_support_dilate_px": dilate_px,
                        "colony_support_dilate_um": dilate_um,
                        "is_edge_band": False,
                        "is_inner_fit": False,
                        "edge_metric_status": "no_distance",
                    })
                    continue

                boundary_distance_um = float(np.percentile(dvals_um, EDGE_DISTANCE_OBJECT_PERCENTILE))
                boundary_distance_min_um = float(np.min(dvals_um))
                boundary_distance_px = float(np.percentile(dvals_px, EDGE_DISTANCE_OBJECT_PERCENTILE))
                boundary_distance_min_px = float(np.min(dvals_px))
                is_edge_band = bool(boundary_distance_um <= edge_band_um)
                is_inner_fit = bool(boundary_distance_um >= inner_fit_cutoff_um)

                rows.append({
                    "row_index": idx,
                    "pixel_size_row_um": pixel_size_row_um,
                    "pixel_size_col_um": pixel_size_col_um,
                    "pixel_size_mean_um": pixel_size_mean_um,
                    "boundary_distance_px": boundary_distance_px,
                    "boundary_distance_um": boundary_distance_um,
                    "boundary_distance_min_px": boundary_distance_min_px,
                    "boundary_distance_min_um": boundary_distance_min_um,
                    "edge_band_px": edge_band_px,
                    "edge_band_um": edge_band_um,
                    "inner_fit_cutoff_px": inner_fit_cutoff_px,
                    "inner_fit_cutoff_um": inner_fit_cutoff_um,
                    "median_eqdiam_px": median_eqdiam_px,
                    "median_eqdiam_um": median_eqdiam_um,
                    "colony_support_dilate_px": dilate_px,
                    "colony_support_dilate_um": dilate_um,
                    "is_edge_band": is_edge_band,
                    "is_inner_fit": is_inner_fit,
                    "edge_metric_status": "ok",
                })

        except Exception as e:
            print(f"[edge metric fail] {image_name}: {e}")
            for idx in sub.index:
                rows.append({
                    "row_index": idx,
                    "pixel_size_row_um": np.nan,
                    "pixel_size_col_um": np.nan,
                    "pixel_size_mean_um": np.nan,
                    "boundary_distance_px": np.nan,
                    "boundary_distance_um": np.nan,
                    "boundary_distance_min_px": np.nan,
                    "boundary_distance_min_um": np.nan,
                    "edge_band_px": np.nan,
                    "edge_band_um": np.nan,
                    "inner_fit_cutoff_px": np.nan,
                    "inner_fit_cutoff_um": np.nan,
                    "median_eqdiam_px": np.nan,
                    "median_eqdiam_um": np.nan,
                    "colony_support_dilate_px": np.nan,
                    "colony_support_dilate_um": np.nan,
                    "is_edge_band": False,
                    "is_inner_fit": False,
                    "edge_metric_status": f"failed: {e}",
                })

    edge_df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    if len(edge_df) == 0:
        raise RuntimeError("Failed to compute edge metrics: no rows generated.")
    return edge_df


def make_inner_fit_mask(edge_df: pd.DataFrame) -> np.ndarray:
    ok = edge_df["edge_metric_status"].astype(str).eq("ok").to_numpy()
    inner = edge_df["is_inner_fit"].fillna(False).to_numpy(dtype=bool)
    fit_mask = ok & inner

    min_required = max(MIN_INNER_FIT_ABS, int(np.ceil(len(edge_df) * MIN_INNER_FIT_RATIO)))
    if fit_mask.sum() < min_required:
        dist = pd.to_numeric(edge_df["boundary_distance_um"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(dist) & ok
        if valid.sum() == 0:
            print("[warn] No valid nuclei available for inner-colony fitting; falling back to all nuclei.")
            return np.ones(len(edge_df), dtype=bool)
        order = np.argsort(np.where(valid, dist, -np.inf))
        chosen = order[-min(min_required, valid.sum()):]
        fit_mask = np.zeros(len(edge_df), dtype=bool)
        fit_mask[chosen] = True
        fit_mask &= valid

    return fit_mask


def add_edge_adjusted_feature_columns(df: pd.DataFrame, edge_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    out = df.copy()
    if "boundary_distance_um" not in edge_df.columns:
        return out

    dist = pd.to_numeric(edge_df["boundary_distance_um"], errors="coerce")
    ok = edge_df["edge_metric_status"].astype(str).eq("ok") & dist.notna()
    if ok.sum() < max(100, n_bins * 5):
        return out

    work = pd.DataFrame({"dist": dist})
    work = work.loc[ok].copy()
    work["bin"] = pd.qcut(work["dist"], q=min(n_bins, int(ok.sum())), duplicates="drop")
    if work["bin"].nunique() < 3:
        return out

    confounded_cols = [
        "nn1_distance_um",
        "knn6_distance_mean_um",
        "local_density_per_um2",
        "adaptive_nb_area_mean_um2",
        "fixed_neighbor_count",
        "fixed_nb_area_mean_um2",
    ]
    for col in confounded_cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        tmp = pd.DataFrame({"v": vals, "dist": dist})
        tmp = tmp.loc[ok].copy()
        tmp["bin"] = work["bin"]
        med_by_bin = tmp.groupby("bin")["v"].median()

        all_bins = pd.cut(dist, bins=med_by_bin.index.categories) if hasattr(med_by_bin.index, "categories") else None
        if all_bins is None:
            continue
        baseline = all_bins.map(med_by_bin).astype(float)
        resid = vals - baseline
        # shift to positive for log1p-safe downstream use
        min_val = float(np.nanmin(resid.to_numpy(dtype=float))) if np.isfinite(resid).any() else 0.0
        if np.isfinite(min_val) and min_val <= 0:
            resid = resid - min_val + 1e-6
        out[f"{col}__edge_adj"] = resid

    return out


def fit_transform_pipeline_on_inner(
    X_df: pd.DataFrame,
    fit_mask: np.ndarray,
    feature_weights: dict[str, float],
    n_pca_components: int | None = None,
):
    X_fit_df = X_df.loc[fit_mask].copy()

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_fit_df)
    X_imp_all = imputer.transform(X_df)

    scaler = RobustScaler()
    scaler.fit(imputer.transform(X_fit_df))
    X_scaled_all = scaler.transform(X_imp_all)

    feature_weight_vector = np.array([feature_weights.get(c, 1.0) for c in X_df.columns], dtype=float)
    X_weighted_all = X_scaled_all * feature_weight_vector[np.newaxis, :]

    X_fit_weighted = X_weighted_all[fit_mask]
    target_n_pca = N_PCA_COMPONENTS_FOR_CLUSTERING if n_pca_components is None else int(n_pca_components)
    n_pca = min(target_n_pca, X_fit_weighted.shape[1], X_fit_weighted.shape[0])
    if n_pca < 2:
        raise ValueError("Not enough inner-colony samples/features for PCA.")

    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    pca.fit(X_fit_weighted)
    X_pca_all = pca.transform(X_weighted_all)

    return imputer, scaler, pca, X_weighted_all, X_pca_all, feature_weight_vector


def fit_best_main_gmm(
    X_pca_all: np.ndarray,
    fit_mask: np.ndarray,
    df_for_eval: pd.DataFrame,
):
    X_fit = X_pca_all[fit_mask]
    rows = []
    best = None
    best_key = None
    for cov in GMM_COVARIANCE_CANDIDATES:
        for seed in GMM_RANDOM_SEED_CANDIDATES:
            try:
                gmm = GaussianMixture(
                    n_components=N_CLUSTERS,
                    covariance_type=cov,
                    random_state=seed,
                )
                gmm.fit(X_fit)
                pred_fit = gmm.predict(X_fit)
                if len(np.unique(pred_fit)) < 2:
                    sil = -1.0
                else:
                    sil = float(silhouette_score(X_fit, pred_fit))
                bic = float(gmm.bic(X_fit))
                aic = float(gmm.aic(X_fit))
                pred_all = gmm.predict(X_pca_all)
                prob_all = gmm.predict_proba(X_pca_all)
                resolved = resolve_cluster_to_state_from_data(df_for_eval, pred_all)
                dev_idx, undiff_idx = get_cluster_prob_columns(resolved)
                p_dev = pd.Series(prob_all[:, dev_idx])
                p_undiff = pd.Series(prob_all[:, undiff_idx])
                margin = p_dev - p_undiff
                rho = np.nan
                rho_abs = np.nan
                intensity_col = None
                if "mean_intensity" in df_for_eval.columns:
                    intensity_col = "mean_intensity"
                elif "mean_intensity_img_rel" in df_for_eval.columns:
                    intensity_col = "mean_intensity_img_rel"
                if intensity_col is not None:
                    intensity = pd.to_numeric(df_for_eval[intensity_col], errors="coerce")
                    valid = intensity.notna() & p_dev.notna()
                    if valid.sum() >= 20:
                        rho = float(p_dev.loc[valid].corr(intensity.loc[valid], method="spearman"))
                        rho_abs = float(abs(rho))

                row = {
                    "covariance_type": cov,
                    "seed": int(seed),
                    "silhouette": sil,
                    "bic": bic,
                    "aic": aic,
                    "spearman_rho": rho,
                    "spearman_abs": rho_abs,
                    "resolved_cluster_to_state": str(resolved),
                }
                rows.append(row)
                if MAIN_MODEL_SELECTION_OBJECTIVE == "spearman_abs" and np.isfinite(rho_abs):
                    key = (rho_abs, sil, -bic)
                else:
                    key = (sil, -bic)
                if best is None or key > best_key:
                    best = gmm
                    best_key = key
            except Exception as e:
                rows.append(
                    {"covariance_type": cov, "seed": int(seed), "silhouette": np.nan, "bic": np.nan, "aic": np.nan, "error": str(e)}
                )
                continue

    if best is None:
        raise RuntimeError("Failed to fit any main GMM candidate.")
    return best, pd.DataFrame(rows)


def _weight_with_scales(base_weights: dict[str, float], cols: list[str], dist_scale: float, density_scale: float, morph_scale: float):
    out = {}
    for c in cols:
        w = float(base_weights.get(c, 1.0))
        lc = c.lower()
        if ("distance" in lc) or ("neighbor_count" in lc) or ("density" in lc):
            w *= dist_scale
        if ("area" in lc) or ("density" in lc):
            w *= density_scale
        if ("circularity" in lc) or ("eccentricity" in lc) or ("aspect_ratio" in lc):
            w *= morph_scale
        out[c] = w
    return out


def bayesian_optimize_spearman(
    X_df: pd.DataFrame,
    fit_mask: np.ndarray,
    df_eval: pd.DataFrame,
    base_weights: dict[str, float],
):
    rng = np.random.default_rng(RANDOM_STATE)
    cov_map = {0: "diag", 1: "full", 2: "tied"}
    bounds = np.array([
        [float(min(PCA_COMPONENT_CANDIDATES)), float(max(PCA_COMPONENT_CANDIDATES))],  # n_pca
        [0.0, 2.0],  # cov idx
        [0.6, 1.5],  # dist scale
        [0.6, 1.5],  # density scale
        [0.6, 1.5],  # morph scale
        [0.0, float(len(GMM_RANDOM_SEED_CANDIDATES) - 1)],  # seed idx
    ], dtype=float)

    def decode(x):
        n_pca = int(np.clip(round(x[0]), min(PCA_COMPONENT_CANDIDATES), max(PCA_COMPONENT_CANDIDATES)))
        cov = cov_map[int(np.clip(round(x[1]), 0, 2))]
        dist_s = float(x[2]); dens_s = float(x[3]); morph_s = float(x[4])
        seed = int(GMM_RANDOM_SEED_CANDIDATES[int(np.clip(round(x[5]), 0, len(GMM_RANDOM_SEED_CANDIDATES)-1))])
        return n_pca, cov, dist_s, dens_s, morph_s, seed

    def evaluate_vector(x):
        n_pca, cov, dist_s, dens_s, morph_s, seed = decode(x)
        weights_t = _weight_with_scales(base_weights, list(X_df.columns), dist_s, dens_s, morph_s)
        imputer_t, scaler_t, pca_t, X_weighted_t, X_pca_t, feat_w_t = fit_transform_pipeline_on_inner(
            X_df, fit_mask, weights_t, n_pca_components=n_pca
        )
        X_fit = X_pca_t[fit_mask]
        gmm_t = GaussianMixture(n_components=N_CLUSTERS, covariance_type=cov, random_state=seed)
        gmm_t.fit(X_fit)
        pred_fit = gmm_t.predict(X_fit)
        sil = float(silhouette_score(X_fit, pred_fit)) if len(np.unique(pred_fit)) > 1 else -1.0
        bic = float(gmm_t.bic(X_fit))
        pred_all = gmm_t.predict(X_pca_t)
        prob_all = gmm_t.predict_proba(X_pca_t)
        resolved = resolve_cluster_to_state_from_data(df_eval, pred_all)
        dev_idx, _ = get_cluster_prob_columns(resolved)
        intensity_col = "mean_intensity" if "mean_intensity" in df_eval.columns else ("mean_intensity_img_rel" if "mean_intensity_img_rel" in df_eval.columns else None)
        rho = np.nan
        rho_abs = np.nan
        if intensity_col is not None:
            intensity = pd.to_numeric(df_eval[intensity_col], errors="coerce")
            p_dev = pd.Series(prob_all[:, dev_idx])
            valid = intensity.notna() & p_dev.notna()
            if valid.sum() >= 20:
                rho = float(p_dev.loc[valid].corr(intensity.loc[valid], method="spearman"))
                rho_abs = float(abs(rho))
        y = float(rho_abs) if np.isfinite(rho_abs) else -1.0
        row = {
            "n_pca": n_pca, "covariance_type": cov, "seed": seed,
            "dist_scale": dist_s, "density_scale": dens_s, "morph_scale": morph_s,
            "silhouette": sil, "bic": bic, "spearman_rho": rho, "spearman_abs": rho_abs,
            "resolved_cluster_to_state": str(resolved),
        }
        pack = (imputer_t, scaler_t, pca_t, X_weighted_t, X_pca_t, feat_w_t, gmm_t, row, weights_t)
        return y, row, pack

    def sample_random(n=1):
        return bounds[:, 0] + rng.random((n, bounds.shape[0])) * (bounds[:, 1] - bounds[:, 0])

    records = []
    X_hist = []
    y_hist = []
    best_pack = None
    best_y = -np.inf

    total_trials = BAYES_N_INIT + BAYES_N_ITER
    for t in range(total_trials):
        if t < BAYES_N_INIT or len(X_hist) < BAYES_N_INIT:
            x = sample_random(1)[0]
        else:
            X_train = np.array(X_hist, dtype=float)
            y_train = np.array(y_hist, dtype=float)
            kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X_train.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-4)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_STATE)
            gp.fit(X_train, y_train)
            cand = sample_random(300)
            mu, sigma = gp.predict(cand, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            z = (mu - np.max(y_train)) / sigma
            # EI up to a positive constant
            ei = (mu - np.max(y_train)) * 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0))) + sigma * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            x = cand[int(np.argmax(ei))]

        try:
            y, row, pack = evaluate_vector(x)
            row["trial"] = t
            records.append(row)
            X_hist.append(x.tolist())
            y_hist.append(y)
            if y > best_y:
                best_y = y
                best_pack = pack
        except Exception as e:
            records.append({"trial": t, "error": str(e)})
            continue

    if USE_DE_REFINEMENT:
        def objective(x):
            try:
                y, _, _ = evaluate_vector(x)
                return -y
            except Exception:
                return 1e6
        de_result = differential_evolution(
            objective,
            bounds=[tuple(b) for b in bounds],
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            seed=RANDOM_STATE,
            polish=False,
            workers=1,
        )
        try:
            y_de, row_de, pack_de = evaluate_vector(de_result.x)
            row_de["trial"] = int(total_trials)
            row_de["optimizer"] = "differential_evolution"
            records.append(row_de)
            if y_de > best_y:
                best_y = y_de
                best_pack = pack_de
        except Exception as e:
            records.append({"trial": int(total_trials), "optimizer": "differential_evolution", "error": str(e)})

    if best_pack is None:
        raise RuntimeError("Bayesian tuning failed to produce a valid model.")
    return best_pack, pd.DataFrame(records)


def resolve_cluster_to_state_from_data(df_out: pd.DataFrame, gmm_cluster_raw: np.ndarray) -> dict[int, str]:
    if not AUTO_RESOLVE_CLUSTER_TO_STATE:
        return {int(k): str(v) for k, v in CLUSTER_TO_STATE.items()}

    cluster_ids = sorted(pd.Series(gmm_cluster_raw).dropna().astype(int).unique().tolist())
    if len(cluster_ids) != 2:
        return {int(k): str(v) for k, v in CLUSTER_TO_STATE.items()}

    work = df_out.copy()
    work["__gmm_cluster_tmp"] = pd.Series(gmm_cluster_raw, index=work.index).astype(int)
    grp = work.groupby("__gmm_cluster_tmp")
    scores = {int(c): 0.0 for c in cluster_ids}

    for col in ["adaptive_nb_eccentricity_mean", "adaptive_nb_aspect_ratio_mean"]:
        if col in work.columns:
            m = grp[col].mean()
            vmin, vmax = float(m.min()), float(m.max())
            span = max(vmax - vmin, 1e-9)
            for c in cluster_ids:
                scores[c] += (float(m.get(c, vmin)) - vmin) / span
    for col in ["adaptive_nb_circularity_mean"]:
        if col in work.columns:
            m = grp[col].mean()
            vmin, vmax = float(m.min()), float(m.max())
            span = max(vmax - vmin, 1e-9)
            for c in cluster_ids:
                scores[c] += (vmax - float(m.get(c, vmax))) / span

    dev_cluster = max(scores, key=scores.get)
    undiff_cluster = [c for c in cluster_ids if c != dev_cluster][0]

    return {int(dev_cluster): "deviated", int(undiff_cluster): "undifferentiated"}


def fit_morphology_aux_model(df: pd.DataFrame):
    """
    Fit a morphology-only GMM on ALL nuclei.
    不使用 density features，避免把 sparse/dense 学成 biology。
    """
    morph_features, morph_log1p_cols, _ = resolve_feature_specs(df, MORPH_FEATURE_SPECS, "df for morphology aux model")

    df_m = df.copy()
    morph_model_cols = []

    for col in morph_features:
        new_col = f"{col}__morph_model"
        if col in morph_log1p_cols:
            df_m[new_col] = safe_log1p(df_m[col])
        else:
            df_m[new_col] = pd.to_numeric(df_m[col], errors="coerce")
        morph_model_cols.append(new_col)

    X_m_df = df_m[morph_model_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_m_df)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_pca = min(MORPH_N_PCA, X_scaled.shape[1], X_scaled.shape[0])
    if n_pca < 2:
        raise ValueError("Not enough samples/features for morphology PCA.")

    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type=MORPH_GMM_COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
    )
    gmm.fit(X_pca)

    cluster_raw = gmm.predict(X_pca)
    prob = gmm.predict_proba(X_pca)

    # Decide which morphology cluster is more deviated-like
    morph_pos_cols = [c for c in MORPH_IRREGULARITY_POS if c in df.columns]
    morph_neg_cols = [c for c in MORPH_IRREGULARITY_NEG if c in df.columns]
    cluster_scores = {}
    for c in sorted(np.unique(cluster_raw)):
        sub = df.loc[cluster_raw == c]

        pos_score = 0.0
        neg_score = 0.0

        for col in morph_pos_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            pos_score += float(vals.mean())

        for col in morph_neg_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            neg_score += float(vals.mean())

        cluster_scores[c] = pos_score - neg_score

    morph_dev_cluster = max(cluster_scores, key=cluster_scores.get)
    morph_undiff_cluster = min(cluster_scores, key=cluster_scores.get)

    p_dev = prob[:, morph_dev_cluster]
    p_undiff = prob[:, morph_undiff_cluster]

    out = pd.DataFrame({
        "morph_cluster_raw": cluster_raw,
        "morph_prob_cluster0": prob[:, 0],
        "morph_prob_cluster1": prob[:, 1],
        "morph_dev_cluster": morph_dev_cluster,
        "morph_undiff_cluster": morph_undiff_cluster,
        "morph_prob_deviated": p_dev,
        "morph_prob_undifferentiated": p_undiff,
        "morph_prob_margin_dev_minus_undiff": p_dev - p_undiff,
    })

    info = {
        "morph_model_cols": morph_model_cols,
        "morph_irregularity_pos_cols": morph_pos_cols,
        "morph_irregularity_neg_cols": morph_neg_cols,
        "cluster_scores": cluster_scores,
        "morph_dev_cluster": morph_dev_cluster,
        "morph_undiff_cluster": morph_undiff_cluster,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }
    return out, info


def assign_state_labels(
    gmm_prob: np.ndarray,
    cluster_to_state: dict[int, str] | None = None,
    morph_prob_deviated: np.ndarray | None = None,
):
    """
    Main model decides the global pseudo-label first.
    Morphology is NOT used here as a global hard gate anymore.
    这里 morphology 不再参与全局 red/green 的硬判定，
    否则很容易把所有 red 都压没。
    """
    c2s = cluster_to_state if cluster_to_state is not None else CLUSTER_TO_STATE
    dev_idx, undiff_idx = get_cluster_prob_columns(c2s)

    p_dev_raw = gmm_prob[:, dev_idx]
    p_undiff_raw = gmm_prob[:, undiff_idx]
    margin_raw = p_dev_raw - p_undiff_raw

    final_labels = []
    if morph_prob_deviated is None:
        morph_prob_deviated = np.full_like(p_dev_raw, np.nan, dtype=float)

    for pdv, pun, pmdev, mrg in zip(p_dev_raw, p_undiff_raw, morph_prob_deviated, margin_raw):
        if pdv >= FULL_DEV_MIN:
            final_labels.append("deviated")
        elif pun >= FULL_UNDIFF_MIN:
            final_labels.append("undifferentiated")
        elif (
            (pdv >= UNCERTAIN_TO_DEV_MAIN_MIN)
            and (mrg >= UNCERTAIN_TO_DEV_MARGIN_MIN)
            and np.isfinite(pmdev)
            and (pmdev >= UNCERTAIN_TO_DEV_MORPH_MIN)
        ):
            # Rescue obvious deviated-like nuclei that conservative thresholds put into uncertain.
            final_labels.append("deviated")
        else:
            final_labels.append("uncertain")

    return final_labels, p_dev_raw, p_undiff_raw, margin_raw


def apply_edge_correction(df_out: pd.DataFrame):
    """
    Edge is only a weak veto for suspicious red calls.
    新逻辑:
    - 不再因为在 edge 就压 red
    - 只在 edge 且 red 看起来明显是 density-driven 时，才 red -> uncertain
    - 暂时不主动把 edge green 改成 uncertain
    """
    require_columns(
        df_out,
        [
            "final_state_label_before_edge",
            "gmm_prob_deviated_raw",
            "gmm_prob_undifferentiated_raw",
            "morph_prob_deviated",
            "morph_prob_undifferentiated",
            "is_edge_band",
        ],
        "df_out",
    )

    final_after = []
    reasons = []
    changed = []

    for _, row in df_out.iterrows():
        before = str(row["final_state_label_before_edge"])
        is_edge = bool(row["is_edge_band"])

        p_dev = float(row["gmm_prob_deviated_raw"])
        p_undiff = float(row["gmm_prob_undifferentiated_raw"])
        pm_dev = float(row["morph_prob_deviated"])
        _pm_undiff = float(row["morph_prob_undifferentiated"])

        after = before
        reason = "unchanged"

        # Non-edge nuclei: keep unchanged
        if not is_edge:
            final_after.append(after)
            reasons.append(reason)
            changed.append(False)
            continue

        # -------------------------
        # Case 1: edge red
        # -------------------------
        if before == "deviated":
            # Keep edge red only when both models support it strongly.
            if (p_dev >= EDGE_DEV_KEEP_MAIN_MIN) and (pm_dev >= EDGE_DEV_KEEP_MORPH_MIN):
                after = "deviated"
                reason = "edge_red_kept_dual_strong"

            # If both models lean away from deviated, relabel to undifferentiated.
            elif (p_undiff >= EDGE_UNDIFF_PROMOTE_MAIN_MIN) and (pm_dev <= EDGE_UNDIFF_PROMOTE_MORPH_MAX):
                after = "undifferentiated"
                reason = "edge_red_relabel_undiff_dual_support"

            # Likely density/edge-driven false red => downgrade to uncertain.
            elif (p_dev <= EDGE_DEV_DOWNGRADE_MAIN_MAX) and (pm_dev <= EDGE_DEV_DOWNGRADE_MORPH_MAX):
                after = "uncertain"
                reason = "edge_red_downgraded_density_like"

            else:
                after = "uncertain"
                reason = "edge_red_downgraded_intermediate"

        # -------------------------
        # Case 2: edge green
        # -------------------------
        elif before == "undifferentiated":
            # Keep green by default for now
            after = "undifferentiated"
            reason = "edge_green_kept"

        # -------------------------
        # Case 3: edge uncertain
        # -------------------------
        else:
            after = "uncertain"
            reason = "edge_uncertain_kept"

        final_after.append(after)
        reasons.append(reason)
        changed.append(after != before)

    return final_after, reasons, np.asarray(changed, dtype=bool)


def export_review_masks_and_overlays(
    df_out: pd.DataFrame,
    label_col: str,
    overlay_dir: Path,
    mask_dir: Path,
    file_tag: str,
):
    done = 0
    failed = 0
    skipped = 0

    require_columns(df_out, ["image_name", "label", label_col], "df_out")

    for image_name, sub in df_out.groupby("image_name"):
        try:
            img_path = find_matching_image(image_name)
            orig_mask_path = find_matching_orig_mask(image_name)
            qc_keep_mask_path = find_matching_qc_keep_mask(image_name)

            if img_path is None:
                print(f"[review skip] {image_name} -> image not found")
                skipped += 1
                continue
            if orig_mask_path is None:
                print(f"[review skip] {image_name} -> original mask not found")
                skipped += 1
                continue
            if qc_keep_mask_path is None:
                print(f"[review skip] {image_name} -> qc_keep_mask not found")
                skipped += 1
                continue

            img_gray = ensure_2d_image(tiff.imread(img_path))
            orig_mask = np.load(orig_mask_path)
            qc_keep_mask = np.load(qc_keep_mask_path)

            if orig_mask.ndim != 2 or qc_keep_mask.ndim != 2:
                raise ValueError("orig_mask or qc_keep_mask is not 2D")
            if img_gray.shape != orig_mask.shape:
                raise ValueError(
                    f"Shape mismatch: image={img_gray.shape}, orig_mask={orig_mask.shape}, image={image_name}"
                )
            if qc_keep_mask.shape != orig_mask.shape:
                raise ValueError(
                    f"Shape mismatch: qc_keep_mask={qc_keep_mask.shape}, orig_mask={orig_mask.shape}, image={image_name}"
                )

            keep_binary = qc_keep_mask > 0

            deviated_labels = set(pd.to_numeric(
                sub.loc[sub[label_col] == "deviated", "label"],
                errors="coerce"
            ).dropna().astype(int).tolist())

            undiff_labels = set(pd.to_numeric(
                sub.loc[sub[label_col] == "undifferentiated", "label"],
                errors="coerce"
            ).dropna().astype(int).tolist())

            uncertain_labels = set(pd.to_numeric(
                sub.loc[sub[label_col] == "uncertain", "label"],
                errors="coerce"
            ).dropna().astype(int).tolist())

            deviated_mask = np.where(np.isin(orig_mask, list(deviated_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
            undiff_mask = np.where(np.isin(orig_mask, list(undiff_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
            uncertain_mask = np.where(np.isin(orig_mask, list(uncertain_labels)) & keep_binary, orig_mask, 0).astype(np.int32)

            if SAVE_REVIEW_MASKS:
                np.save(mask_dir / f"{image_name}_{file_tag}_deviated_mask.npy", relabel_compact(deviated_mask))
                np.save(mask_dir / f"{image_name}_{file_tag}_undifferentiated_mask.npy", relabel_compact(undiff_mask))
                np.save(mask_dir / f"{image_name}_{file_tag}_uncertain_mask.npy", relabel_compact(uncertain_mask))

            if SAVE_REVIEW_OVERLAYS:
                overlay = make_label_overlay(
                    img_gray=img_gray,
                    deviated_mask=deviated_mask,
                    undiff_mask=undiff_mask,
                    uncertain_mask=uncertain_mask,
                )
                tiff.imwrite(str(overlay_dir / f"{image_name}_{file_tag}_label_overlay.tif"), overlay)

            done += 1

        except Exception as e:
            print(f"[review fail] {image_name}: {e}")
            failed += 1

    return done, failed, skipped


def choose_preview_images(df_out: pd.DataFrame, label_col: str) -> list[str]:
    image_order = []
    for image_name, sub in df_out.groupby("image_name"):
        labs = set(sub[label_col].astype(str).tolist())
        score = int("deviated" in labs) + int("undifferentiated" in labs) + int("uncertain" in labs)
        image_order.append((image_name, score, len(sub)))
    image_order.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return [x[0] for x in image_order[:PREVIEW_MAX_IMAGES]]


def show_final_previews_in_python(df_out: pd.DataFrame):
    if not SHOW_PREVIEW_AFTER_RUN:
        return

    image_names = choose_preview_images(df_out, "final_state_label")
    if len(image_names) == 0:
        print("No preview images selected for display.")
        return

    valid_names = []
    overlay_paths = []

    for name in image_names:
        overlay_path = REVIEW_OVERLAY_DIR / f"{name}_final_label_label_overlay.tif"
        if overlay_path.exists():
            valid_names.append(name)
            overlay_paths.append(overlay_path)

    if len(valid_names) == 0:
        print("No final review overlay tif found to display.")
        return

    n = len(valid_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 7 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (name, overlay_path) in enumerate(zip(valid_names, overlay_paths)):
        axes[i].imshow(tiff.imread(overlay_path))
        axes[i].set_title(f"{name}\nFinal labels")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show(block=True)


def cleanup_compact_outputs(out_dir: Path) -> None:
    if not COMPACT_OUTPUT_ONLY:
        return
    keep_names = {
        "umap2d_clusters.png",
        "umap3d_clusters.png",
        "run_info.txt",
        "run_info.json",
    }
    for p in out_dir.iterdir():
        if p.is_file() and p.name not in keep_names:
            try:
                p.unlink()
            except Exception:
                pass


# =========================
# Main
# =========================
def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df = add_feature_aliases(df)
    df = augment_with_intensity_features(df, INTENSITY_CSV)
    if len(df) == 0:
        raise RuntimeError("Input CSV is empty.")

    # -------------------------
    # Meta + edge geometry first
    # -------------------------
    keep_meta_cols = [c for c in PREFERRED_META_COLS if c in df.columns]
    df_meta = df[keep_meta_cols].copy()

    edge_df = compute_edge_metrics_for_all_rows(df_meta)
    if len(edge_df) != len(df_meta):
        raise RuntimeError(f"edge_df row count mismatch: {len(edge_df)} vs {len(df_meta)}")
    if not np.array_equal(edge_df["row_index"].to_numpy(), np.arange(len(df_meta))):
        raise RuntimeError("edge_df row_index does not align with input row order")
    print("[meta] Edge metric status counts:")
    print(edge_df["edge_metric_status"].value_counts(dropna=False))
    edge_ok = edge_df["edge_metric_status"].astype(str).eq("ok")
    if edge_ok.any():
        px_mean = float(pd.to_numeric(edge_df.loc[edge_ok, "pixel_size_mean_um"], errors="coerce").median())
        eqdiam_mean = float(pd.to_numeric(edge_df.loc[edge_ok, "median_eqdiam_um"], errors="coerce").median())
        edge_band_mean = float(pd.to_numeric(edge_df.loc[edge_ok, "edge_band_um"], errors="coerce").median())
        inner_fit_mean = float(pd.to_numeric(edge_df.loc[edge_ok, "inner_fit_cutoff_um"], errors="coerce").median())
        print(
            "[meta] Edge geometry in physical units: "
            f"pixel_size_mean_um~{px_mean:.6f}, "
            f"median_eqdiam_um~{eqdiam_mean:.6f}, "
            f"edge_band_um~{edge_band_mean:.6f}, "
            f"inner_fit_cutoff_um~{inner_fit_mean:.6f}"
        )
    else:
        print("[meta] Edge geometry physical-unit summary unavailable because no rows had edge_metric_status=ok")

    # Optional edge-aware feature adjustment (disabled by default when recovering baseline model behavior).
    if USE_EDGE_ADJUSTED_FEATURES:
        df = add_edge_adjusted_feature_columns(df, edge_df, n_bins=10)

    features, log1p_cols, feature_weights = resolve_feature_specs(df, FEATURE_SPECS, "INPUT_CSV")
    print("[info] Resolved main model features:")
    print(features)
    print("[info] Resolved main log1p features:")
    print(sorted(log1p_cols))

    fit_mask = make_inner_fit_mask(edge_df)

    # -------------------------
    # Main model feature transform
    # -------------------------
    df_work = df.copy()
    transformed_feature_cols = []

    for col in features:
        new_col = f"{col}__model"
        if col in log1p_cols:
            df_work[new_col] = safe_log1p(df_work[col])
        else:
            df_work[new_col] = pd.to_numeric(df_work[col], errors="coerce")
        transformed_feature_cols.append(new_col)

    X_df = df_work[transformed_feature_cols].copy()
    if USE_BAYESIAN_TUNING:
        best_pack, gmm_selection_df = bayesian_optimize_spearman(
            X_df=X_df,
            fit_mask=fit_mask,
            df_eval=df,
            base_weights=feature_weights,
        )
        imputer, scaler, pca, X_weighted_all, X_pca_all, feature_weight_vector, gmm, best_row, tuned_weights = best_pack
        feature_weights = tuned_weights
        print("[info] Bayesian model-selection best trial:")
        print(pd.Series(best_row).to_string())
    else:
        trial_rows = []
        best_trial = None
        for n_pca_try in PCA_COMPONENT_CANDIDATES:
            imputer_t, scaler_t, pca_t, X_weighted_all_t, X_pca_all_t, feature_weight_vector_t = fit_transform_pipeline_on_inner(
                X_df,
                fit_mask,
                feature_weights,
                n_pca_components=n_pca_try,
            )
            gmm_t, gmm_sel_t = fit_best_main_gmm(X_pca_all_t, fit_mask, df)
            gmm_sel_t = gmm_sel_t.copy()
            gmm_sel_t["n_pca"] = int(pca_t.n_components_)
            gmm_sel_t["pca_try"] = int(n_pca_try)
            trial_rows.append(gmm_sel_t)
            top = gmm_sel_t.sort_values(
                ["spearman_abs", "silhouette", "bic"],
                ascending=[False, False, True],
                na_position="last",
            ).iloc[0]
            score_key = (float(top["spearman_abs"]), float(top["silhouette"]), -float(top["bic"])) if np.isfinite(top.get("spearman_abs", np.nan)) else (float(top["silhouette"]), -float(top["bic"]))
            if best_trial is None or score_key > best_trial["score_key"]:
                best_trial = {
                    "score_key": score_key,
                    "imputer": imputer_t,
                    "scaler": scaler_t,
                    "pca": pca_t,
                    "X_weighted_all": X_weighted_all_t,
                    "X_pca_all": X_pca_all_t,
                    "feature_weight_vector": feature_weight_vector_t,
                    "gmm": gmm_t,
                }
        if best_trial is None:
            raise RuntimeError("No valid PCA/GMM trial found.")
        imputer = best_trial["imputer"]
        scaler = best_trial["scaler"]
        pca = best_trial["pca"]
        X_weighted_all = best_trial["X_weighted_all"]
        X_pca_all = best_trial["X_pca_all"]
        feature_weight_vector = best_trial["feature_weight_vector"]
        gmm = best_trial["gmm"]
        gmm_selection_df = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()

    print("[info] Main model-selection top candidates:")
    print(
        gmm_selection_df.sort_values(["spearman_abs", "silhouette", "bic"], ascending=[False, False, True], na_position="last")
        .head(8)
        .to_string(index=False)
    )

    # -------------------------
    # Main GMM fit on inner only, predict all
    # -------------------------
    gmm_cluster_raw = gmm.predict(X_pca_all)
    gmm_prob = gmm.predict_proba(X_pca_all)
    gmm_max_prob = gmm_prob.max(axis=1)
    raw_gmm_label = [f"cluster_{c}" for c in gmm_cluster_raw]
    resolved_cluster_to_state = resolve_cluster_to_state_from_data(df, gmm_cluster_raw)
    print(f"[info] Resolved cluster->state mapping: {resolved_cluster_to_state}")

    # -------------------------
    # Morphology auxiliary model for edge correction
    # -------------------------
    morph_aux_df = None
    try:
        morph_aux_df, morph_info = fit_morphology_aux_model(df)
        print(
            "[info] Morph aux model ready: "
            f"dev_cluster={morph_info['morph_dev_cluster']}, "
            f"undiff_cluster={morph_info['morph_undiff_cluster']}"
        )
    except Exception as e:
        print(f"[warn] Morph aux model failed; edge correction will use fallback probabilities. reason={e}")

    # -------------------------
    # KMeans reference fit on inner only, predict all
    # -------------------------
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    kmeans.fit(X_pca_all[fit_mask])
    kmeans_cluster = kmeans.predict(X_pca_all)
    kmeans_label = [f"cluster_{c}" for c in kmeans_cluster]

    # -------------------------
    # Final labels
    # -------------------------
    final_labels, p_dev_raw, p_undiff_raw, margin_raw = assign_state_labels(
        gmm_prob=gmm_prob,
        cluster_to_state=resolved_cluster_to_state,
        morph_prob_deviated=(morph_aux_df["morph_prob_deviated"].to_numpy() if morph_aux_df is not None and "morph_prob_deviated" in morph_aux_df.columns else None),
    )

    # -------------------------
    # Optional UMAP on all nuclei
    # -------------------------
    umap_embed = None
    umap3_embed = None
    if MAKE_UMAP and HAS_UMAP:
        if MAKE_UMAP_2D:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                random_state=UMAP_RANDOM_STATE,
            )
            X_umap = reducer.fit_transform(X_weighted_all)
            umap_embed = pd.DataFrame({"umap_1": X_umap[:, 0], "umap_2": X_umap[:, 1]})
            if not COMPACT_OUTPUT_ONLY:
                umap_embed.to_csv(OUT_UMAP_EMBEDDING, index=False)
                umap_embed.to_csv(OUT_UMAP2_EMBEDDING, index=False)
                print(f"[saved] {OUT_UMAP_EMBEDDING}")
                print(f"[saved] {OUT_UMAP2_EMBEDDING}")

        if MAKE_UMAP_3D:
            reducer3 = umap.UMAP(
                n_components=3,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                random_state=UMAP_RANDOM_STATE,
            )
            X_umap3 = reducer3.fit_transform(X_weighted_all)
            umap3_embed = pd.DataFrame({"umap3_1": X_umap3[:, 0], "umap3_2": X_umap3[:, 1], "umap3_3": X_umap3[:, 2]})
            if not COMPACT_OUTPUT_ONLY:
                umap3_embed.to_csv(OUT_UMAP3_EMBEDDING, index=False)
                print(f"[saved] {OUT_UMAP3_EMBEDDING}")
    else:
        print("[skip] UMAP not available or disabled.")

    # -------------------------
    # Assemble output
    # -------------------------
    df_out = df.copy()

    for col in transformed_feature_cols:
        df_out[col] = df_work[col]

    df_out["pca_1"] = X_pca_all[:, 0]
    df_out["pca_2"] = X_pca_all[:, 1]

    if umap_embed is not None:
        df_out["umap_1"] = umap_embed["umap_1"]
        df_out["umap_2"] = umap_embed["umap_2"]
        df_out["umap2_1"] = umap_embed["umap_1"]
        df_out["umap2_2"] = umap_embed["umap_2"]
    if umap3_embed is not None:
        df_out["umap3_1"] = umap3_embed["umap3_1"]
        df_out["umap3_2"] = umap3_embed["umap3_2"]
        df_out["umap3_3"] = umap3_embed["umap3_3"]

    for col in [
        "boundary_distance_px",
        "boundary_distance_um",
        "boundary_distance_min_px",
        "boundary_distance_min_um",
        "edge_band_px",
        "edge_band_um",
        "inner_fit_cutoff_px",
        "inner_fit_cutoff_um",
        "median_eqdiam_px",
        "median_eqdiam_um",
        "colony_support_dilate_px",
        "colony_support_dilate_um",
        "pixel_size_row_um",
        "pixel_size_col_um",
        "pixel_size_mean_um",
        "is_edge_band",
        "is_inner_fit",
        "edge_metric_status",
    ]:
        df_out[col] = edge_df[col].to_numpy()

    df_out["fit_used_for_model"] = fit_mask

    df_out["gmm_cluster_raw"] = gmm_cluster_raw
    df_out["gmm_label"] = raw_gmm_label
    df_out["gmm_max_prob"] = gmm_max_prob
    df_out["gmm_prob_cluster0"] = gmm_prob[:, 0]
    df_out["gmm_prob_cluster1"] = gmm_prob[:, 1]
    df_out["gmm_prob_deviated_raw"] = p_dev_raw
    df_out["deviated_score"] = p_dev_raw
    df_out["gmm_prob_undifferentiated_raw"] = p_undiff_raw
    df_out["gmm_prob_margin_dev_minus_undiff_raw"] = margin_raw

    if morph_aux_df is not None and len(morph_aux_df) == len(df_out):
        for col in [
            "morph_cluster_raw",
            "morph_prob_cluster0",
            "morph_prob_cluster1",
            "morph_dev_cluster",
            "morph_undiff_cluster",
            "morph_prob_deviated",
            "morph_prob_undifferentiated",
            "morph_prob_margin_dev_minus_undiff",
        ]:
            if col in morph_aux_df.columns:
                df_out[col] = morph_aux_df[col].to_numpy()
    else:
        df_out["morph_prob_deviated"] = df_out["gmm_prob_deviated_raw"]
        df_out["morph_prob_undifferentiated"] = df_out["gmm_prob_undifferentiated_raw"]
        df_out["morph_prob_margin_dev_minus_undiff"] = (
            df_out["morph_prob_deviated"] - df_out["morph_prob_undifferentiated"]
        )

    df_out["final_state_label_before_edge"] = final_labels
    if EDGE_AWARE_ENABLED:
        final_after, edge_reasons, edge_changed = apply_edge_correction(df_out)
        df_out["final_state_label"] = final_after
        df_out["edge_correction_reason"] = edge_reasons
        df_out["edge_label_changed"] = edge_changed
    else:
        df_out["final_state_label"] = final_labels
        df_out["edge_correction_reason"] = "edge_disabled"
        df_out["edge_label_changed"] = False

    df_out["kmeans_cluster_raw"] = kmeans_cluster
    df_out["kmeans_label"] = kmeans_label

    # -------------------------
    # Save outputs
    # -------------------------
    if not COMPACT_OUTPUT_ONLY:
        df_out.to_csv(OUT_CLUSTERED_CSV, index=False)
        print(f"[saved] {OUT_CLUSTERED_CSV}")

    pca_save_cols = keep_meta_cols + [
        "pca_1", "pca_2",
        "gmm_label",
        "fit_used_for_model",
        "is_edge_band",
        "is_inner_fit",
        "final_state_label",
    ]
    if umap_embed is not None:
        pca_save_cols += ["umap_1", "umap_2"]
    if umap3_embed is not None:
        pca_save_cols += ["umap3_1", "umap3_2", "umap3_3"]
    if not COMPACT_OUTPUT_ONLY:
        df_out[pca_save_cols].to_csv(OUT_PCA_EMBEDDING, index=False)
        print(f"[saved] {OUT_PCA_EMBEDDING}")

    if not COMPACT_OUTPUT_ONLY:
        summary_raw = make_group_summary(df_out, features, "gmm_label")
        summary_raw.to_csv(OUT_FEATURE_SUMMARY_RAW, index=False)
        print(f"[saved] {OUT_FEATURE_SUMMARY_RAW}")

    if not COMPACT_OUTPUT_ONLY:
        summary_final = make_group_summary(df_out, features, "final_state_label")
        summary_final.to_csv(OUT_FEATURE_SUMMARY, index=False)
        print(f"[saved] {OUT_FEATURE_SUMMARY}")

    review_cols = keep_meta_cols + features + [
        "pca_1", "pca_2",
        "gmm_cluster_raw",
        "gmm_prob_cluster0", "gmm_prob_cluster1",
        "gmm_prob_deviated_raw", "gmm_prob_undifferentiated_raw",
        "gmm_prob_margin_dev_minus_undiff_raw",
        "gmm_max_prob",
        "gmm_label",
        "boundary_distance_px", "boundary_distance_um",
        "boundary_distance_min_px", "boundary_distance_min_um",
        "edge_band_px", "edge_band_um",
        "inner_fit_cutoff_px", "inner_fit_cutoff_um",
        "median_eqdiam_px", "median_eqdiam_um",
        "colony_support_dilate_px", "colony_support_dilate_um",
        "pixel_size_row_um", "pixel_size_col_um", "pixel_size_mean_um",
        "is_edge_band", "is_inner_fit", "fit_used_for_model", "edge_metric_status",
        "final_state_label",
        "kmeans_cluster_raw", "kmeans_label",
    ]
    if umap_embed is not None:
        review_cols += ["umap_1", "umap_2", "umap2_1", "umap2_2"]
    if umap3_embed is not None:
        review_cols += ["umap3_1", "umap3_2", "umap3_3"]

    if not COMPACT_OUTPUT_ONLY:
        review_df = sample_for_review(df_out[review_cols], "final_state_label", N_REVIEW_PER_CLUSTER, seed=RANDOM_STATE)
        review_df.to_csv(OUT_REVIEW_SAMPLES, index=False)
        print(f"[saved] {OUT_REVIEW_SAMPLES}")

    if SAVE_PCA_FIGURES:
        save_scatter_figure(
            df_out,
            "pca_1",
            "pca_2",
            "gmm_label",
            "PCA scatter colored by raw GMM label",
            OUT_FIG_PCA_RAW,
        )
        print(f"[saved] {OUT_FIG_PCA_RAW}")

        save_scatter_figure(
            df_out,
            "pca_1",
            "pca_2",
            "final_state_label",
            f"PCA scatter final labels ({LABELING_MODE})",
            OUT_FIG_PCA,
        )
        print(f"[saved] {OUT_FIG_PCA}")

    if umap_embed is not None:
        if not COMPACT_OUTPUT_ONLY:
            save_scatter_figure(
                df_out,
                "umap_1",
                "umap_2",
                "gmm_label",
                "UMAP scatter colored by raw GMM label",
                OUT_FIG_UMAP_RAW,
            )
            print(f"[saved] {OUT_FIG_UMAP_RAW}")
            save_scatter_figure(
                df_out,
                "umap2_1",
                "umap2_2",
                "gmm_label",
                "UMAP 2D scatter colored by raw GMM label",
                OUT_FIG_UMAP2_RAW,
            )
            print(f"[saved] {OUT_FIG_UMAP2_RAW}")

        save_scatter_figure(
            df_out,
            "umap_1",
            "umap_2",
            "final_state_label",
            f"UMAP scatter final labels ({LABELING_MODE})",
            OUT_FIG_UMAP,
        )
        print(f"[saved] {OUT_FIG_UMAP}")
        save_scatter_figure(
            df_out,
            "umap2_1",
            "umap2_2",
            "final_state_label",
            f"UMAP 2D scatter final labels ({LABELING_MODE})",
            OUT_FIG_UMAP2,
        )
        print(f"[saved] {OUT_FIG_UMAP2}")

    if umap3_embed is not None:
        if not COMPACT_OUTPUT_ONLY:
            save_scatter_figure_3d(
                df_out,
                ["umap3_1", "umap3_2", "umap3_3"],
                "gmm_label",
                "UMAP 3D scatter colored by raw GMM label",
                OUT_FIG_UMAP3_RAW,
            )
            print(f"[saved] {OUT_FIG_UMAP3_RAW}")
        save_scatter_figure_3d(
            df_out,
            ["umap3_1", "umap3_2", "umap3_3"],
            "final_state_label",
            f"UMAP 3D scatter final labels ({LABELING_MODE})",
            OUT_FIG_UMAP3,
        )
        print(f"[saved] {OUT_FIG_UMAP3}")

    review_done, review_failed, review_skipped = export_review_masks_and_overlays(
        df_out=df_out,
        label_col="final_state_label",
        overlay_dir=REVIEW_OVERLAY_DIR,
        mask_dir=REVIEW_CLUSTER_MASK_DIR,
        file_tag="final_label",
    )

    deviation_intensity_info = save_deviation_intensity_relation(
        df_out=df_out,
        intensity_csv=INTENSITY_CSV,
        out_csv=OUT_DEVIATION_INTENSITY_CSV,
        out_fig=OUT_DEVIATION_INTENSITY_FIG,
    )
    if deviation_intensity_info is not None:
        rel_df = pd.read_csv(OUT_DEVIATION_INTENSITY_CSV)
        save_deviation_intensity_boxplot(rel_df, OUT_DEVIATION_INTENSITY_BOXPLOT)

    with open(OUT_FEATURE_INFO, "w", encoding="utf-8") as f:
        f.write("Main model features\n")
        f.write("===================\n")
        for c in features:
            f.write(f"{c}\n")

        f.write("\nMain log1p features\n")
        f.write("-------------------\n")
        for c in features:
            if c in log1p_cols:
                f.write(f"{c}\n")

        f.write("\nMain feature weights\n")
        f.write("--------------------\n")
        for c in features:
            f.write(f"{c}: {feature_weights.get(c, 1.0)}\n")

    with open(OUT_RUN_INFO, "w", encoding="utf-8") as f:
        f.write("=== Neighbor-focused clustering + inner-colony fitting ===\n")
        f.write(f"INPUT_CSV = {INPUT_CSV}\n")
        f.write(f"OUT_DIR = {OUT_DIR}\n\n")

        f.write(f"LABELING_MODE = {LABELING_MODE}\n")
        f.write(f"DEVIATED_PROB_MIN = {DEVIATED_PROB_MIN}\n")
        f.write(f"UNDIFF_PROB_MIN = {UNDIFF_PROB_MIN}\n")
        f.write(f"UNCERTAIN_TO_DEV_MAIN_MIN = {UNCERTAIN_TO_DEV_MAIN_MIN}\n")
        f.write(f"UNCERTAIN_TO_DEV_MORPH_MIN = {UNCERTAIN_TO_DEV_MORPH_MIN}\n")
        f.write(f"UNCERTAIN_TO_DEV_MARGIN_MIN = {UNCERTAIN_TO_DEV_MARGIN_MIN}\n")
        f.write(f"CLUSTER_TO_STATE(default) = {CLUSTER_TO_STATE}\n")
        f.write(f"AUTO_RESOLVE_CLUSTER_TO_STATE = {AUTO_RESOLVE_CLUSTER_TO_STATE}\n")
        f.write(f"resolved_cluster_to_state = {resolved_cluster_to_state}\n")
        f.write(f"MAIN_MODEL_SELECTION_OBJECTIVE = {MAIN_MODEL_SELECTION_OBJECTIVE}\n")
        f.write(f"USE_BAYESIAN_TUNING = {USE_BAYESIAN_TUNING}\n")
        f.write(f"USE_DE_REFINEMENT = {USE_DE_REFINEMENT}\n")
        f.write(f"PCA_COMPONENT_CANDIDATES = {PCA_COMPONENT_CANDIDATES}\n")
        f.write(f"GMM_COVARIANCE_CANDIDATES = {GMM_COVARIANCE_CANDIDATES}\n")
        f.write(f"GMM_RANDOM_SEED_CANDIDATES = {GMM_RANDOM_SEED_CANDIDATES}\n")
        f.write(f"selected_gmm_covariance_type = {gmm.covariance_type}\n\n")

        f.write(f"COLONY_SUPPORT_DILATE_EQDIAM_FACTOR = {COLONY_SUPPORT_DILATE_EQDIAM_FACTOR}\n")
        f.write(f"COLONY_SUPPORT_MIN_DILATE_UM = {COLONY_SUPPORT_MIN_DILATE_UM}\n")
        f.write(f"COLONY_SUPPORT_ERODE_RATIO = {COLONY_SUPPORT_ERODE_RATIO}\n")
        f.write(f"EDGE_BAND_EQDIAM_FACTOR = {EDGE_BAND_EQDIAM_FACTOR}\n")
        f.write(f"EDGE_BAND_MIN_UM = {EDGE_BAND_MIN_UM}\n")
        f.write(f"EDGE_DISTANCE_OBJECT_PERCENTILE = {EDGE_DISTANCE_OBJECT_PERCENTILE}\n")
        f.write(f"EDGE_DEV_KEEP_MAIN_MIN = {EDGE_DEV_KEEP_MAIN_MIN}\n")
        f.write(f"EDGE_DEV_KEEP_MORPH_MIN = {EDGE_DEV_KEEP_MORPH_MIN}\n")
        f.write(f"EDGE_DEV_DOWNGRADE_MAIN_MAX = {EDGE_DEV_DOWNGRADE_MAIN_MAX}\n")
        f.write(f"EDGE_DEV_DOWNGRADE_MORPH_MAX = {EDGE_DEV_DOWNGRADE_MORPH_MAX}\n")
        f.write(f"EDGE_UNDIFF_PROMOTE_MAIN_MIN = {EDGE_UNDIFF_PROMOTE_MAIN_MIN}\n")
        f.write(f"EDGE_UNDIFF_PROMOTE_MORPH_MAX = {EDGE_UNDIFF_PROMOTE_MORPH_MAX}\n")
        f.write(f"INNER_FIT_EQDIAM_FACTOR = {INNER_FIT_EQDIAM_FACTOR}\n")
        f.write(f"INNER_FIT_MIN_UM = {INNER_FIT_MIN_UM}\n")
        f.write(f"MIN_INNER_FIT_RATIO = {MIN_INNER_FIT_RATIO}\n")
        f.write(f"MIN_INNER_FIT_ABS = {MIN_INNER_FIT_ABS}\n\n")

        f.write(f"RESOLVED_MAIN_FEATURES = {features}\n")
        f.write(f"RESOLVED_MAIN_LOG1P_FEATURES = {sorted(log1p_cols)}\n")
        f.write(f"RESOLVED_MAIN_FEATURE_WEIGHTS = {feature_weights}\n")
        f.write("\n")

        f.write(f"Total samples = {len(df_out)}\n")
        f.write(f"Inner-fit samples = {int(df_out['fit_used_for_model'].sum())}\n")
        f.write(f"Inner-fit ratio = {float(df_out['fit_used_for_model'].mean()):.4f}\n")
        f.write(f"Main PCA explained variance ratio = {pca.explained_variance_ratio_}\n\n")

        f.write("Raw GMM label counts\n")
        f.write(df_out["gmm_label"].value_counts(dropna=False).to_string())

        f.write("\n\nFinal label counts\n")
        f.write(df_out["final_state_label"].value_counts(dropna=False).to_string())

        f.write("\n\nEdge-band counts\n")
        f.write(df_out["is_edge_band"].value_counts(dropna=False).to_string())

        f.write("\n\nColor legend\n")
        f.write("deviated = red\n")
        f.write("undifferentiated = green\n")
        f.write("uncertain = yellow\n")

        f.write("\nReview overlay export\n")
        f.write(f"done={review_done}, failed={review_failed}, skipped={review_skipped}\n")
        if deviation_intensity_info is not None:
            f.write("\nDeviation score vs mean intensity\n")
            f.write(f"n_valid_rows = {deviation_intensity_info['n_valid_rows']}\n")
            f.write(f"pearson_r = {deviation_intensity_info['pearson_r']}\n")
            f.write(f"spearman_r = {deviation_intensity_info['spearman_r']}\n")
            f.write(f"csv = {deviation_intensity_info['csv']}\n")
            f.write(f"figure = {deviation_intensity_info['figure']}\n")

    print(f"[saved] {OUT_RUN_INFO}")

    run_info_json = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUT_DIR),
        "labeling_mode": LABELING_MODE,
        "n_clusters": N_CLUSTERS,
        "cluster_to_state_default": CLUSTER_TO_STATE,
        "cluster_to_state_resolved": resolved_cluster_to_state,
        "gmm_covariance_candidates": GMM_COVARIANCE_CANDIDATES,
        "gmm_seed_candidates": GMM_RANDOM_SEED_CANDIDATES,
        "gmm_selected_covariance_type": gmm.covariance_type,
        "keep_extended_output_dir": KEEP_EXTENDED_OUTPUT_DIR,
        "feature_columns": features,
        "log1p_columns": sorted(log1p_cols),
        "feature_weights": feature_weights,
        "total_samples": int(len(df_out)),
        "inner_fit_samples": int(df_out["fit_used_for_model"].sum()),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "main_gmm_model_selection": gmm_selection_df.to_dict(orient="records"),
        "outputs": {
            "clustered_csv": str(OUT_CLUSTERED_CSV),
            "feature_summary_raw_csv": str(OUT_FEATURE_SUMMARY_RAW),
            "feature_summary_csv": str(OUT_FEATURE_SUMMARY),
            "review_csv": str(OUT_REVIEW_SAMPLES),
            "pca_embedding_csv": str(OUT_PCA_EMBEDDING),
            "pca_raw_figure": str(OUT_FIG_PCA_RAW) if SAVE_PCA_FIGURES else None,
            "pca_figure": str(OUT_FIG_PCA) if SAVE_PCA_FIGURES else None,
            "umap_embedding_csv": str(OUT_UMAP_EMBEDDING) if umap_embed is not None else None,
            "umap_raw_figure": str(OUT_FIG_UMAP_RAW) if umap_embed is not None else None,
            "umap_figure": str(OUT_FIG_UMAP) if umap_embed is not None else None,
            "umap2d_embedding_csv": str(OUT_UMAP2_EMBEDDING) if umap_embed is not None else None,
            "umap2d_raw_figure": str(OUT_FIG_UMAP2_RAW) if umap_embed is not None else None,
            "umap2d_figure": str(OUT_FIG_UMAP2) if umap_embed is not None else None,
            "umap3d_embedding_csv": str(OUT_UMAP3_EMBEDDING) if umap3_embed is not None else None,
            "umap3d_raw_figure": str(OUT_FIG_UMAP3_RAW) if umap3_embed is not None else None,
            "umap3d_figure": str(OUT_FIG_UMAP3) if umap3_embed is not None else None,
            "deviation_intensity_csv": str(OUT_DEVIATION_INTENSITY_CSV) if deviation_intensity_info is not None else None,
            "deviation_intensity_figure": str(OUT_DEVIATION_INTENSITY_FIG) if deviation_intensity_info is not None else None,
        },
    }
    with open(OUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info_json, f, ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_RUN_INFO_JSON}")

    if MINIMAL_OUTPUT_MODE:
        df_out.to_csv(FINAL_CLUSTERED_CSV, index=False, encoding="utf-8-sig")
        preferred_fig = OUT_FIG_UMAP2 if OUT_FIG_UMAP2.exists() else (OUT_FIG_UMAP if OUT_FIG_UMAP.exists() else None)
        if preferred_fig is not None and preferred_fig.exists():
            shutil.copy2(preferred_fig, FINAL_CLUSTER_FIG)
            print(f"[saved] Final cluster figure: {FINAL_CLUSTER_FIG}")
        if not KEEP_EXTENDED_OUTPUT_DIR:
            shutil.rmtree(OUT_DIR, ignore_errors=True)
        print(f"[saved] Final clustered nucleus CSV: {FINAL_CLUSTERED_CSV} (overwritten in place)")

    cleanup_compact_outputs(OUT_DIR)

    print("\n=== Done ===")
    print(f"Total nuclei: {len(df_out)}")
    print(f"Inner-fit nuclei: {int(df_out['fit_used_for_model'].sum())} / {len(df_out)}")

    print("\nFinal state label counts:")
    print(df_out["final_state_label"].value_counts(dropna=False))

    print("\nColor legend:")
    print("deviated = red")
    print("undifferentiated = green")
    print("uncertain = yellow")

    show_final_previews_in_python(df_out)


if __name__ == "__main__":
    main()
