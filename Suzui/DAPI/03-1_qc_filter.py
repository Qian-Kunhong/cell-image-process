from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries
from skimage.morphology import (
    binary_dilation,
    binary_closing,
    remove_small_holes,
    remove_small_objects,
    disk,
)
from skimage.measure import label as cc_label, regionprops

# =========================================================
# 03-2_qc_filter_training.py
# ---------------------------------------------------------
# QC logic for DAPI nuclei masks
#
# Main idea / 当前逻辑:
# 1) remove border-touching objects
# 2) remove objects outside main colony envelope  <-- NEW
# 3) DO NOT remove oversized objects alone by default
# 4) remove oversized + truly isolated objects
# 5) optional: remove extremely isolated objects alone
# 6) isolation is defined by FIXED-RADIUS neighbors
# 7) trace back to original masks
# 8) export qc masks + overlay previews
# 9) directly show preview images in Python after run
# =========================================================


# =========================
# 路径设置 / Path settings
# =========================
mode = globals().get("mode", 1)
SUZUI_ROOT = Path(globals().get("SUZUI_ROOT", r"F:\Suzui"))
ANALYSIS_ROOT = Path(globals().get("ANALYSIS_ROOT", SUZUI_ROOT / "analysis_out"))
DATASET_NAME = globals().get("DATASET_NAME", "A-1-3")
TRAINING_ROOT = Path(globals().get("TRAINING_ROOT", SUZUI_ROOT / "training data"))

DATASETS = {
    1: {
        "name": "data",
        "IMG_DIR": ANALYSIS_ROOT / DATASET_NAME,
        "MASK_DIR": (ANALYSIS_ROOT / DATASET_NAME) / "masks",
        "FEATURE_DIR": (ANALYSIS_ROOT / DATASET_NAME) / "features",
        "message": "processing data",
    },
    2: {
        "name": "SNL_training",
        "IMG_DIR": TRAINING_ROOT / "SNL",
        "MASK_DIR": ANALYSIS_ROOT / "masks_training" / "SNL",
        "FEATURE_DIR": ANALYSIS_ROOT / "features_training" / "SNL",
        "message": "processing SNL training data",
    },
    3: {
        "name": "MEF_training",
        "IMG_DIR": TRAINING_ROOT / "MEF",
        "MASK_DIR": ANALYSIS_ROOT / "masks_training" / "MEF",
        "FEATURE_DIR": ANALYSIS_ROOT / "features_training" / "MEF",
        "message": "processing MEF training data",
    },
}

if mode not in DATASETS:
    raise ValueError(f"Unsupported mode={mode}. Available modes: {list(DATASETS.keys())}")

CFG = DATASETS[mode]
print(CFG["message"])

IMG_DIR = CFG["IMG_DIR"]
MASK_DIR = CFG["MASK_DIR"]
FEATURE_DIR = CFG["FEATURE_DIR"]

NUCLEUS_CSV = FEATURE_DIR / "nucleus_features.csv"
IMAGE_CSV = FEATURE_DIR / "image_features.csv"
NUCLEUS_INTENSITY_CSV = FEATURE_DIR / "nucleus_intensity_features.csv"
IMAGE_INTENSITY_CSV = FEATURE_DIR / "image_intensity_features.csv"

OUT_DIR = FEATURE_DIR / "qc_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_QC_KEEP = OUT_DIR / "nucleus_features_qc.csv"
OUT_QC_REMOVE = OUT_DIR / "nucleus_features_removed.csv"
OUT_IMAGE_SUMMARY = OUT_DIR / "image_qc_summary.csv"
OUT_THRESHOLDS_JSON = OUT_DIR / "qc_thresholds.json"

SAVE_QC_MASKS = True
SAVE_QC_PREVIEWS = True
MINIMAL_OUTPUT_MODE = True
OVERWRITE_MASK_WITH_QC = True
OVERWRITE_NUCLEUS_FEATURE_CSV_WITH_QC_KEEP = True
KEEP_REMOVED_OBJECTS_CSV = False
KEEP_IMAGE_SUMMARY_CSV = False
KEEP_THRESHOLDS_JSON = False

if MINIMAL_OUTPUT_MODE:
    SAVE_QC_MASKS = False
    SAVE_QC_PREVIEWS = False

QC_KEEP_MASK_DIR = OUT_DIR / "qc_keep_masks"
QC_REMOVE_MASK_DIR = OUT_DIR / "qc_removed_masks"
QC_PREVIEW_DIR = OUT_DIR / "qc_preview_tif"

if SAVE_QC_MASKS:
    QC_KEEP_MASK_DIR.mkdir(parents=True, exist_ok=True)
    QC_REMOVE_MASK_DIR.mkdir(parents=True, exist_ok=True)

if SAVE_QC_PREVIEWS:
    QC_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

MASK_SUFFIX = "_mask.npy"
IMG_EXTENSIONS = [".tif", ".tiff"]

NUCLEUS_KEY_COLS = ["image_name", "label"]
IMAGE_KEY_COLS = ["image_name"]
BRIGHTNESS_DERIVED_COLS = {
    "is_bright",
    "is_bright_mean",
    "is_bright_range",
}


# =========================
# 参数设置 / Parameters
# =========================

# ---- 1) Border filtering / 边缘对象删除 ----
EXCLUDE_BORDER_OBJECTS = True
BORDER_MARGIN_PX = 1


# ---- 2) Outside-colony filtering / 主群体外侧对象删除 ----
# 这是这次新增的关键逻辑：
# 不再只靠 oversized/isolation 去猜，而是直接定义主群体外侧对象
USE_OUTSIDE_COLONY_FILTER = True
REMOVE_OUTSIDE_COLONY_OBJECTS = True

# 用 mask 构建 colony envelope 时的形态学参数
# 注意：这里的目标是把主群体连起来，但不要把右下角杂片并进去
COLONY_DILATION_RADIUS_PX = 8
COLONY_CLOSING_RADIUS_PX = 12

# 填掉主群体内部小孔洞
COLONY_FILL_HOLES_AREA_PX = 20000

# 去掉太小的连通片
COLONY_MIN_COMPONENT_AREA_PX = 15000

# 是否只保留最大的主连通区域
# 对你这种“一个主要群体 + 外侧一堆杂片”的图，非常有效
COLONY_KEEP_ONLY_LARGEST_COMPONENT = True


# ---- 3) Oversize filtering / 大对象判定 ----
USE_AREA_FILTER = True
USE_EQDIAM_FILTER = True

AREA_UPPER_MAD_K = 6.0
EQDIAM_UPPER_MAD_K = 6.0

OVERSIZE_FALLBACK_HIGH_QUANTILE = 0.999

# "and" 更保守
OVERSIZE_LOGIC = "and"

# 不单独删 oversized
REMOVE_OVERSIZE_ALONE = False


# ---- 4) Isolation filtering / 孤立对象判定 ----
USE_ISOLATION_FILTER = True

FIXED_RADIUS_MODE = "eqdiam_median_x_factor"
FIXED_RADIUS_FACTOR = 2.5

ISOLATED_FIXED_NEIGHBOR_MAX = 1

NN1_UPPER_MAD_K = 4.0
NN1_FALLBACK_HIGH_QUANTILE = 0.995

REMOVE_OVERSIZE_AND_ISOLATED = True


# ---- 4b) Extreme-isolation filtering / 极端孤立对象（可选） ----
USE_EXTREME_ISOLATION_FILTER = True
EXTREME_ISOLATED_NEIGHBOR_MAX = 0
EXTREME_NN1_UPPER_MAD_K = 6.0
EXTREME_NN1_FALLBACK_HIGH_QUANTILE = 0.999

# 默认先不单独删 extreme isolated
REMOVE_EXTREME_ISOLATED_ALONE = False


# ---- 4c) Brightness filtering / 亮度异常（可选） ----
USE_BRIGHT_FILTER = False
MEAN_INT_UPPER_MAD_K = 6.0
INT_RANGE_UPPER_MAD_K = 6.0
BRIGHT_FALLBACK_HIGH_QUANTILE = 0.999
REMOVE_BRIGHT_AND_ISOLATED = False


# ---- 5) Preview / 运行结束后直接显示图片 ----
SHOW_PREVIEW_AFTER_RUN = True
N_PREVIEW_TO_SHOW = 6
PREVIEW_SELECT_MODE = "highest_removed_fraction"
PREVIEW_IMAGE_NAMES = []
PREVIEW_NCOLS = 2


# ---- 6) Warning threshold / 提示 ----
HIGH_REMOVAL_FRACTION_WARN = 0.20


# =========================
# Optional scipy KDTree
# =========================
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# =========================
# 工具函数 / Utilities
# =========================
def to_py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad)


def robust_upper_threshold(series: pd.Series, k: float, fallback_q: float):
    x = series.to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, "empty"

    med = np.median(x)
    mad = robust_mad(x)

    if np.isfinite(mad) and mad > 0:
        thr = med + k * mad
        return float(thr), "median_plus_kMAD"
    else:
        thr = np.quantile(x, fallback_q)
        return float(thr), f"quantile_{fallback_q}"


def ensure_2d_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        if img.shape[-1] == 1:
            return img[..., 0]
        if img.shape[-1] in (3, 4):
            return img[..., 0]
        if img.shape[0] == 1:
            return img[0]
        if img.shape[0] in (3, 4):
            return img[0]

    raise ValueError(f"Unsupported image shape: {img.shape}")


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def find_matching_image(image_name: str):
    for ext in IMG_EXTENSIONS:
        candidate = IMG_DIR / f"{image_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_matching_mask(image_name: str):
    candidate = MASK_DIR / f"{image_name}{MASK_SUFFIX}"
    if candidate.exists():
        return candidate
    return None


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    return find_boundaries(mask, mode="outer")


def make_qc_preview(img_gray: np.ndarray, keep_mask: np.ndarray, remove_mask: np.ndarray) -> np.ndarray:
    """
    green = kept
    red   = removed
    """
    base = normalize_to_u8(img_gray)
    rgb = np.stack([base, base, base], axis=-1)

    keep_boundary = boundary_from_mask(keep_mask)
    remove_boundary = boundary_from_mask(remove_mask)

    rgb[keep_boundary] = np.array([0, 255, 0], dtype=np.uint8)
    rgb[remove_boundary] = np.array([255, 0, 0], dtype=np.uint8)

    return rgb


def relabel_compact(mask: np.ndarray) -> np.ndarray:
    labels = np.unique(mask)
    labels = labels[labels > 0]

    out = np.zeros_like(mask, dtype=np.int32)
    for new_label, old_label in enumerate(labels, start=1):
        out[mask == old_label] = new_label
    return out


def is_intensity_related_column(col: str) -> bool:
    cl = str(col).lower().strip()
    return ("intensity" in cl) or (cl in BRIGHTNESS_DERIVED_COLS)


def split_feature_and_intensity_tables(
    df: pd.DataFrame,
    key_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    intensity_cols = [c for c in df.columns if c not in key_cols and is_intensity_related_column(c)]
    feature_cols = [c for c in df.columns if c not in intensity_cols]
    intensity_table_cols = [c for c in key_cols if c in df.columns] + intensity_cols
    return (
        df.loc[:, feature_cols].copy(),
        df.loc[:, intensity_table_cols].copy(),
        intensity_cols,
    )


def merge_intensity_table(
    feature_df: pd.DataFrame,
    intensity_csv: Path,
    key_cols: list[str],
    table_name: str,
) -> pd.DataFrame:
    if not intensity_csv.exists():
        return feature_df

    intensity_df = pd.read_csv(intensity_csv)
    merge_keys = [c for c in key_cols if c in feature_df.columns and c in intensity_df.columns]
    if len(merge_keys) != len(key_cols):
        raise ValueError(f"{table_name} is missing required merge keys: {key_cols}")

    dup_mask = intensity_df.duplicated(subset=merge_keys, keep=False)
    if dup_mask.any():
        raise ValueError(f"{table_name} has duplicated rows for keys: {merge_keys}")

    value_cols = [c for c in intensity_df.columns if c not in merge_keys]
    if not value_cols:
        return feature_df

    return feature_df.merge(intensity_df, on=merge_keys, how="left")


def add_border_flag(df: pd.DataFrame, image_meta: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    require_columns(
        df,
        ["image_name", "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"],
        "nucleus_features"
    )
    require_columns(
        image_meta,
        ["image_name", "image_height", "image_width"],
        "image_features"
    )

    # If nucleus table already has image size columns, normalize them first.
    if "image_height" in df.columns:
        df["image_height"] = pd.to_numeric(df["image_height"], errors="coerce")
    if "image_width" in df.columns:
        df["image_width"] = pd.to_numeric(df["image_width"], errors="coerce")

    meta = image_meta[["image_name", "image_height", "image_width"]].drop_duplicates()
    df = df.merge(meta, on="image_name", how="left", suffixes=("", "_meta"))

    # Resolve merge suffix columns robustly.
    if "image_height" not in df.columns and "image_height_meta" in df.columns:
        df["image_height"] = df["image_height_meta"]
    elif "image_height" in df.columns and "image_height_meta" in df.columns:
        a = pd.to_numeric(df["image_height"], errors="coerce")
        b = pd.to_numeric(df["image_height_meta"], errors="coerce")
        df["image_height"] = a.where(a.notna(), b)

    if "image_width" not in df.columns and "image_width_meta" in df.columns:
        df["image_width"] = df["image_width_meta"]
    elif "image_width" in df.columns and "image_width_meta" in df.columns:
        a = pd.to_numeric(df["image_width"], errors="coerce")
        b = pd.to_numeric(df["image_width_meta"], errors="coerce")
        df["image_width"] = a.where(a.notna(), b)

    if df["image_height"].isna().any() or df["image_width"].isna().any():
        missing_imgs = df.loc[df["image_height"].isna() | df["image_width"].isna(), "image_name"].unique().tolist()
        raise ValueError(f"Missing image_height/image_width for images: {missing_imgs}")

    df["touches_border"] = (
        (df["bbox_min_row"] <= BORDER_MARGIN_PX) |
        (df["bbox_min_col"] <= BORDER_MARGIN_PX) |
        (df["bbox_max_row"] >= (df["image_height"] - BORDER_MARGIN_PX)) |
        (df["bbox_max_col"] >= (df["image_width"] - BORDER_MARGIN_PX))
    )

    return df


def build_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}

    if USE_AREA_FILTER and "area" in df.columns:
        thr, method = robust_upper_threshold(
            df["area"],
            AREA_UPPER_MAD_K,
            OVERSIZE_FALLBACK_HIGH_QUANTILE
        )
        thresholds["area"] = {
            "type": "upper",
            "value": to_py(thr),
            "method": method,
        }

    if USE_EQDIAM_FILTER and "equivalent_diameter" in df.columns:
        thr, method = robust_upper_threshold(
            df["equivalent_diameter"],
            EQDIAM_UPPER_MAD_K,
            OVERSIZE_FALLBACK_HIGH_QUANTILE
        )
        thresholds["equivalent_diameter"] = {
            "type": "upper",
            "value": to_py(thr),
            "method": method,
        }

    thr, method = robust_upper_threshold(
        df["nn1_distance"],
        NN1_UPPER_MAD_K,
        NN1_FALLBACK_HIGH_QUANTILE
    )
    thresholds["nn1_distance"] = {
        "type": "upper",
        "value": to_py(thr),
        "method": method,
    }

    if USE_EXTREME_ISOLATION_FILTER:
        thr, method = robust_upper_threshold(
            df["nn1_distance"],
            EXTREME_NN1_UPPER_MAD_K,
            EXTREME_NN1_FALLBACK_HIGH_QUANTILE
        )
        thresholds["nn1_distance_extreme"] = {
            "type": "upper",
            "value": to_py(thr),
            "method": method,
        }

    if USE_BRIGHT_FILTER and "mean_intensity" in df.columns:
        thr, method = robust_upper_threshold(
            df["mean_intensity"],
            MEAN_INT_UPPER_MAD_K,
            BRIGHT_FALLBACK_HIGH_QUANTILE
        )
        thresholds["mean_intensity"] = {
            "type": "upper",
            "value": to_py(thr),
            "method": method,
        }

    if USE_BRIGHT_FILTER and "intensity_range" in df.columns:
        thr, method = robust_upper_threshold(
            df["intensity_range"],
            INT_RANGE_UPPER_MAD_K,
            BRIGHT_FALLBACK_HIGH_QUANTILE
        )
        thresholds["intensity_range"] = {
            "type": "upper",
            "value": to_py(thr),
            "method": method,
        }

    return thresholds


def compute_fixed_radius_neighbor_count(centroids: np.ndarray, radius: float) -> np.ndarray:
    n = len(centroids)
    if n == 0:
        return np.array([], dtype=int)

    if HAS_SCIPY:
        tree = cKDTree(centroids)
        counts = np.zeros(n, dtype=int)
        for i in range(n):
            idxs = tree.query_ball_point(centroids[i], r=radius)
            counts[i] = len([j for j in idxs if j != i])
        return counts
    else:
        diff = centroids[:, None, :] - centroids[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        counts = ((dist <= radius) & (dist > 0)).sum(axis=1)
        return counts.astype(int)


def add_fixed_radius_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    require_columns(
        df,
        ["image_name", "centroid_row", "centroid_col", "equivalent_diameter"],
        "nucleus_features"
    )

    out_list = []

    for image_name, sub in df.groupby("image_name", sort=False):
        sub = sub.copy()

        if FIXED_RADIUS_MODE == "eqdiam_median_x_factor":
            med_eqdiam = np.nanmedian(sub["equivalent_diameter"].to_numpy(dtype=float))
            radius = med_eqdiam * FIXED_RADIUS_FACTOR
        else:
            raise ValueError(f"Unsupported FIXED_RADIUS_MODE: {FIXED_RADIUS_MODE}")

        coords = sub[["centroid_row", "centroid_col"]].to_numpy(dtype=float)
        fixed_neighbor_count = compute_fixed_radius_neighbor_count(coords, radius)

        sub["fixed_neighbor_radius_px"] = float(radius)
        sub["fixed_neighbor_count"] = fixed_neighbor_count

        out_list.append(sub)

    return pd.concat(out_list, axis=0, ignore_index=True)


# =========================
# NEW: colony envelope logic
# =========================
def build_colony_region_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    从整张分割 mask 构建主群体包络区。
    核心目标：保留主 colony，排除右下角外侧杂片区域。
    """
    fg = mask > 0
    if np.count_nonzero(fg) == 0:
        return np.zeros_like(fg, dtype=bool)

    region = fg.copy()

    if COLONY_DILATION_RADIUS_PX > 0:
        region = binary_dilation(region, footprint=disk(COLONY_DILATION_RADIUS_PX))

    if COLONY_CLOSING_RADIUS_PX > 0:
        region = binary_closing(region, footprint=disk(COLONY_CLOSING_RADIUS_PX))

    if COLONY_FILL_HOLES_AREA_PX > 0:
        region = remove_small_holes(region, area_threshold=COLONY_FILL_HOLES_AREA_PX)

    if COLONY_MIN_COMPONENT_AREA_PX > 0:
        region = remove_small_objects(region, min_size=COLONY_MIN_COMPONENT_AREA_PX)

    if np.count_nonzero(region) == 0:
        # fallback：如果参数太严导致空了，退回到原始 fg 的最大连通区
        lab = cc_label(fg)
        props = regionprops(lab)
        if len(props) == 0:
            return np.zeros_like(fg, dtype=bool)
        largest_label = max(props, key=lambda x: x.area).label
        return lab == largest_label

    if COLONY_KEEP_ONLY_LARGEST_COMPONENT:
        lab = cc_label(region)
        props = regionprops(lab)
        if len(props) == 0:
            return np.zeros_like(region, dtype=bool)
        largest_label = max(props, key=lambda x: x.area).label
        region = lab == largest_label

    return region.astype(bool)


def add_outside_colony_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    给每个对象打 is_outside_colony flag:
    True = 质心位于主 colony 包络区外
    """
    df = df.copy()

    require_columns(
        df,
        ["image_name", "centroid_row", "centroid_col"],
        "nucleus_features"
    )

    out_list = []

    for image_name, sub in df.groupby("image_name", sort=False):
        sub = sub.copy()

        mask_path = find_matching_mask(image_name)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {image_name}")

        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape={mask.shape}, image={image_name}")

        colony_region = build_colony_region_from_mask(mask)
        h, w = colony_region.shape

        outside_flags = []
        for _, row in sub.iterrows():
            rr = int(np.clip(np.round(row["centroid_row"]), 0, h - 1))
            cc = int(np.clip(np.round(row["centroid_col"]), 0, w - 1))
            inside = bool(colony_region[rr, cc])
            outside_flags.append(not inside)

        sub["is_outside_colony"] = outside_flags
        out_list.append(sub)

    return pd.concat(out_list, axis=0, ignore_index=True)


def apply_qc_flags(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    核心判定逻辑：
    - is_oversized
    - is_truly_isolated
    - is_extreme_isolated
    - is_outside_colony
    - qc_exclude / qc_reason
    """
    df = df.copy()

    # ---- oversize flags ----
    df["is_large_area"] = False
    df["is_large_eqdiam"] = False
    df["is_oversized"] = False

    if USE_AREA_FILTER and "area" in thresholds:
        thr = thresholds["area"]["value"]
        df["is_large_area"] = df["area"] > thr

    if USE_EQDIAM_FILTER and "equivalent_diameter" in thresholds:
        thr = thresholds["equivalent_diameter"]["value"]
        df["is_large_eqdiam"] = df["equivalent_diameter"] > thr

    if USE_AREA_FILTER and USE_EQDIAM_FILTER:
        if OVERSIZE_LOGIC.lower() == "and":
            df["is_oversized"] = df["is_large_area"] & df["is_large_eqdiam"]
        elif OVERSIZE_LOGIC.lower() == "or":
            df["is_oversized"] = df["is_large_area"] | df["is_large_eqdiam"]
        else:
            raise ValueError(f"Unsupported OVERSIZE_LOGIC={OVERSIZE_LOGIC}")
    elif USE_AREA_FILTER:
        df["is_oversized"] = df["is_large_area"]
    elif USE_EQDIAM_FILTER:
        df["is_oversized"] = df["is_large_eqdiam"]

    # ---- isolation flags ----
    df["is_low_fixed_neighbor_count"] = False
    df["is_far_nn1"] = False
    df["is_truly_isolated"] = False

    if USE_ISOLATION_FILTER:
        df["is_low_fixed_neighbor_count"] = df["fixed_neighbor_count"] <= ISOLATED_FIXED_NEIGHBOR_MAX
        df["is_far_nn1"] = df["nn1_distance"] > thresholds["nn1_distance"]["value"]
        df["is_truly_isolated"] = df["is_low_fixed_neighbor_count"] & df["is_far_nn1"]

    # ---- extreme isolation flags ----
    df["is_extreme_low_fixed_neighbor_count"] = False
    df["is_extreme_far_nn1"] = False
    df["is_extreme_isolated"] = False

    if USE_EXTREME_ISOLATION_FILTER and "nn1_distance_extreme" in thresholds:
        df["is_extreme_low_fixed_neighbor_count"] = df["fixed_neighbor_count"] <= EXTREME_ISOLATED_NEIGHBOR_MAX
        df["is_extreme_far_nn1"] = df["nn1_distance"] > thresholds["nn1_distance_extreme"]["value"]
        df["is_extreme_isolated"] = df["is_extreme_low_fixed_neighbor_count"] & df["is_extreme_far_nn1"]

    # ---- brightness flags ----
    df["is_bright_mean"] = False
    df["is_bright_range"] = False
    df["is_bright"] = False

    if USE_BRIGHT_FILTER and "mean_intensity" in thresholds:
        df["is_bright_mean"] = df["mean_intensity"] > thresholds["mean_intensity"]["value"]

    if USE_BRIGHT_FILTER and "intensity_range" in thresholds:
        df["is_bright_range"] = df["intensity_range"] > thresholds["intensity_range"]["value"]

    if USE_BRIGHT_FILTER:
        df["is_bright"] = df["is_bright_mean"] | df["is_bright_range"]

    if "is_outside_colony" not in df.columns:
        df["is_outside_colony"] = False

    # ---- final exclude logic ----
    qc_exclude = np.zeros(len(df), dtype=bool)
    reason_list = [[] for _ in range(len(df))]

    for pos, (_, row) in enumerate(df.iterrows()):
        reasons = []

        # 1) border
        if EXCLUDE_BORDER_OBJECTS and bool(row["touches_border"]):
            reasons.append("touches_border")

        # 2) outside colony  <-- NEW main rule
        if USE_OUTSIDE_COLONY_FILTER and REMOVE_OUTSIDE_COLONY_OBJECTS and bool(row["is_outside_colony"]):
            reasons.append("outside_colony_object")

        # 3) oversized alone
        if REMOVE_OVERSIZE_ALONE and bool(row["is_oversized"]):
            reasons.append("oversized_object")

        # 4) oversized + truly isolated
        if REMOVE_OVERSIZE_AND_ISOLATED and bool(row["is_oversized"]) and bool(row["is_truly_isolated"]):
            reasons.append("oversized_and_isolated")

        # 5) optional extreme isolated alone
        if REMOVE_EXTREME_ISOLATED_ALONE and bool(row["is_extreme_isolated"]):
            reasons.append("extreme_isolated_object")

        # 6) bright + isolated
        if REMOVE_BRIGHT_AND_ISOLATED and bool(row["is_bright"]) and bool(row["is_truly_isolated"]):
            reasons.append("bright_and_isolated")

        if len(reasons) > 0:
            qc_exclude[pos] = True
            reason_list[pos] = sorted(set(reasons))

    df["qc_exclude"] = qc_exclude
    df["qc_keep"] = ~df["qc_exclude"]
    df["qc_reason"] = [";".join(r) if len(r) > 0 else "" for r in reason_list]

    return df


def build_image_qc_summary(df_qc: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for image_name, sub in df_qc.groupby("image_name"):
        n_total = len(sub)
        n_keep = int(sub["qc_keep"].sum())
        n_remove = int(sub["qc_exclude"].sum())

        reasons = sub.loc[sub["qc_exclude"], "qc_reason"].astype(str).tolist()

        row = {
            "image_name": image_name,
            "n_total_objects": n_total,
            "n_kept_objects": n_keep,
            "n_removed_objects": n_remove,
            "removed_fraction": (n_remove / n_total) if n_total > 0 else np.nan,

            "n_touches_border": int(sub["touches_border"].sum()) if "touches_border" in sub.columns else 0,
            "n_outside_colony": int(sub["is_outside_colony"].sum()) if "is_outside_colony" in sub.columns else 0,
            "n_oversized": int(sub["is_oversized"].sum()) if "is_oversized" in sub.columns else 0,
            "n_truly_isolated": int(sub["is_truly_isolated"].sum()) if "is_truly_isolated" in sub.columns else 0,
            "n_extreme_isolated": int(sub["is_extreme_isolated"].sum()) if "is_extreme_isolated" in sub.columns else 0,

            "removed_by_touches_border": sum("touches_border" in x.split(";") for x in reasons if x),
            "removed_by_outside_colony_object": sum("outside_colony_object" in x.split(";") for x in reasons if x),
            "removed_by_oversized_object": sum("oversized_object" in x.split(";") for x in reasons if x),
            "removed_by_oversized_and_isolated": sum("oversized_and_isolated" in x.split(";") for x in reasons if x),
            "removed_by_extreme_isolated_object": sum("extreme_isolated_object" in x.split(";") for x in reasons if x),
            "removed_by_bright_and_isolated": sum("bright_and_isolated" in x.split(";") for x in reasons if x),

            "high_removal_warning": bool((n_remove / n_total) > HIGH_REMOVAL_FRACTION_WARN) if n_total > 0 else False,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_qc_masks_and_previews(df_qc: pd.DataFrame):
    done = 0
    failed = 0
    skipped = 0

    for image_name, sub in df_qc.groupby("image_name"):
        try:
            mask_path = find_matching_mask(image_name)
            img_path = find_matching_image(image_name)

            if mask_path is None:
                print(f"[qc-mask skip] {image_name} -> mask not found")
                skipped += 1
                continue

            mask = np.load(mask_path)
            if mask.ndim != 2:
                raise ValueError(f"Mask must be 2D, got shape={mask.shape}")

            keep_labels = set(sub.loc[sub["qc_keep"], "label"].astype(int).tolist())
            remove_labels = set(sub.loc[sub["qc_exclude"], "label"].astype(int).tolist())

            keep_mask = np.where(np.isin(mask, list(keep_labels)), mask, 0).astype(np.int32)
            remove_mask = np.where(np.isin(mask, list(remove_labels)), mask, 0).astype(np.int32)

            if SAVE_QC_MASKS:
                keep_mask_compact = relabel_compact(keep_mask)
                remove_mask_compact = relabel_compact(remove_mask)

                np.save(QC_KEEP_MASK_DIR / f"{image_name}_qc_keep_mask.npy", keep_mask_compact)
                np.save(QC_REMOVE_MASK_DIR / f"{image_name}_qc_removed_mask.npy", remove_mask_compact)
            elif OVERWRITE_MASK_WITH_QC:
                keep_mask_compact = relabel_compact(keep_mask)
                np.save(mask_path, keep_mask_compact)

            if SAVE_QC_PREVIEWS:
                if img_path is None:
                    print(f"[qc-preview skip] {image_name} -> image not found")
                else:
                    img = tiff.imread(img_path)
                    img_gray = ensure_2d_image(img)

                    if img_gray.shape != mask.shape:
                        raise ValueError(
                            f"Shape mismatch for preview: image={img_gray.shape}, mask={mask.shape}, image={image_name}"
                        )

                    overlay = make_qc_preview(img_gray, keep_mask, remove_mask)
                    tiff.imwrite(str(QC_PREVIEW_DIR / f"{image_name}_qc_overlay.tif"), overlay)

            done += 1

        except Exception as e:
            print(f"[qc-mask fail] {image_name}: {e}")
            failed += 1

    return done, failed, skipped


def choose_preview_images(image_summary: pd.DataFrame):
    if len(PREVIEW_IMAGE_NAMES) > 0:
        return PREVIEW_IMAGE_NAMES[:N_PREVIEW_TO_SHOW]

    if len(image_summary) == 0:
        return []

    if PREVIEW_SELECT_MODE == "highest_removed_fraction":
        tmp = image_summary.sort_values(["removed_fraction", "n_removed_objects"], ascending=[False, False])
        return tmp["image_name"].head(N_PREVIEW_TO_SHOW).tolist()

    if PREVIEW_SELECT_MODE == "first_n":
        return image_summary["image_name"].head(N_PREVIEW_TO_SHOW).tolist()

    raise ValueError(f"Unsupported PREVIEW_SELECT_MODE={PREVIEW_SELECT_MODE}")


def show_previews_in_python(image_summary: pd.DataFrame):
    if not SHOW_PREVIEW_AFTER_RUN:
        return

    image_names = choose_preview_images(image_summary)
    if len(image_names) == 0:
        print("No preview images selected for display.")
        return

    paths = []
    valid_names = []
    for name in image_names:
        p = QC_PREVIEW_DIR / f"{name}_qc_overlay.tif"
        if p.exists():
            paths.append(p)
            valid_names.append(name)

    if len(paths) == 0:
        print("No preview tif found to display.")
        return

    n = len(paths)
    ncols = PREVIEW_NCOLS
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, name, p in zip(axes, valid_names, paths):
        img = tiff.imread(p)
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=True)


def main():
    if not NUCLEUS_CSV.exists():
        raise FileNotFoundError(f"nucleus_features.csv not found: {NUCLEUS_CSV}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

    df = pd.read_csv(NUCLEUS_CSV)
    if IMAGE_CSV.exists():
        image_meta = pd.read_csv(IMAGE_CSV)
    else:
        # Fallback for pipelines that only output nucleus_features.csv + intensity CSVs.
        require_columns(df, ["image_name", "image_height", "image_width"], "nucleus_features")
        image_meta = (
            df[["image_name", "image_height", "image_width"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"[warn] image_features.csv not found, fallback to nucleus_features columns: {IMAGE_CSV}")
    df = merge_intensity_table(df, NUCLEUS_INTENSITY_CSV, NUCLEUS_KEY_COLS, "nucleus_intensity_features.csv")
    image_meta = merge_intensity_table(image_meta, IMAGE_INTENSITY_CSV, IMAGE_KEY_COLS, "image_intensity_features.csv")

    if len(df) == 0:
        raise RuntimeError("nucleus_features.csv is empty.")

    print(f"Dataset          : {CFG['name']}")
    print(f"Image dir        : {IMG_DIR}")
    print(f"Mask dir         : {MASK_DIR}")
    print(f"Input nucleus CSV: {NUCLEUS_CSV}")
    print(f"Input intensity  : {NUCLEUS_INTENSITY_CSV}")
    print(f"Input image CSV  : {IMAGE_CSV if IMAGE_CSV.exists() else 'fallback_from_nucleus_features'}")
    print(f"Output dir       : {OUT_DIR}")
    print(f"N objects        : {len(df)}")
    print(f"N images         : {df['image_name'].nunique() if 'image_name' in df.columns else 'NA'}")

    require_columns(
        df,
        [
            "image_name",
            "label",
            "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
            "centroid_row", "centroid_col",
            "area",
            "equivalent_diameter",
            "nn1_distance",
        ],
        "nucleus_features"
    )

    # 1) border
    df_qc = add_border_flag(df, image_meta)

    # 2) fixed-radius neighbors
    df_qc = add_fixed_radius_neighbor_features(df_qc)

    # 3) outside-colony flag  <-- NEW
    if USE_OUTSIDE_COLONY_FILTER:
        df_qc = add_outside_colony_flag(df_qc)
    else:
        df_qc["is_outside_colony"] = False

    # 4) thresholds
    thresholds = build_thresholds(df_qc)

    # 5) final qc logic
    df_qc = apply_qc_flags(df_qc, thresholds)

    # 6) save tables
    df_keep = df_qc[df_qc["qc_keep"]].copy()
    df_remove = df_qc[df_qc["qc_exclude"]].copy()
    image_summary = build_image_qc_summary(df_qc)

    df_keep_feature, df_keep_intensity, keep_intensity_cols = split_feature_and_intensity_tables(df_keep, NUCLEUS_KEY_COLS)
    df_remove_feature, df_remove_intensity, _ = split_feature_and_intensity_tables(df_remove, NUCLEUS_KEY_COLS)

    if OVERWRITE_NUCLEUS_FEATURE_CSV_WITH_QC_KEEP:
        df_keep_feature.to_csv(NUCLEUS_CSV, index=False, encoding="utf-8-sig")
        df_keep_intensity.to_csv(NUCLEUS_INTENSITY_CSV, index=False, encoding="utf-8-sig")
    else:
        df_keep_feature.to_csv(OUT_QC_KEEP, index=False, encoding="utf-8-sig")
        df_keep_intensity.to_csv(OUT_DIR / "nucleus_intensity_features_qc.csv", index=False, encoding="utf-8-sig")

    if KEEP_REMOVED_OBJECTS_CSV:
        df_remove_feature.to_csv(OUT_QC_REMOVE, index=False, encoding="utf-8-sig")
        if len(df_remove_intensity.columns) > len(NUCLEUS_KEY_COLS):
            df_remove_intensity.to_csv(OUT_DIR / "nucleus_intensity_features_removed.csv", index=False, encoding="utf-8-sig")

    if KEEP_IMAGE_SUMMARY_CSV:
        image_summary.to_csv(OUT_IMAGE_SUMMARY, index=False, encoding="utf-8-sig")

    qc_info = {
        "dataset_name": CFG["name"],
        "mode": mode,
        "img_dir": str(IMG_DIR),
        "mask_dir": str(MASK_DIR),
        "feature_dir": str(FEATURE_DIR),
        "input_nucleus_csv": str(NUCLEUS_CSV),
        "input_image_csv": str(IMAGE_CSV) if IMAGE_CSV.exists() else "fallback_from_nucleus_features",
        "n_total_objects": int(len(df_qc)),
        "n_kept_objects": int(len(df_keep)),
        "n_removed_objects": int(len(df_remove)),
        "removed_fraction": float(len(df_remove) / len(df_qc)) if len(df_qc) > 0 else np.nan,
        "params": {
            "EXCLUDE_BORDER_OBJECTS": EXCLUDE_BORDER_OBJECTS,
            "BORDER_MARGIN_PX": BORDER_MARGIN_PX,

            "USE_OUTSIDE_COLONY_FILTER": USE_OUTSIDE_COLONY_FILTER,
            "REMOVE_OUTSIDE_COLONY_OBJECTS": REMOVE_OUTSIDE_COLONY_OBJECTS,
            "COLONY_DILATION_RADIUS_PX": COLONY_DILATION_RADIUS_PX,
            "COLONY_CLOSING_RADIUS_PX": COLONY_CLOSING_RADIUS_PX,
            "COLONY_FILL_HOLES_AREA_PX": COLONY_FILL_HOLES_AREA_PX,
            "COLONY_MIN_COMPONENT_AREA_PX": COLONY_MIN_COMPONENT_AREA_PX,
            "COLONY_KEEP_ONLY_LARGEST_COMPONENT": COLONY_KEEP_ONLY_LARGEST_COMPONENT,

            "USE_AREA_FILTER": USE_AREA_FILTER,
            "USE_EQDIAM_FILTER": USE_EQDIAM_FILTER,
            "AREA_UPPER_MAD_K": AREA_UPPER_MAD_K,
            "EQDIAM_UPPER_MAD_K": EQDIAM_UPPER_MAD_K,
            "OVERSIZE_FALLBACK_HIGH_QUANTILE": OVERSIZE_FALLBACK_HIGH_QUANTILE,
            "OVERSIZE_LOGIC": OVERSIZE_LOGIC,
            "REMOVE_OVERSIZE_ALONE": REMOVE_OVERSIZE_ALONE,

            "USE_ISOLATION_FILTER": USE_ISOLATION_FILTER,
            "FIXED_RADIUS_MODE": FIXED_RADIUS_MODE,
            "FIXED_RADIUS_FACTOR": FIXED_RADIUS_FACTOR,
            "ISOLATED_FIXED_NEIGHBOR_MAX": ISOLATED_FIXED_NEIGHBOR_MAX,
            "NN1_UPPER_MAD_K": NN1_UPPER_MAD_K,
            "NN1_FALLBACK_HIGH_QUANTILE": NN1_FALLBACK_HIGH_QUANTILE,
            "REMOVE_OVERSIZE_AND_ISOLATED": REMOVE_OVERSIZE_AND_ISOLATED,

            "USE_EXTREME_ISOLATION_FILTER": USE_EXTREME_ISOLATION_FILTER,
            "EXTREME_ISOLATED_NEIGHBOR_MAX": EXTREME_ISOLATED_NEIGHBOR_MAX,
            "EXTREME_NN1_UPPER_MAD_K": EXTREME_NN1_UPPER_MAD_K,
            "EXTREME_NN1_FALLBACK_HIGH_QUANTILE": EXTREME_NN1_FALLBACK_HIGH_QUANTILE,
            "REMOVE_EXTREME_ISOLATED_ALONE": REMOVE_EXTREME_ISOLATED_ALONE,

            "USE_BRIGHT_FILTER": USE_BRIGHT_FILTER,
            "MEAN_INT_UPPER_MAD_K": MEAN_INT_UPPER_MAD_K,
            "INT_RANGE_UPPER_MAD_K": INT_RANGE_UPPER_MAD_K,
            "BRIGHT_FALLBACK_HIGH_QUANTILE": BRIGHT_FALLBACK_HIGH_QUANTILE,
            "REMOVE_BRIGHT_AND_ISOLATED": REMOVE_BRIGHT_AND_ISOLATED,

            "SHOW_PREVIEW_AFTER_RUN": SHOW_PREVIEW_AFTER_RUN,
            "N_PREVIEW_TO_SHOW": N_PREVIEW_TO_SHOW,
            "PREVIEW_SELECT_MODE": PREVIEW_SELECT_MODE,
            "PREVIEW_IMAGE_NAMES": PREVIEW_IMAGE_NAMES,
            "PREVIEW_NCOLS": PREVIEW_NCOLS,
            "HIGH_REMOVAL_FRACTION_WARN": HIGH_REMOVAL_FRACTION_WARN,

            "SAVE_QC_MASKS": SAVE_QC_MASKS,
            "SAVE_QC_PREVIEWS": SAVE_QC_PREVIEWS,
        },
        "thresholds": thresholds,
    }

    if KEEP_THRESHOLDS_JSON:
        with open(OUT_THRESHOLDS_JSON, "w", encoding="utf-8") as f:
            json.dump(qc_info, f, ensure_ascii=False, indent=2)

    # 7) export mask + preview
    mask_done, mask_failed, mask_skipped = export_qc_masks_and_previews(df_qc)

    if MINIMAL_OUTPUT_MODE:
        if IMAGE_CSV.exists():
            IMAGE_CSV.unlink()
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR, ignore_errors=True)

    # 8) print summary
    print("=" * 72)
    print("Simple QC finished.")
    if OVERWRITE_NUCLEUS_FEATURE_CSV_WITH_QC_KEEP:
        print(f"Final nucleus CSV: {NUCLEUS_CSV} (overwritten with QC-kept nuclei)")
        print(f"Final intensity  : {NUCLEUS_INTENSITY_CSV} (overwritten with QC-kept nuclei)")
    else:
        print(f"Kept CSV         : {OUT_QC_KEEP}")
    if KEEP_REMOVED_OBJECTS_CSV:
        print(f"Removed CSV      : {OUT_QC_REMOVE}")
    if KEEP_IMAGE_SUMMARY_CSV:
        print(f"Image summary    : {OUT_IMAGE_SUMMARY}")
    if KEEP_THRESHOLDS_JSON:
        print(f"Thresholds JSON  : {OUT_THRESHOLDS_JSON}")
    if SAVE_QC_MASKS:
        print(f"QC keep masks    : {QC_KEEP_MASK_DIR}")
        print(f"QC removed masks : {QC_REMOVE_MASK_DIR}")
    if SAVE_QC_PREVIEWS:
        print(f"QC previews      : {QC_PREVIEW_DIR}")
    if OVERWRITE_MASK_WITH_QC and not SAVE_QC_MASKS:
        print(f"Final masks      : {MASK_DIR} (original masks overwritten by QC-kept masks)")

    print("-" * 72)
    print(f"Total   : {len(df_qc)}")
    print(f"Kept    : {len(df_keep)}")
    print(f"Removed : {len(df_remove)}")
    print(f"Removed fraction: {len(df_remove) / len(df_qc):.4f}")

    print("\nMask/preview export:")
    print(f"Done   : {mask_done}")
    print(f"Skipped: {mask_skipped}")
    print(f"Failed : {mask_failed}")

    if len(df_remove) > 0:
        print("\nTop removal reasons:")
        print(df_remove["qc_reason"].value_counts().head(10).to_string())
    else:
        print("\nNo objects removed.")

    if keep_intensity_cols:
        print(f"\nSeparated intensity columns kept independent from model input ({len(keep_intensity_cols)}):")
        print(keep_intensity_cols)

    if len(image_summary) > 0:
        n_warn = int(image_summary["high_removal_warning"].sum())
        print(f"\nImages with high removal warning (> {HIGH_REMOVAL_FRACTION_WARN:.2%}): {n_warn}")
        if n_warn > 0:
            warn_imgs = image_summary.loc[image_summary["high_removal_warning"], "image_name"].tolist()
            print("Warning images:", warn_imgs)

    print("\nPreview legend / 预览图说明:")
    print("- green boundary = kept objects")
    print("- red boundary   = removed objects")

    print("\nCurrent default delete logic / 当前默认删除逻辑:")
    print("- touches_border")
    print("- outside_colony_object")
    print("- oversized_and_isolated")
    if REMOVE_EXTREME_ISOLATED_ALONE:
        print("- extreme_isolated_object")
    if REMOVE_BRIGHT_AND_ISOLATED:
        print("- bright_and_isolated")

    # 9) directly show preview images in Python
    show_previews_in_python(image_summary)


if __name__ == "__main__":
    main()
