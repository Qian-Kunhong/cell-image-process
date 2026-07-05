from pathlib import Path
import math
import re
import runpy
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops

# ===== 可选：优先使用 scipy 的 KDTree，加速近邻计算 =====
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    warnings.warn("scipy.spatial.cKDTree not available, fallback to brute-force distance calculation.")

# =========================================================
# 03_extract_nuclear_and_neighborhood_features.py
# ---------------------------------------------------------
# 功能 / Function:
# 1) 读取 DAPI 图像和 02 脚本生成的 mask
# 2) 提取单核形态特征 + 强度特征
# 3) 基于 kth-nearest-neighbor 构建自适应邻域
# 4) 提取 neighborhood-aware features
# 5) 输出 nucleus-level 与 image-level CSV
# =========================================================

# =========================
# 路径设置 / Path settings
# =========================
mode = globals().get("mode", 1)
SUZUI_ROOT = Path(globals().get("SUZUI_ROOT", r"F:\Suzui"))
ANALYSIS_ROOT = Path(globals().get("ANALYSIS_ROOT", SUZUI_ROOT / "analysis_out"))
DATASET_NAME = globals().get("DATASET_NAME", "A-1-3")
TRAINING_ROOT = Path(globals().get("TRAINING_ROOT", SUZUI_ROOT / "training data"))

# data
if mode == 1:
    IMG_DIR = ANALYSIS_ROOT / DATASET_NAME
    MASK_DIR = IMG_DIR / "masks"
    OUT_DIR = IMG_DIR / "features"
    print('processing data')

# training data SNL
if mode == 2:
    IMG_DIR = TRAINING_ROOT / "SNL"
    MASK_DIR = ANALYSIS_ROOT / "masks_training" / "SNL"
    OUT_DIR = ANALYSIS_ROOT / "features_training" / "SNL"
    print('processing SNL training data')

# training data MEF
if mode == 3:
    IMG_DIR = TRAINING_ROOT / "MEF"
    MASK_DIR = ANALYSIS_ROOT / "masks_training" / "MEF"
    OUT_DIR = ANALYSIS_ROOT / "features_training" / "MEF"
    print('processing MEF training data')

OUT_DIR.mkdir(parents=True, exist_ok=True)

NUCLEUS_CSV = OUT_DIR / "nucleus_features.csv"
IMAGE_CSV = OUT_DIR / "image_features.csv"
NUCLEUS_INTENSITY_CSV = OUT_DIR / "nucleus_intensity_features.csv"
IMAGE_INTENSITY_CSV = OUT_DIR / "image_intensity_features.csv"

MASK_SUFFIX = "_mask.npy"
IMG_EXTENSIONS = [".tif", ".tiff"]

# =========================
# 参数设置 / Parameters
# =========================
K_NEIGHBORS = 6        # 第k近邻，用于定义自适应邻域
MIN_AREA = 0           # 可后续调，比如 20 / 30；现在先不过滤小核
SAVE_PER_IMAGE_CSV = False  # 若想每张图单独存一个 nucleus CSV，可改 True
DEFAULT_PIXEL_SIZE_UM = None
RUN_QC_AFTER_EXTRACTION = True

NUCLEUS_KEY_COLS = ["image_name", "label"]
IMAGE_KEY_COLS = ["image_name"]
BRIGHTNESS_DERIVED_COLS = {
    "is_bright",
    "is_bright_mean",
    "is_bright_range",
}

# =========================
# 工具函数 / Utilities
# =========================
def ensure_2d_image(img: np.ndarray) -> np.ndarray:
    """
    将图像转成 2D 灰度图。
    Convert input image to 2D grayscale.
    """
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        # HWC, 单通道
        if img.shape[-1] == 1:
            return img[..., 0]
        # HWC, 多通道 -> 取第一通道（DAPI通常就是单通道）
        if img.shape[-1] in (3, 4):
            return img[..., 0]
        # CHW
        if img.shape[0] == 1:
            return img[0]
        if img.shape[0] in (3, 4):
            return img[0]

    raise ValueError(f"Unsupported image shape: {img.shape}")


def find_matching_image(mask_path: Path) -> Path | None:
    """
    根据 xxx_mask.npy 找到对应的 xxx.tif / xxx.tiff
    """
    stem = mask_path.stem
    if stem.endswith("_mask"):
        img_stem = stem[:-5]
    else:
        img_stem = stem

    for ext in IMG_EXTENSIONS:
        candidate = IMG_DIR / f"{img_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_float_safe(value) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) and x > 0 else None


def parse_pixel_size_from_text(text: str) -> tuple[float | None, float | None, str | None]:
    if not text:
        return None, None, None

    patterns = [
        (r"PhysicalSizeX\s*=\s*['\"]?([0-9.eE+-]+)", r"PhysicalSizeY\s*=\s*['\"]?([0-9.eE+-]+)", "ome_physical_size"),
        (r"pixel[_\s-]*size[_\s-]*x\s*[:=]\s*([0-9.eE+-]+)", r"pixel[_\s-]*size[_\s-]*y\s*[:=]\s*([0-9.eE+-]+)", "text_pixel_size"),
        (r"spacing[_\s-]*x\s*[:=]\s*([0-9.eE+-]+)", r"spacing[_\s-]*y\s*[:=]\s*([0-9.eE+-]+)", "text_spacing"),
    ]
    for px_pat, py_pat, source in patterns:
        mx = re.search(px_pat, text, flags=re.IGNORECASE)
        my = re.search(py_pat, text, flags=re.IGNORECASE)
        if mx and my:
            sx = parse_float_safe(mx.group(1))
            sy = parse_float_safe(my.group(1))
            if sx is not None and sy is not None:
                return sy, sx, source

    for pat, source in [
        (r"PhysicalSizeX\s*=\s*['\"]?([0-9.eE+-]+)", "ome_physical_size_x_only"),
        (r"pixel[_\s-]*size\s*[:=]\s*([0-9.eE+-]+)", "text_pixel_size_scalar"),
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            s = parse_float_safe(m.group(1))
            if s is not None:
                return s, s, source

    return None, None, None


def extract_pixel_size_um(img_path: Path) -> tuple[float, float, str]:
    with tiff.TiffFile(img_path) as tif:
        page = tif.pages[0]

        row_um, col_um, source = parse_pixel_size_from_text(page.description or "")
        if row_um is not None and col_um is not None:
            return row_um, col_um, source or "description"

        if tif.ome_metadata:
            row_um, col_um, source = parse_pixel_size_from_text(tif.ome_metadata)
            if row_um is not None and col_um is not None:
                return row_um, col_um, source or "ome_metadata"

        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        if xres_tag is not None and yres_tag is not None and unit_tag is not None:
            try:
                xres = xres_tag.value
                yres = yres_tag.value
                unit = unit_tag.value
                xpp = float(xres[0]) / float(xres[1]) if isinstance(xres, tuple) else float(xres)
                ypp = float(yres[0]) / float(yres[1]) if isinstance(yres, tuple) else float(yres)
                if xpp > 0 and ypp > 0:
                    if unit == 2:
                        return 25400.0 / ypp, 25400.0 / xpp, "tiff_resolution_inch"
                    if unit == 3:
                        return 10000.0 / ypp, 10000.0 / xpp, "tiff_resolution_cm"
            except Exception:
                pass

    manual = parse_float_safe(DEFAULT_PIXEL_SIZE_UM)
    if manual is not None:
        return manual, manual, "manual_default"

    raise RuntimeError(
        f"Cannot determine pixel size for image: {img_path}\n"
        "Please embed TIFF physical size metadata or set DEFAULT_PIXEL_SIZE_UM manually."
    )


def is_intensity_related_column(col: str) -> bool:
    cl = str(col).lower().strip()
    return ("intensity" in cl) or (cl in BRIGHTNESS_DERIVED_COLS)


def split_feature_and_intensity_tables(
    df: pd.DataFrame,
    key_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if df is None:
        empty = pd.DataFrame()
        return empty, empty, []

    intensity_cols = [c for c in df.columns if c not in key_cols and is_intensity_related_column(c)]
    feature_cols = [c for c in df.columns if c not in intensity_cols]
    intensity_table_cols = [c for c in key_cols if c in df.columns] + intensity_cols

    feature_df = df.loc[:, feature_cols].copy()
    intensity_df = df.loc[:, intensity_table_cols].copy()
    return feature_df, intensity_df, intensity_cols


def safe_div(a, b):
    return a / b if b not in (0, None) else np.nan


def circularity(area: float, perimeter: float) -> float:
    """
    4πA / P²
    """
    if perimeter is None or perimeter <= 0:
        return np.nan
    return 4.0 * math.pi * area / (perimeter ** 2)


def bbox_aspect_ratio(min_row, min_col, max_row, max_col) -> float:
    h = max_row - min_row
    w = max_col - min_col
    if h <= 0 or w <= 0:
        return np.nan
    return max(h, w) / min(h, w)


def robust_std(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def get_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    """
    fallback brute-force 距离矩阵
    """
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist


def compute_knn_and_adaptive_neighbors(centroids: np.ndarray, k_neighbors: int):
    """
    基于质心计算：
    - 第1近邻距离
    - 第k近邻距离（adaptive radius）
    - 前k个近邻距离均值/标准差
    - 自适应邻域中的 neighbor indices（不含自己）
    """
    n = len(centroids)

    nn1 = np.full(n, np.nan, dtype=float)
    kth_radius = np.full(n, np.nan, dtype=float)
    knn_mean = np.full(n, np.nan, dtype=float)
    knn_std = np.full(n, np.nan, dtype=float)
    neighbor_lists = [[] for _ in range(n)]

    if n == 0:
        return nn1, kth_radius, knn_mean, knn_std, neighbor_lists

    if n == 1:
        return nn1, kth_radius, knn_mean, knn_std, neighbor_lists

    k_eff = min(k_neighbors, n - 1)

    if HAS_SCIPY:
        tree = cKDTree(centroids)
        dists, inds = tree.query(centroids, k=k_eff + 1)

        if k_eff + 1 == 1:
            dists = dists[:, None]
            inds = inds[:, None]

        for i in range(n):
            # 第0个是自己
            di = np.asarray(dists[i][1:], dtype=float)
            ii = np.asarray(inds[i][1:], dtype=int)

            if len(di) > 0:
                nn1[i] = di[0]
                kth_radius[i] = di[-1]
                knn_mean[i] = float(np.mean(di))
                knn_std[i] = robust_std(di)

                # 自适应邻域：半径 = 第k近邻距离
                idxs = tree.query_ball_point(centroids[i], r=float(kth_radius[i]) + 1e-8)
                idxs = [j for j in idxs if j != i]
                neighbor_lists[i] = idxs
    else:
        dist_mat = get_distance_matrix(centroids)

        for i in range(n):
            di = dist_mat[i].copy()
            di[i] = np.inf

            sort_idx = np.argsort(di)
            nn_idx = sort_idx[:k_eff]
            nn_dist = di[nn_idx]

            nn_dist = nn_dist[np.isfinite(nn_dist)]
            nn_idx = nn_idx[np.isfinite(di[nn_idx])]

            if len(nn_dist) > 0:
                nn1[i] = nn_dist[0]
                kth_radius[i] = nn_dist[-1]
                knn_mean[i] = float(np.mean(nn_dist))
                knn_std[i] = robust_std(nn_dist)

                radius = kth_radius[i]
                idxs = np.where((dist_mat[i] <= radius + 1e-8) & (np.arange(n) != i))[0]
                neighbor_lists[i] = idxs.tolist()

    return nn1, kth_radius, knn_mean, knn_std, neighbor_lists


def summarize_neighbors(
    df_img: pd.DataFrame,
    neighbor_lists,
    kth_radius,
    coords: np.ndarray,
    area_col: str,
    radius_col: str,
    density_col: str,
    area_mean_col: str,
    area_std_col: str,
    distance_mean_col: str,
    distance_std_col: str,
):
    """
    对每个 nucleus 计算邻域统计特征
    """
    n = len(df_img)

    # 准备输出列
    neighbor_count = np.zeros(n, dtype=int)
    local_density = np.full(n, np.nan, dtype=float)

    nb_area_mean = np.full(n, np.nan, dtype=float)
    nb_area_std = np.full(n, np.nan, dtype=float)

    nb_circularity_mean = np.full(n, np.nan, dtype=float)
    nb_circularity_std = np.full(n, np.nan, dtype=float)

    nb_ecc_mean = np.full(n, np.nan, dtype=float)
    nb_ecc_std = np.full(n, np.nan, dtype=float)

    nb_aspect_mean = np.full(n, np.nan, dtype=float)
    nb_aspect_std = np.full(n, np.nan, dtype=float)

    nb_intensity_mean = np.full(n, np.nan, dtype=float)
    nb_intensity_std = np.full(n, np.nan, dtype=float)

    nb_distance_mean = np.full(n, np.nan, dtype=float)
    nb_distance_std = np.full(n, np.nan, dtype=float)

    area_arr = df_img[area_col].to_numpy(dtype=float)
    circ_arr = df_img["circularity"].to_numpy(dtype=float)
    ecc_arr = df_img["eccentricity"].to_numpy(dtype=float)
    ar_arr = df_img["aspect_ratio"].to_numpy(dtype=float)
    inten_arr = df_img["mean_intensity"].to_numpy(dtype=float)

    for i, nb_idx in enumerate(neighbor_lists):
        if len(nb_idx) == 0:
            neighbor_count[i] = 0
            if np.isfinite(kth_radius[i]) and kth_radius[i] > 0:
                local_density[i] = 0.0
            continue

        neighbor_count[i] = len(nb_idx)

        r = kth_radius[i]
        if np.isfinite(r) and r > 0:
            local_density[i] = len(nb_idx) / (math.pi * (r ** 2))

        nb_area = area_arr[nb_idx]
        nb_circ = circ_arr[nb_idx]
        nb_ecc = ecc_arr[nb_idx]
        nb_ar = ar_arr[nb_idx]
        nb_int = inten_arr[nb_idx]

        dxy = coords[nb_idx] - coords[i]
        dists = np.sqrt(np.sum(dxy ** 2, axis=1))

        nb_area_mean[i] = np.nanmean(nb_area)
        nb_area_std[i] = np.nanstd(nb_area, ddof=1) if len(nb_area) > 1 else 0.0

        nb_circularity_mean[i] = np.nanmean(nb_circ)
        nb_circularity_std[i] = np.nanstd(nb_circ, ddof=1) if len(nb_circ) > 1 else 0.0

        nb_ecc_mean[i] = np.nanmean(nb_ecc)
        nb_ecc_std[i] = np.nanstd(nb_ecc, ddof=1) if len(nb_ecc) > 1 else 0.0

        nb_aspect_mean[i] = np.nanmean(nb_ar)
        nb_aspect_std[i] = np.nanstd(nb_ar, ddof=1) if len(nb_ar) > 1 else 0.0

        nb_intensity_mean[i] = np.nanmean(nb_int)
        nb_intensity_std[i] = np.nanstd(nb_int, ddof=1) if len(nb_int) > 1 else 0.0

        nb_distance_mean[i] = np.nanmean(dists)
        nb_distance_std[i] = np.nanstd(dists, ddof=1) if len(dists) > 1 else 0.0

    df_img["adaptive_neighbor_count"] = neighbor_count
    df_img[radius_col] = kth_radius
    df_img[density_col] = local_density

    df_img[area_mean_col] = nb_area_mean
    df_img[area_std_col] = nb_area_std

    df_img["nb_circularity_mean"] = nb_circularity_mean
    df_img["nb_circularity_std"] = nb_circularity_std

    df_img["nb_eccentricity_mean"] = nb_ecc_mean
    df_img["nb_eccentricity_std"] = nb_ecc_std

    df_img["nb_aspect_ratio_mean"] = nb_aspect_mean
    df_img["nb_aspect_ratio_std"] = nb_aspect_std

    df_img["nb_mean_intensity_mean"] = nb_intensity_mean
    df_img["nb_mean_intensity_std"] = nb_intensity_std

    df_img[distance_mean_col] = nb_distance_mean
    df_img[distance_std_col] = nb_distance_std

    return df_img


def extract_nucleus_features(
    image: np.ndarray,
    mask: np.ndarray,
    image_name: str,
    pixel_size_row_um: float,
    pixel_size_col_um: float,
) -> pd.DataFrame:
    """
    提取单张图像的 nucleus-level features
    """
    props = regionprops(mask, intensity_image=image)

    rows = []
    for rp in props:
        if rp.area < MIN_AREA:
            continue

        coords = rp.coords
        pix_vals = image[coords[:, 0], coords[:, 1]].astype(np.float32)

        min_row, min_col, max_row, max_col = rp.bbox

        area = float(rp.area)
        perimeter = float(rp.perimeter)
        major = float(rp.major_axis_length)
        minor = float(rp.minor_axis_length)

        row = {
            "image_name": image_name,
            "label": int(rp.label),

            # centroid
            "centroid_row": float(rp.centroid[0]),
            "centroid_col": float(rp.centroid[1]),

            # bbox
            "bbox_min_row": int(min_row),
            "bbox_min_col": int(min_col),
            "bbox_max_row": int(max_row),
            "bbox_max_col": int(max_col),
            "bbox_height": int(max_row - min_row),
            "bbox_width": int(max_col - min_col),
            "bbox_aspect_ratio": bbox_aspect_ratio(min_row, min_col, max_row, max_col),

            # morphology / shape
            "area": area,
            "perimeter": perimeter,
            "equivalent_diameter": float(rp.equivalent_diameter_area),
            "major_axis_length": major,
            "minor_axis_length": minor,
            "aspect_ratio": safe_div(major, minor) if minor > 0 else np.nan,
            "eccentricity": float(rp.eccentricity),
            "solidity": float(rp.solidity),
            "extent": float(rp.extent),
            "convex_area": float(rp.convex_area),
            "filled_area": float(rp.filled_area),
            "circularity": circularity(area, perimeter),

            # intensity / DAPI
            "mean_intensity": float(np.mean(pix_vals)),
            "std_intensity": float(np.std(pix_vals, ddof=1)) if len(pix_vals) > 1 else 0.0,
            "min_intensity": float(np.min(pix_vals)),
            "max_intensity": float(np.max(pix_vals)),
            "intensity_range": float(np.max(pix_vals) - np.min(pix_vals)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    pixel_area_um2 = pixel_size_row_um * pixel_size_col_um
    pixel_size_mean_um = math.sqrt(pixel_area_um2)

    df["pixel_size_row_um"] = pixel_size_row_um
    df["pixel_size_col_um"] = pixel_size_col_um
    df["pixel_size_mean_um"] = pixel_size_mean_um

    df["centroid_row_um"] = df["centroid_row"] * pixel_size_row_um
    df["centroid_col_um"] = df["centroid_col"] * pixel_size_col_um
    df["bbox_height_um"] = df["bbox_height"] * pixel_size_row_um
    df["bbox_width_um"] = df["bbox_width"] * pixel_size_col_um

    df["area_um2"] = df["area"] * pixel_area_um2
    df["convex_area_um2"] = df["convex_area"] * pixel_area_um2
    df["filled_area_um2"] = df["filled_area"] * pixel_area_um2

    df["perimeter_um"] = df["perimeter"] * pixel_size_mean_um
    df["equivalent_diameter_um"] = df["equivalent_diameter"] * pixel_size_mean_um
    df["major_axis_length_um"] = df["major_axis_length"] * pixel_size_mean_um
    df["minor_axis_length_um"] = df["minor_axis_length"] * pixel_size_mean_um

    # ===== 邻域特征 / neighborhood features =====
    centroids_px = df[["centroid_row", "centroid_col"]].to_numpy(dtype=float)
    centroids_um = df[["centroid_row_um", "centroid_col_um"]].to_numpy(dtype=float)

    nn1_px, kth_radius_px, knn_mean_px, knn_std_px, neighbor_lists_px = compute_knn_and_adaptive_neighbors(
        centroids=centroids_px,
        k_neighbors=K_NEIGHBORS
    )
    nn1_um, kth_radius_um, knn_mean_um, knn_std_um, neighbor_lists_um = compute_knn_and_adaptive_neighbors(
        centroids=centroids_um,
        k_neighbors=K_NEIGHBORS
    )

    df["nn1_distance"] = nn1_px
    df[f"nn{K_NEIGHBORS}_distance"] = kth_radius_px
    df[f"knn{K_NEIGHBORS}_distance_mean"] = knn_mean_px
    df[f"knn{K_NEIGHBORS}_distance_std"] = knn_std_px

    df["nn1_distance_um"] = nn1_um
    df[f"nn{K_NEIGHBORS}_distance_um"] = kth_radius_um
    df[f"knn{K_NEIGHBORS}_distance_mean_um"] = knn_mean_um
    df[f"knn{K_NEIGHBORS}_distance_std_um"] = knn_std_um

    df = summarize_neighbors(
        df_img=df,
        neighbor_lists=neighbor_lists_px,
        kth_radius=kth_radius_px,
        coords=centroids_px,
        area_col="area",
        radius_col="adaptive_radius",
        density_col="local_density",
        area_mean_col="nb_area_mean",
        area_std_col="nb_area_std",
        distance_mean_col="nb_distance_mean",
        distance_std_col="nb_distance_std",
    )
    df = summarize_neighbors(
        df_img=df,
        neighbor_lists=neighbor_lists_um,
        kth_radius=kth_radius_um,
        coords=centroids_um,
        area_col="area_um2",
        radius_col="adaptive_radius_um",
        density_col="local_density_per_um2",
        area_mean_col="nb_area_mean_um2",
        area_std_col="nb_area_std_um2",
        distance_mean_col="nb_distance_mean_um",
        distance_std_col="nb_distance_std_um",
    )

    return df


def summarize_image_features(
    df_img: pd.DataFrame,
    image_name: str,
    image_shape: tuple,
    pixel_size_row_um: float,
    pixel_size_col_um: float,
    pixel_size_source: str,
) -> dict:
    """
    将 nucleus-level 特征汇总成 image-level 特征
    """
    h, w = image_shape
    image_area = h * w
    image_height_um = h * pixel_size_row_um
    image_width_um = w * pixel_size_col_um
    image_area_um2 = image_height_um * image_width_um
    n_nuclei = len(df_img)

    summary = {
        "image_name": image_name,
        "image_height": h,
        "image_width": w,
        "image_area_px": image_area,
        "pixel_size_row_um": pixel_size_row_um,
        "pixel_size_col_um": pixel_size_col_um,
        "pixel_size_mean_um": math.sqrt(pixel_size_row_um * pixel_size_col_um),
        "pixel_size_source": pixel_size_source,
        "image_height_um": image_height_um,
        "image_width_um": image_width_um,
        "image_area_um2": image_area_um2,
        "n_nuclei": n_nuclei,
        "nuclei_density_per_1e5px": (n_nuclei / image_area) * 1e5 if image_area > 0 else np.nan,
        "nuclei_density_per_mm2": (n_nuclei / image_area_um2) * 1e6 if image_area_um2 > 0 else np.nan,
    }

    if n_nuclei == 0:
        return summary

    # 不把坐标、bbox、label纳入最终图像级汇总
    exclude_cols = {
        "label",
        "centroid_row", "centroid_col",
        "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
        "image_name"
    }

    numeric_cols = [
        c for c in df_img.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_img[c])
    ]

    for col in numeric_cols:
        vals = df_img[col].to_numpy(dtype=float)
        summary[f"{col}__mean"] = float(np.nanmean(vals))
        summary[f"{col}__std"] = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else 0.0
        summary[f"{col}__median"] = float(np.nanmedian(vals))

    return summary


def run_qc_stage():
    qc_script = Path(__file__).with_name("03-1_qc_filter.py")
    if not qc_script.exists():
        raise FileNotFoundError(f"QC script not found: {qc_script}")

    print("\n" + "=" * 70)
    print("Starting QC stage from 03-1_qc_filter.py ...")
    print("=" * 70)

    runpy.run_path(
        str(qc_script),
        run_name="__main__",
        init_globals={
            "mode": mode,
            "SUZUI_ROOT": SUZUI_ROOT,
            "ANALYSIS_ROOT": ANALYSIS_ROOT,
            "DATASET_NAME": DATASET_NAME,
            "TRAINING_ROOT": TRAINING_ROOT,
        },
    )


def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

    mask_files = sorted(MASK_DIR.glob(f"*{MASK_SUFFIX}"))
    if not mask_files:
        raise RuntimeError(f"No mask files found in: {MASK_DIR}")

    print(f"Image folder : {IMG_DIR}")
    print(f"Mask folder  : {MASK_DIR}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Found {len(mask_files)} mask files")
    print(f"K neighbors  : {K_NEIGHBORS}")

    all_nucleus_dfs = []
    image_rows = []

    done = 0
    failed = 0
    skipped = 0

    for mask_path in mask_files:
        try:
            img_path = find_matching_image(mask_path)
            if img_path is None:
                print(f"[skip] {mask_path.name} -> matching image not found")
                skipped += 1
                continue

            image_name = img_path.stem

            img = tiff.imread(img_path)
            img = ensure_2d_image(img).astype(np.float32)
            pixel_size_row_um, pixel_size_col_um, pixel_size_source = extract_pixel_size_um(img_path)
            print(
                f"[meta] {image_name} -> "
                f"pixel_size_row_um={pixel_size_row_um:.6f}, "
                f"pixel_size_col_um={pixel_size_col_um:.6f}, "
                f"pixel_size_mean_um={math.sqrt(pixel_size_row_um * pixel_size_col_um):.6f}, "
                f"source={pixel_size_source}"
            )

            mask = np.load(mask_path)
            if mask.ndim != 2:
                raise ValueError(f"Mask must be 2D, got shape={mask.shape}")

            if mask.shape != img.shape:
                raise ValueError(
                    f"Shape mismatch: image={img.shape}, mask={mask.shape}, file={mask_path.name}"
                )

            df_img = extract_nucleus_features(
                img,
                mask,
                image_name,
                pixel_size_row_um=pixel_size_row_um,
                pixel_size_col_um=pixel_size_col_um,
            )

            if len(df_img) == 0:
                print(f"[ok] {image_name} -> no valid nuclei after filtering")
            else:
                all_nucleus_dfs.append(df_img)

                if SAVE_PER_IMAGE_CSV:
                    per_img_feature_df, per_img_intensity_df, _ = split_feature_and_intensity_tables(
                        df_img,
                        NUCLEUS_KEY_COLS,
                    )
                    per_img_csv = OUT_DIR / f"{image_name}_nucleus_features.csv"
                    per_img_int_csv = OUT_DIR / f"{image_name}_nucleus_intensity_features.csv"
                    per_img_feature_df.to_csv(per_img_csv, index=False, encoding="utf-8-sig")
                    per_img_intensity_df.to_csv(per_img_int_csv, index=False, encoding="utf-8-sig")

                print(f"[ok] {image_name} -> nuclei={len(df_img)}")

            img_summary = summarize_image_features(
                df_img,
                image_name,
                img.shape,
                pixel_size_row_um=pixel_size_row_um,
                pixel_size_col_um=pixel_size_col_um,
                pixel_size_source=pixel_size_source,
            )
            image_rows.append(img_summary)

            done += 1

        except Exception as e:
            print(f"[fail] {mask_path.name}: {e}")
            failed += 1

    # ===== 合并输出 / Save outputs =====
    if len(all_nucleus_dfs) > 0:
        nucleus_df = pd.concat(all_nucleus_dfs, axis=0, ignore_index=True)
    else:
        nucleus_df = pd.DataFrame()

    image_df = pd.DataFrame(image_rows)

    nucleus_feature_df, nucleus_intensity_df, nucleus_intensity_cols = split_feature_and_intensity_tables(
        nucleus_df,
        NUCLEUS_KEY_COLS,
    )
    image_feature_df, image_intensity_df, image_intensity_cols = split_feature_and_intensity_tables(
        image_df,
        IMAGE_KEY_COLS,
    )

    nucleus_feature_df.to_csv(NUCLEUS_CSV, index=False, encoding="utf-8-sig")
    nucleus_intensity_df.to_csv(NUCLEUS_INTENSITY_CSV, index=False, encoding="utf-8-sig")
    image_feature_df.to_csv(IMAGE_CSV, index=False, encoding="utf-8-sig")
    image_intensity_df.to_csv(IMAGE_INTENSITY_CSV, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print(f"Done   : {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")
    print(f"Nucleus feature CSV  : {NUCLEUS_CSV}")
    print(f"Nucleus intensity CSV: {NUCLEUS_INTENSITY_CSV}")
    print(f"Image feature CSV    : {IMAGE_CSV}")
    print(f"Image intensity CSV  : {IMAGE_INTENSITY_CSV}")

    if len(nucleus_feature_df) > 0:
        print("\nExtracted nucleus-level features include:")
        print("- morphology: area, perimeter, equivalent_diameter, major/minor axis, aspect_ratio, eccentricity, solidity, extent, circularity")
        print("- intensity : saved separately for independent validation, not in nucleus_features.csv")
        print(f"- neighborhood (k={K_NEIGHBORS}): nn1_distance, adaptive_radius, local_density, neighbor summary stats")
        print(f"- separated nucleus intensity columns ({len(nucleus_intensity_cols)}): {nucleus_intensity_cols}")
        if image_intensity_cols:
            print(f"- separated image intensity columns ({len(image_intensity_cols)}): {image_intensity_cols[:10]}")
    else:
        print("\nWarning: nucleus_features.csv is empty. Please check masks or MIN_AREA.")

    if RUN_QC_AFTER_EXTRACTION:
        run_qc_stage()


if __name__ == "__main__":
    main()
