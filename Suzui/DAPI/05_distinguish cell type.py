from __future__ import annotations

import json
import math
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError(
        "xgboost is not available.\n"
        "Please install it first, for example:\n"
        "  pip install xgboost\n"
        f"Original error: {e}"
    )

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    warnings.warn("umap-learn is not installed. Inference UMAP plotting will be skipped.")


# ============================================================
# 0) Key paths & parameters / 关键路径与参数
# ============================================================

# ----------------------------
# Training side / 训练侧
# X / features 来自原始训练特征表
# y / label 来自最新版 morphveto clustered CSV 的 final_state_label
# ----------------------------
SUZUI_ROOT = Path(r"F:\Suzui")
ANALYSIS_ROOT = SUZUI_ROOT / "analysis_out"
TRAINING_ROOT = SUZUI_ROOT / "training data"
TRAINING_SET_NAME = "SNL"
INFERENCE_SET_NAME = "A-1-3"

# ----------------------------
# Pipeline variant / 流程模式
# ----------------------------
# "05"  : baseline
# "05b" : relative absolute-feature normalization
# "05c" : relative normalization + one global timelapse UMAP, then split by image
PIPELINE_VARIANT = "05c"

PIPELINE_VARIANTS = {
    "05": {
        "output_suffix": "_supervised_prediction_xgb",
        "absolute_feature_normalization": "none",
        "infer_umap_mode": "per_image_fit",
    },
    "05b": {
        "output_suffix": "_supervised_prediction_xgb_relative",
        "absolute_feature_normalization": "per_image_robust",
        "infer_umap_mode": "per_image_fit",
    },
    "05c": {
        "output_suffix": "_supervised_prediction_xgb_relative_global_umap",
        "absolute_feature_normalization": "per_image_robust",
        "infer_umap_mode": "global_fit_then_split",
    },
}

if PIPELINE_VARIANT not in PIPELINE_VARIANTS:
    raise ValueError(f"Unsupported PIPELINE_VARIANT: {PIPELINE_VARIANT}")

PIPELINE_CONFIG = PIPELINE_VARIANTS[PIPELINE_VARIANT]

TRAIN_FEATURE_CSV = (
    ANALYSIS_ROOT / "features_training" / TRAINING_SET_NAME / "nucleus_features.csv"
)

TRAIN_LABEL_CSV = TRAIN_FEATURE_CSV

TRAIN_LABEL_COL = "final_state_label"

# ----------------------------
# Inference side / 推理侧
# ----------------------------
INFER_IMAGE_DIR = ANALYSIS_ROOT / INFERENCE_SET_NAME
INFER_FEATURE_CSV = ANALYSIS_ROOT / INFERENCE_SET_NAME / "features" / "nucleus_features.csv"
INFER_MASK_DIR = ANALYSIS_ROOT / INFERENCE_SET_NAME / "masks"

# ----------------------------
# Output / 输出
# ----------------------------
OUTPUT_ROOT = ANALYSIS_ROOT / f"{INFERENCE_SET_NAME}{PIPELINE_CONFIG['output_suffix']}"
MINIMAL_OUTPUT_MODE = True
PREDICTION_CSV_NAME = "nucleus_predictions.csv"
MODEL_INFO_JSON_NAME = "run_info_model_info.json"
FEATURE_IMPORTANCE_CSV_NAME = "feature_importance_xgb_gain.csv"
MODEL_JSON_NAME = "xgb_model.json"
OVERLAY_LOG_JSON_NAME = "overlay_log.json"
INFER_UMAP_BY_IMAGE_DIRNAME = "inference_umap_by_image"
TIMEPOINT_RATIO_CSV_NAME = "timepoint_cluster_ratios.csv"
TIMEPOINT_RATIO_FIG_NAME = "timepoint_cluster_ratios.png"

# ----------------------------
# Optional manual column override / 可选手动列名覆盖
# 如自动识别失败，可手动填列名
# ----------------------------
MANUAL_IMAGE_COL_TRAIN_FEATURE = None
MANUAL_NUCLEUS_COL_TRAIN_FEATURE = None

MANUAL_IMAGE_COL_TRAIN_LABEL = None
MANUAL_NUCLEUS_COL_TRAIN_LABEL = None

MANUAL_IMAGE_COL_INFER = None
MANUAL_NUCLEUS_COL_INFER = None

# ----------------------------
# Merge settings / 合并设置
# ----------------------------
ALLOW_ROW_ORDER_FALLBACK = False

# ----------------------------
# Model behavior / 模型行为
# ----------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20

# 仅用 deviated / undifferentiated 训练主分类器
# uncertain 通过阈值后处理产生
TRAIN_WITH_ONLY_HARD_LABELS = True

# Hard label thresholds / 硬分类阈值
# 这里仅影响 hard label overlay，不影响连续 heatmap 的颜色细节
# 调得更保守一些：
# - 更高的 deviated threshold，避免测试数据被过度打红
# - 更高的 undiff threshold，让低到中低概率更容易进入绿色
DEVIATED_THRESHOLD = 0.995
UNDIFF_THRESHOLD = 0.90

# ----------------------------
# XGBoost params / XGBoost 参数
# ----------------------------
XGB_N_ESTIMATORS = 700
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.035
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_MIN_CHILD_WEIGHT = 4.0
XGB_REG_ALPHA = 0.5
XGB_REG_LAMBDA = 2.0
XGB_GAMMA = 0.1
XGB_N_JOBS = -1

# deviated 为正类；默认不额外上调，避免 overly aggressive red calling
POSITIVE_CLASS_WEIGHT = 1.0

# ----------------------------
# Feature filtering / 特征过滤
# ----------------------------
EXCLUDE_ABSOLUTE_POSITION_FEATURES = True
EXCLUDE_NEIGHBOR_DENSITY_FEATURES = False
MIN_COMMON_FEATURE_COUNT = 8
ABSOLUTE_FEATURE_NORMALIZATION = PIPELINE_CONFIG["absolute_feature_normalization"]
# options:
# - "none": use raw features directly
# - "per_image_robust": robust-center absolute-scale features within each image
EXCLUDE_INTENSITY_FEATURES = True

# ----------------------------
# Heatmap score / 偏离程度连续热图设置
# 核心思路：
# 1) hard label 继续用 raw probability + threshold
# 2) heatmap 不直接用 raw probability 着色
# 3) 而是把 raw probability 转成 deviation_score（拉伸后的连续分数）
# 这样颜色会从绿到红更细、更有层次
# ----------------------------
HEATMAP_REFERENCE_SOURCE = "inference_global"
# options:
# - "inference_global": 用本次推理全体 nuclei 的分布做拉伸，视觉层次最丰富
# - "validation_global": 用验证集分布做拉伸，跨批次可比性稍好

HEATMAP_STRETCH_SPACE = "logit"
# options:
# - "probability": 直接对 probability 做 quantile stretch
# - "logit": 先转 logit 再拉伸，通常更能把中高分细节展开

HEATMAP_LOW_QUANTILE = 0.02
HEATMAP_HIGH_QUANTILE = 0.98
HEATMAP_GAMMA = 0.85
HEATMAP_SCORE_MODE = "threshold_aware"
# options:
# - "threshold_aware": red only appears for nuclei above DEVIATED_THRESHOLD
# - "quantile_stretch": original continuous quantile-stretched probability heatmap
# gamma < 1 会增强中高分细节；>1 会压暗中间层次

PER_IMAGE_HEATMAP_STRETCH = False
# False: 全局统一拉伸，图与图之间可比
# True : 每张图单独拉伸，单图内部层次更明显，但跨图不可直接比较

# ----------------------------
# Overlay visualization / 可视化
# ----------------------------
HARD_OVERLAY_ALPHA = 0.45
HEATMAP_OVERLAY_ALPHA = 0.50
BOUNDARY_BRIGHTNESS = 255
SAVE_HARD_OVERLAY = True
SAVE_HEATMAP_OVERLAY = True
MAKE_INFER_UMAP = True
INFER_UMAP_MODE = PIPELINE_CONFIG["infer_umap_mode"]
# options:
# - "per_image_fit": fit UMAP independently for each image
# - "global_fit_then_split": fit one UMAP on all inference nuclei, then split by image
INFER_UMAP_N_NEIGHBORS = 30
INFER_UMAP_MIN_DIST = 0.10
INFER_UMAP_N_EPOCHS = 200
INFER_UMAP_RANDOM_STATE = None

INTERNAL_CLUSTER_NEGATIVE = "undifferentiated"
INTERNAL_CLUSTER_POSITIVE = "deviated"
OUTPUT_CLUSTER1_LABEL = "cluster1"
OUTPUT_CLUSTER2_LABEL = "cluster2"
OUTPUT_UNCERTAIN_LABEL = "uncertain"
OUTPUT_LABEL_EXPLANATION = {
    OUTPUT_CLUSTER1_LABEL: "Model cluster assigned to the lower-probability reference side during supervised classification.",
    OUTPUT_CLUSTER2_LABEL: "Model cluster assigned to the higher-probability reference side during supervised classification.",
    OUTPUT_UNCERTAIN_LABEL: "Intermediate nuclei that do not pass either hard threshold.",
}

COLOR_CLUSTER2 = np.array([230, 50, 50], dtype=np.uint8)      # red
COLOR_CLUSTER1 = np.array([60, 190, 90], dtype=np.uint8)      # green
COLOR_UNCERTAIN = np.array([245, 205, 70], dtype=np.uint8)    # yellow


# ============================================================
# 1) Utility functions / 工具函数
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"[fail] Cannot read CSV: {path}\nLast error: {last_err}")


def normalize_stem_like(x: object) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    s = s.lower().strip()

    suffixes = [
        "_mask",
        "_masks",
        "_qc_keep_mask",
        "_qc_keep",
        "_overlay",
        "_prob",
        "_hard",
    ]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True

    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_nucleus_id(x: object) -> str:
    if pd.isna(x):
        return ""
    try:
        fx = float(x)
        if math.isfinite(fx) and abs(fx - round(fx)) < 1e-6:
            return str(int(round(fx)))
        return str(x).strip()
    except Exception:
        s = str(x).strip()
        m = re.match(r"^([0-9]+)\.0+$", s)
        if m:
            return m.group(1)
        return s


def choose_column_by_keywords(
    columns: Sequence[str],
    exact_priority: Sequence[str],
    contains_priority: Sequence[str],
    exclude: Optional[Sequence[str]] = None,
) -> Optional[str]:
    exclude = set(x.lower() for x in (exclude or []))
    cols = list(columns)

    lower_map = {c.lower(): c for c in cols}
    for key in exact_priority:
        if key.lower() in lower_map and key.lower() not in exclude:
            return lower_map[key.lower()]

    scored = []
    for c in cols:
        cl = c.lower()
        if cl in exclude:
            continue
        score = 0
        for i, kw in enumerate(contains_priority):
            if kw.lower() in cl:
                score += 100 - i
        if score > 0:
            scored.append((score, len(cl), c))

    if scored:
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored[0][2]
    return None


def choose_image_col(df: pd.DataFrame, manual: Optional[str] = None) -> Optional[str]:
    if manual and manual in df.columns:
        return manual
    return choose_column_by_keywords(
        df.columns,
        exact_priority=[
            "image_name",
            "image_file",
            "filename",
            "file_name",
            "image",
            "img_name",
            "img_file",
            "mask_name",
            "source_image",
            "image_path",
        ],
        contains_priority=[
            "image_name",
            "image_file",
            "filename",
            "file_name",
            "img_name",
            "img_file",
            "source_image",
            "image_path",
            "mask_name",
            "image",
            "file",
        ],
        exclude=[TRAIN_LABEL_COL, "predicted_label"],
    )


def choose_nucleus_col(df: pd.DataFrame, manual: Optional[str] = None) -> Optional[str]:
    if manual and manual in df.columns:
        return manual
    return choose_column_by_keywords(
        df.columns,
        exact_priority=[
            "nucleus_label",
            "mask_label",
            "label_id",
            "nucleus_id",
            "cell_label",
            "cell_id",
            "object_id",
            "label",
            "id",
        ],
        contains_priority=[
            "nucleus_label",
            "mask_label",
            "label_id",
            "nucleus_id",
            "cell_label",
            "cell_id",
            "object_id",
            "nucleus",
            "label",
            "id",
        ],
        exclude=[TRAIN_LABEL_COL, "cluster_label", "predicted_label"],
    )


def build_merge_key(
    df: pd.DataFrame,
    image_col: Optional[str],
    nucleus_col: Optional[str],
    key_name: str = "__merge_key__",
) -> pd.DataFrame:
    out = df.copy()

    if image_col is not None and nucleus_col is not None:
        out[key_name] = (
            out[image_col].map(normalize_stem_like).astype(str)
            + "||"
            + out[nucleus_col].map(normalize_nucleus_id).astype(str)
        )
        return out

    if nucleus_col is not None:
        nuc_series = out[nucleus_col].map(normalize_nucleus_id).astype(str)
        if nuc_series.duplicated().any():
            raise RuntimeError(
                "[fail] Only nucleus id column is available, but nucleus ids are not unique. "
                "Cannot safely merge label table to feature table."
            )
        out[key_name] = nuc_series
        return out

    raise RuntimeError("[fail] Cannot build merge key because nucleus id column was not found.")


def merge_training_feature_and_label(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_image_col: Optional[str],
    feature_nucleus_col: Optional[str],
    label_image_col: Optional[str],
    label_nucleus_col: Optional[str],
    label_col: str,
) -> pd.DataFrame:
    if label_col in feature_df.columns:
        n_labeled = feature_df[label_col].notna().sum()
        if n_labeled > 0:
            print(
                f"[info] Training feature CSV already contains label column `{label_col}`; "
                "skip merge and use it directly."
            )
            return feature_df.copy()

    ff = build_merge_key(feature_df, feature_image_col, feature_nucleus_col, key_name="__merge_key__")
    lf = build_merge_key(label_df, label_image_col, label_nucleus_col, key_name="__merge_key__")

    if label_col not in lf.columns:
        raise RuntimeError(f"[fail] Label CSV does not contain `{label_col}`.")

    keep_cols = ["__merge_key__", label_col]
    merged = ff.merge(lf[keep_cols], on="__merge_key__", how="left")

    n_labeled = merged[label_col].notna().sum()
    print(f"[info] Training merge done: labeled rows = {n_labeled} / {len(merged)}")

    if n_labeled == 0 and ALLOW_ROW_ORDER_FALLBACK:
        if len(feature_df) == len(label_df):
            print("[warn] Merge by key failed. Falling back to row-order alignment because ALLOW_ROW_ORDER_FALLBACK=True.")
            tmp = feature_df.copy().reset_index(drop=True)
            tmp[label_col] = label_df[label_col].reset_index(drop=True)
            return tmp
        raise RuntimeError(
            "[fail] Merge by key failed and row-order fallback is not possible because row counts differ."
        )

    if n_labeled == 0:
        raise RuntimeError(
            "[fail] 0 training rows received labels after merge.\n"
            "Please inspect image/nucleus key columns or set manual column names."
        )

    return merged


def is_probably_position_feature(col: str) -> bool:
    cl = col.lower().strip()
    exact_bad = {
        "x", "y", "cx", "cy", "row", "col",
        "center_x", "center_y", "centroid_x", "centroid_y",
        "centroid-0", "centroid-1", "bbox-0", "bbox-1", "bbox-2", "bbox-3",
    }
    contains_bad = [
        "centroid",
        "center_x",
        "center_y",
        "bbox",
        "coord",
        "position",
        "x_pos",
        "y_pos",
        "left",
        "top",
        "right",
        "bottom",
    ]
    if cl in exact_bad:
        return True
    return any(k in cl for k in contains_bad)


def is_probably_density_feature(col: str) -> bool:
    cl = col.lower().strip()
    density_kws = [
        "neighbor", "neighbour", "knn", "nn_", "density", "local_density",
        "crowd", "sparse", "adjacent", "nearby", "dist_to_nn", "nearest"
    ]
    return any(k in cl for k in density_kws)


def is_leakage_or_nonfeature_col(
    col: str,
    image_col: Optional[str],
    nucleus_col: Optional[str],
    label_col: str,
) -> bool:
    cl = col.lower()

    exact_bad = {
        image_col.lower() if image_col else "",
        nucleus_col.lower() if nucleus_col else "",
        label_col.lower(),
        "predicted_label",
        "pred_label",
        "cluster",
        "__merge_key__",
        "qc_keep",
    }
    if cl in exact_bad:
        return True

    contains_bad = [
        "final_state",
        "pseudo",
        "pred",
        "prob",
        "posterior",
        "confidence",
        "uncertain",
        "deviated_score",
        "undiff_score",
        "cluster",
        "gmm",
        "class",
        "pca",
        "tsne",
        "umap",
        "embed",
        "embedding",
        "main_state",
        "main_label",
        "main_cluster",
        "aux_state",
        "aux_label",
        "veto",
        "morphveto",
        "edge_flag",
        "edge_red",
        "edge_green",
        "innerfit_state",
        "final_label",
        "state_label",
        "pseudo_label",
    ]
    if any(k in cl for k in contains_bad):
        return True

    if EXCLUDE_ABSOLUTE_POSITION_FEATURES and is_probably_position_feature(col):
        return True

    if EXCLUDE_NEIGHBOR_DENSITY_FEATURES and is_probably_density_feature(col):
        return True

    return False


def select_common_feature_columns(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    train_image_col: Optional[str],
    train_nucleus_col: Optional[str],
    infer_image_col: Optional[str],
    infer_nucleus_col: Optional[str],
    label_col: str,
) -> Tuple[List[str], List[str]]:
    train_cols = set(train_df.columns)
    infer_cols = set(infer_df.columns)
    common_cols = sorted(train_cols.intersection(infer_cols))

    final_cols = []
    removed_cols = []

    for c in common_cols:
        if is_leakage_or_nonfeature_col(c, train_image_col, train_nucleus_col, label_col):
            removed_cols.append(c)
            continue
        if is_leakage_or_nonfeature_col(c, infer_image_col, infer_nucleus_col, label_col):
            removed_cols.append(c)
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            removed_cols.append(c)
            continue
        if not pd.api.types.is_numeric_dtype(infer_df[c]):
            removed_cols.append(c)
            continue
        final_cols.append(c)

    return final_cols, removed_cols


def prefer_physical_feature_columns(feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    cols = list(feature_cols)
    col_set = set(cols)
    preferred = []
    dropped = []

    replacement_map = {
        "nn1_distance": "nn1_distance_um",
        "knn6_distance_mean": "knn6_distance_mean_um",
        "knn6_distance_std": "knn6_distance_std_um",
        "adaptive_radius": "adaptive_radius_um",
        "local_density": "local_density_per_um2",
        "area": "area_um2",
        "convex_area": "convex_area_um2",
        "filled_area": "filled_area_um2",
        "perimeter": "perimeter_um",
        "equivalent_diameter": "equivalent_diameter_um",
        "major_axis_length": "major_axis_length_um",
        "minor_axis_length": "minor_axis_length_um",
        "bbox_height": "bbox_height_um",
        "bbox_width": "bbox_width_um",
        "nb_area_mean": "nb_area_mean_um2",
        "nb_area_std": "nb_area_std_um2",
        "nb_distance_mean": "nb_distance_mean_um",
        "nb_distance_std": "nb_distance_std_um",
        "adaptive_nb_area_mean": "adaptive_nb_area_mean_um2",
        "adaptive_nb_area_std": "adaptive_nb_area_std_um2",
        "fixed_nb_area_mean": "fixed_nb_area_mean_um2",
        "fixed_nb_area_std": "fixed_nb_area_std_um2",
        "fixed_nb_distance_mean": "fixed_nb_distance_mean_um",
        "fixed_nb_distance_std": "fixed_nb_distance_std_um",
    }

    for col in cols:
        preferred_col = replacement_map.get(col)
        if preferred_col and preferred_col in col_set:
            dropped.append(col)
            continue
        preferred.append(col)

    return preferred, dropped


def filter_unusable_feature_columns(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    kept = []
    removed = []

    for col in feature_cols:
        train_vals = pd.to_numeric(train_df[col], errors="coerce")
        infer_vals = pd.to_numeric(infer_df[col], errors="coerce")

        if train_vals.notna().sum() == 0:
            removed.append(f"{col} [train_all_missing]")
            continue
        if infer_vals.notna().sum() == 0:
            removed.append(f"{col} [infer_all_missing]")
            continue

        kept.append(col)

    return kept, removed


def is_intensity_feature(col: str) -> bool:
    cl = col.lower().strip()
    return ("intensity" in cl) or (cl in {"is_bright", "is_bright_mean", "is_bright_range"})


def filter_intensity_feature_columns(feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    kept = []
    removed = []
    for col in feature_cols:
        if is_intensity_feature(col):
            removed.append(col)
            continue
        kept.append(col)
    return kept, removed


def is_absolute_scale_feature(col: str) -> bool:
    cl = col.lower().strip()

    if cl.startswith("is_") or cl in {"touches_border", "qc_exclude"}:
        return False

    relative_markers = [
        "aspect_ratio",
        "circularity",
        "eccentricity",
        "solidity",
        "extent",
        "density",
        "neighbor",
        "neighbour",
        "knn",
        "nn1_distance",
        "nn6_distance",
        "nb_distance",
        "distance",
        "count",
    ]
    if any(k in cl for k in relative_markers):
        return False

    absolute_markers = [
        "area",
        "perimeter",
        "diameter",
        "major_axis",
        "minor_axis",
        "bbox_height",
        "bbox_width",
        "radius",
        "intensity",
        "pixel_size",
        "image_height",
        "image_width",
    ]
    return any(k in cl for k in absolute_markers)


def robust_normalize_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    g = groups.astype(str)

    global_med = float(x.median()) if x.notna().any() else 0.0
    global_q25 = float(x.quantile(0.25)) if x.notna().any() else np.nan
    global_q75 = float(x.quantile(0.75)) if x.notna().any() else np.nan
    global_iqr = global_q75 - global_q25 if np.isfinite(global_q25) and np.isfinite(global_q75) else np.nan
    if not np.isfinite(global_iqr) or global_iqr <= 1e-9:
        global_std = float(x.std(ddof=0)) if x.notna().sum() > 1 else np.nan
        global_iqr = global_std if np.isfinite(global_std) and global_std > 1e-9 else 1.0

    group_med = x.groupby(g).transform("median").fillna(global_med)
    group_q25 = x.groupby(g).transform(lambda s: s.quantile(0.25))
    group_q75 = x.groupby(g).transform(lambda s: s.quantile(0.75))
    group_iqr = (group_q75 - group_q25).replace(0, np.nan)
    group_iqr = group_iqr.where(group_iqr > 1e-9, np.nan).fillna(global_iqr)

    return (x - group_med) / group_iqr


def apply_absolute_feature_normalization(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_image_col: Optional[str],
    infer_image_col: Optional[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    if mode == "none":
        return train_df, infer_df, list(feature_cols), []

    if mode != "per_image_robust":
        raise ValueError(f"Unsupported ABSOLUTE_FEATURE_NORMALIZATION: {mode}")

    train_out = train_df.copy()
    infer_out = infer_df.copy()
    updated_cols: List[str] = []
    normalized_original_cols: List[str] = []

    for col in feature_cols:
        if not is_absolute_scale_feature(col):
            updated_cols.append(col)
            continue

        new_col = f"{col}__img_rel"
        train_groups = train_out[train_image_col] if train_image_col is not None else pd.Series("all", index=train_out.index)
        infer_groups = infer_out[infer_image_col] if infer_image_col is not None else pd.Series("all", index=infer_out.index)

        train_out[new_col] = robust_normalize_by_group(train_out[col], train_groups)
        infer_out[new_col] = robust_normalize_by_group(infer_out[col], infer_groups)

        updated_cols.append(new_col)
        normalized_original_cols.append(col)

    return train_out, infer_out, updated_cols, normalized_original_cols


def label_to_binary(y: pd.Series) -> pd.Series:
    norm = y.astype(str).str.strip().str.lower()
    mapping = {
        INTERNAL_CLUSTER_POSITIVE: 1,
        INTERNAL_CLUSTER_NEGATIVE: 0,
        OUTPUT_CLUSTER2_LABEL: 1,
        OUTPUT_CLUSTER1_LABEL: 0,
    }
    return norm.map(mapping)


def internal_to_output_label(label: str) -> str:
    s = str(label).strip().lower()
    if s == INTERNAL_CLUSTER_POSITIVE:
        return OUTPUT_CLUSTER2_LABEL
    if s == INTERNAL_CLUSTER_NEGATIVE:
        return OUTPUT_CLUSTER1_LABEL
    if s == OUTPUT_CLUSTER2_LABEL:
        return OUTPUT_CLUSTER2_LABEL
    if s == OUTPUT_CLUSTER1_LABEL:
        return OUTPUT_CLUSTER1_LABEL
    return OUTPUT_UNCERTAIN_LABEL


def normalize_training_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {INTERNAL_CLUSTER_POSITIVE, OUTPUT_CLUSTER2_LABEL}:
        return INTERNAL_CLUSTER_POSITIVE
    if s in {INTERNAL_CLUSTER_NEGATIVE, OUTPUT_CLUSTER1_LABEL}:
        return INTERNAL_CLUSTER_NEGATIVE
    return OUTPUT_UNCERTAIN_LABEL


def predicted_label_from_prob(p_dev: float) -> str:
    if p_dev >= DEVIATED_THRESHOLD:
        return OUTPUT_CLUSTER2_LABEL
    if p_dev <= UNDIFF_THRESHOLD:
        return OUTPUT_CLUSTER1_LABEL
    return OUTPUT_UNCERTAIN_LABEL


def score_to_rgb(score_01: float) -> np.ndarray:
    s = float(np.clip(score_01, 0.0, 1.0))
    if s <= 0.5:
        t = s / 0.5
        c = (1 - t) * COLOR_CLUSTER1.astype(float) + t * COLOR_UNCERTAIN.astype(float)
    else:
        t = (s - 0.5) / 0.5
        c = (1 - t) * COLOR_UNCERTAIN.astype(float) + t * COLOR_CLUSTER2.astype(float)
    return np.clip(np.round(c), 0, 255).astype(np.uint8)


def hard_label_to_rgb(label: str) -> np.ndarray:
    s = str(label).strip().lower()
    if s == OUTPUT_CLUSTER2_LABEL:
        return COLOR_CLUSTER2
    if s == OUTPUT_CLUSTER1_LABEL:
        return COLOR_CLUSTER1
    return COLOR_UNCERTAIN


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def stretch_to_unit_interval(
    values: np.ndarray,
    ref_values: np.ndarray,
    low_q: float,
    high_q: float,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    v = np.asarray(values, dtype=float)
    r = np.asarray(ref_values, dtype=float)

    r = r[np.isfinite(r)]
    if len(r) == 0:
        out = np.full_like(v, 0.5, dtype=float)
        info = {
            "low_ref": np.nan,
            "high_ref": np.nan,
            "method": "empty_ref_fallback_0.5",
        }
        return out, info

    lo = float(np.quantile(r, low_q))
    hi = float(np.quantile(r, high_q))

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo + 1e-12:
        vv = v[np.isfinite(v)]
        if len(vv) == 0:
            out = np.full_like(v, 0.5, dtype=float)
            info = {
                "low_ref": lo,
                "high_ref": hi,
                "method": "degenerate_ref_fallback_0.5",
            }
            return out, info

        vmin = float(np.min(vv))
        vmax = float(np.max(vv))
        if vmax <= vmin + 1e-12:
            out = np.full_like(v, 0.5, dtype=float)
            info = {
                "low_ref": lo,
                "high_ref": hi,
                "method": "degenerate_values_fallback_0.5",
            }
            return out, info

        out = (v - vmin) / (vmax - vmin)
        out = np.clip(out, 0.0, 1.0)
        if gamma != 1.0:
            out = np.power(out, gamma)
        info = {
            "low_ref": lo,
            "high_ref": hi,
            "method": "value_minmax_fallback",
        }
        return out, info

    out = (v - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, gamma)

    info = {
        "low_ref": lo,
        "high_ref": hi,
        "method": "quantile_stretch",
    }
    return out, info


def compute_deviation_score(
    prob_values: np.ndarray,
    ref_prob_values: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if HEATMAP_SCORE_MODE == "threshold_aware":
        p = np.asarray(prob_values, dtype=float)
        eps = 1e-9
        undiff_thr = float(np.clip(UNDIFF_THRESHOLD, 0.0, 1.0))
        dev_thr = float(np.clip(DEVIATED_THRESHOLD, 0.0, 1.0))
        if dev_thr < undiff_thr:
            dev_thr = undiff_thr

        score = np.zeros_like(p, dtype=float)

        green_mask = p <= undiff_thr
        if undiff_thr > eps:
            score[green_mask] = 0.5 * np.clip(p[green_mask] / undiff_thr, 0.0, 1.0)
        else:
            score[green_mask] = 0.0

        uncertain_mask = (p > undiff_thr) & (p < dev_thr)
        if dev_thr > undiff_thr + eps:
            band = (p[uncertain_mask] - undiff_thr) / (dev_thr - undiff_thr)
            score[uncertain_mask] = 0.5 + 0.34 * np.clip(band, 0.0, 1.0)
        else:
            score[uncertain_mask] = 0.5

        red_mask = p >= dev_thr
        if dev_thr < 1.0 - eps:
            tail = (p[red_mask] - dev_thr) / (1.0 - dev_thr)
            score[red_mask] = 0.84 + 0.16 * np.clip(tail, 0.0, 1.0)
        else:
            score[red_mask] = 0.84

        info = {
            "mode": "threshold_aware",
            "undiff_threshold": undiff_thr,
            "deviated_threshold": dev_thr,
            "green_to_yellow_cap": 0.5,
            "uncertain_to_orange_cap": 0.84,
        }
        return score, info

    p = np.asarray(prob_values, dtype=float)
    ref = np.asarray(ref_prob_values, dtype=float)

    if HEATMAP_STRETCH_SPACE == "logit":
        p_space = prob_to_logit(p)
        ref_space = prob_to_logit(ref)
        score, info = stretch_to_unit_interval(
            p_space,
            ref_space,
            HEATMAP_LOW_QUANTILE,
            HEATMAP_HIGH_QUANTILE,
            HEATMAP_GAMMA,
        )
        info["space"] = "logit"
        return score, info

    score, info = stretch_to_unit_interval(
        p,
        ref,
        HEATMAP_LOW_QUANTILE,
        HEATMAP_HIGH_QUANTILE,
        HEATMAP_GAMMA,
    )
    info["space"] = "probability"
    return score, info


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).astype(np.float32)
    finite_mask = np.isfinite(a)
    if not finite_mask.any():
        return np.zeros_like(a, dtype=np.uint8)

    vals = a[finite_mask]
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99.5)
    if not np.isfinite(lo):
        lo = float(vals.min())
    if not np.isfinite(hi):
        hi = float(vals.max())
    if hi <= lo:
        hi = lo + 1.0

    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return (a * 255.0).round().astype(np.uint8)


def normalize_rgb_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    out = np.zeros_like(a, dtype=np.uint8)
    for ch in range(a.shape[2]):
        out[..., ch] = normalize_to_uint8(a[..., ch])
    return out


def read_image_as_rgb(path: Path) -> np.ndarray:
    img = None
    err_list = []

    try:
        import imageio.v3 as iio
        img = iio.imread(path)
    except Exception as e:
        err_list.append(f"imageio: {e}")

    if img is None:
        try:
            import tifffile
            img = tifffile.imread(path)
        except Exception as e:
            err_list.append(f"tifffile: {e}")

    if img is None:
        try:
            from PIL import Image
            img = np.array(Image.open(path))
        except Exception as e:
            err_list.append(f"PIL: {e}")

    if img is None:
        raise RuntimeError(f"[fail] Cannot read image: {path}\n" + "\n".join(err_list))

    img = np.asarray(img)

    if img.ndim == 2:
        gray = normalize_to_uint8(img)
        return np.stack([gray, gray, gray], axis=-1)

    if img.ndim == 3:
        if img.shape[2] >= 3:
            return normalize_rgb_to_uint8(img[..., :3])
        if img.shape[0] >= 3 and img.shape[2] not in [3, 4]:
            rgb = np.moveaxis(img[:3, ...], 0, -1)
            return normalize_rgb_to_uint8(rgb)

    raise RuntimeError(f"[fail] Unsupported image shape: {img.shape}")


def try_save_rgb(path: Path, rgb: np.ndarray) -> None:
    err_list = []

    try:
        import imageio.v3 as iio
        iio.imwrite(path, rgb)
        return
    except Exception as e:
        err_list.append(f"imageio: {e}")

    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return
    except Exception as e:
        err_list.append(f"PIL: {e}")

    raise RuntimeError(f"[fail] Cannot save image to {path}\n" + "\n".join(err_list))


def load_mask_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray):
        if arr.dtype == object and arr.shape == ():
            obj = arr.item()
            if isinstance(obj, dict):
                for k in ["masks", "mask", "labels", "label"]:
                    if k in obj:
                        return squeeze_2d_mask(np.asarray(obj[k]))
                raise RuntimeError(f"[fail] Object npy dict at {path} does not contain mask-like key.")
        return squeeze_2d_mask(arr)

    raise RuntimeError(f"[fail] Unsupported mask npy format: {path}")


def squeeze_2d_mask(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.astype(np.int32)
    if a.ndim == 3:
        if a.shape[0] == 1:
            return a[0].astype(np.int32)
        if a.shape[-1] == 1:
            return a[..., 0].astype(np.int32)
    raise RuntimeError(f"[fail] Mask array is not 2D-compatible, shape={a.shape}")


def compute_boundaries(label_img: np.ndarray) -> np.ndarray:
    lab = np.asarray(label_img)
    boundary = np.zeros(lab.shape, dtype=bool)

    boundary[:-1, :] |= (lab[:-1, :] != lab[1:, :])
    boundary[1:, :] |= (lab[:-1, :] != lab[1:, :])
    boundary[:, :-1] |= (lab[:, :-1] != lab[:, 1:])
    boundary[:, 1:] |= (lab[:, :-1] != lab[:, 1:])

    boundary &= (lab > 0)
    return boundary


def blend_overlay(
    background_rgb: np.ndarray,
    color_layer: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
    boundary_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    bg = background_rgb.astype(np.float32)
    cl = color_layer.astype(np.float32)
    vm = valid_mask.astype(bool)

    out = bg.copy()
    out[vm] = (1.0 - alpha) * bg[vm] + alpha * cl[vm]

    if boundary_mask is not None:
        out[boundary_mask] = BOUNDARY_BRIGHTNESS

    return np.clip(out, 0, 255).astype(np.uint8)


def build_file_index(folder: Path, exts: Sequence[str]) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    if not folder.exists():
        raise FileNotFoundError(f"[fail] Folder not found: {folder}")
    ext_set = set(e.lower() for e in exts)

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_set:
            key = normalize_stem_like(p.stem)
            idx.setdefault(key, []).append(p)
    return idx


def resolve_matching_file(raw_name: object, file_index: Dict[str, List[Path]]) -> Optional[Path]:
    key = normalize_stem_like(raw_name)
    if key in file_index and len(file_index[key]) >= 1:
        return sorted(file_index[key])[0]

    hits = []
    for k, paths in file_index.items():
        if key and (key in k or k in key):
            hits.extend(paths)
    if hits:
        return sorted(hits)[0]
    return None


def save_xgb_gain_importance(model: XGBClassifier, feature_cols: List[str], out_csv: Path) -> pd.DataFrame:
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    weight_dict = booster.get_score(importance_type="weight")
    cover_dict = booster.get_score(importance_type="cover")

    rows = []
    for feat in feature_cols:
        rows.append(
            {
                "feature": feat,
                "gain": float(gain_dict.get(feat, 0.0)),
                "weight": float(weight_dict.get(feat, 0.0)),
                "cover": float(cover_dict.get(feat, 0.0)),
            }
        )
    fi_df = pd.DataFrame(rows).sort_values(["gain", "weight"], ascending=False)
    fi_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return fi_df


def save_label_scatter_figure(
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    out_path: Path,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xticks: Optional[Sequence[float]] = None,
    yticks: Optional[Sequence[float]] = None,
) -> None:
    plt.figure(figsize=(8, 8))
    color_map = {
        OUTPUT_CLUSTER2_LABEL: "red",
        OUTPUT_CLUSTER1_LABEL: "green",
        OUTPUT_UNCERTAIN_LABEL: "yellow",
    }
    for name, sub in df_plot.groupby(label_col, dropna=False):
        plt.scatter(
            sub[x_col],
            sub[y_col],
            s=8,
            alpha=0.7,
            label=str(name),
            color=color_map.get(str(name), "gray"),
        )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    ax = plt.gca()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
    ax.set_aspect("equal", adjustable="box")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def extract_timepoint_index(image_name: object) -> Optional[int]:
    s = "" if pd.isna(image_name) else str(image_name)
    m = re.search(r"time0*([0-9]+)", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def save_timepoint_ratio_outputs(pred_df: pd.DataFrame, image_col: str, label_col: str, out_csv: Path, out_fig: Path) -> pd.DataFrame:
    work = pred_df[[image_col, label_col]].copy()
    work["time_index"] = work[image_col].map(extract_timepoint_index)
    work = work[work["time_index"].notna()].copy()
    if len(work) == 0:
        return pd.DataFrame()

    work["time_index"] = work["time_index"].astype(int)
    summary_rows = []
    for (time_index, image_name), sub in work.groupby(["time_index", image_col], dropna=False):
        total = len(sub)
        counts = sub[label_col].value_counts(dropna=False)
        row = {
            "time_index": int(time_index),
            "image_name": image_name,
            "n_nuclei": int(total),
            OUTPUT_CLUSTER1_LABEL: int(counts.get(OUTPUT_CLUSTER1_LABEL, 0)),
            OUTPUT_CLUSTER2_LABEL: int(counts.get(OUTPUT_CLUSTER2_LABEL, 0)),
            OUTPUT_UNCERTAIN_LABEL: int(counts.get(OUTPUT_UNCERTAIN_LABEL, 0)),
            f"{OUTPUT_CLUSTER1_LABEL}_ratio": float(counts.get(OUTPUT_CLUSTER1_LABEL, 0) / total),
            f"{OUTPUT_CLUSTER2_LABEL}_ratio": float(counts.get(OUTPUT_CLUSTER2_LABEL, 0) / total),
            f"{OUTPUT_UNCERTAIN_LABEL}_ratio": float(counts.get(OUTPUT_UNCERTAIN_LABEL, 0) / total),
        }
        summary_rows.append(row)

    ratio_df = pd.DataFrame(summary_rows).sort_values(["time_index", "image_name"]).reset_index(drop=True)
    ratio_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(ratio_df["time_index"], ratio_df[f"{OUTPUT_CLUSTER1_LABEL}_ratio"], marker="o", color="green", label=OUTPUT_CLUSTER1_LABEL)
    plt.plot(ratio_df["time_index"], ratio_df[f"{OUTPUT_CLUSTER2_LABEL}_ratio"], marker="o", color="red", label=OUTPUT_CLUSTER2_LABEL)
    plt.plot(ratio_df["time_index"], ratio_df[f"{OUTPUT_UNCERTAIN_LABEL}_ratio"], marker="o", color="#d9a400", label=OUTPUT_UNCERTAIN_LABEL)
    plt.xlabel("Time index")
    plt.ylabel("Fraction of nuclei")
    plt.title("Timelapse cluster ratios by frame")
    plt.xlim(1, 18)
    plt.xticks(list(range(1, 19)))
    plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()

    return ratio_df


def compute_square_axis_meta(df_plot: pd.DataFrame, x_col: str, y_col: str, n_ticks: int = 6) -> Dict[str, object]:
    x = pd.to_numeric(df_plot[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_plot[y_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return {}

    x = x[finite]
    y = y[finite]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = 0.05 * span
    half = 0.5 * span + pad
    xlim = (xmid - half, xmid + half)
    ylim = (ymid - half, ymid + half)
    ticks = np.linspace(-half, half, num=n_ticks)
    xticks = xmid + ticks
    yticks = ymid + ticks
    return {"xlim": xlim, "ylim": ylim, "xticks": xticks, "yticks": yticks}


def save_inference_umap_by_image(
    pred_df: pd.DataFrame,
    image_col: str,
    nucleus_col: str,
    out_dir: Path,
    feature_df_scaled: Optional[pd.DataFrame] = None,
    use_existing_embedding: bool = False,
    shared_axis_meta: Optional[Dict[str, object]] = None,
) -> Tuple[int, List[str]]:
    ensure_dir(out_dir)
    saved_images = []

    for image_name, group in pred_df.groupby(image_col, dropna=False):
        group = group.copy()
        if len(group) < 3:
            continue

        idx = group.index
        if use_existing_embedding:
            if "umap_1" not in group.columns or "umap_2" not in group.columns:
                raise RuntimeError("Existing UMAP columns not found for per-image export.")
        else:
            if feature_df_scaled is None:
                raise RuntimeError("feature_df_scaled is required when fitting per-image UMAP.")
            X_group = feature_df_scaled.loc[idx]
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(INFER_UMAP_N_NEIGHBORS, max(2, len(X_group) - 1)),
                min_dist=INFER_UMAP_MIN_DIST,
                n_epochs=INFER_UMAP_N_EPOCHS,
                random_state=INFER_UMAP_RANDOM_STATE,
            )
            X_umap = reducer.fit_transform(X_group)
            group["umap_1"] = X_umap[:, 0]
            group["umap_2"] = X_umap[:, 1]

        safe_name = normalize_stem_like(image_name) or f"image_{len(saved_images)+1:03d}"
        csv_path = out_dir / f"{safe_name}_umap.csv"
        fig_path = out_dir / f"{safe_name}_umap.png"

        group[
            [
                image_col,
                nucleus_col,
                "predicted_label",
                "cluster2_probability",
                "cluster1_probability",
                "confidence",
                "deviation_score",
                "umap_1",
                "umap_2",
            ]
        ].to_csv(csv_path, index=False, encoding="utf-8-sig")
        save_label_scatter_figure(
            group,
            "umap_1",
            "umap_2",
            "predicted_label",
            f"Per-image inference UMAP ({image_name})",
            fig_path,
            xlim=None if shared_axis_meta is None else shared_axis_meta.get("xlim"),
            ylim=None if shared_axis_meta is None else shared_axis_meta.get("ylim"),
            xticks=None if shared_axis_meta is None else shared_axis_meta.get("xticks"),
            yticks=None if shared_axis_meta is None else shared_axis_meta.get("yticks"),
        )
        saved_images.append(str(image_name))

    return len(saved_images), saved_images


# ============================================================
# 2) Main / 主流程
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    hard_dir = OUTPUT_ROOT / "overlays_hard"
    heatmap_dir = OUTPUT_ROOT / "overlays_deviation_score"
    ensure_dir(hard_dir)
    ensure_dir(heatmap_dir)

    print("============================================================")
    print("Stage 2 supervised learning with XGBoost")
    print("第二阶段监督学习（XGBoost）")
    print("============================================================")

    # ----------------------------
    # Load training feature table / 训练特征表
    # ----------------------------
    if not TRAIN_FEATURE_CSV.exists():
        raise FileNotFoundError(f"[fail] TRAIN_FEATURE_CSV not found: {TRAIN_FEATURE_CSV}")
    train_feature_df = read_csv_robust(TRAIN_FEATURE_CSV)
    print(f"[ok] Train feature CSV loaded: {TRAIN_FEATURE_CSV}")
    print(f"     rows={len(train_feature_df)}, cols={len(train_feature_df.columns)}")

    # ----------------------------
    # Load training label table / 训练标签表
    # ----------------------------
    if not TRAIN_LABEL_CSV.exists():
        raise FileNotFoundError(f"[fail] TRAIN_LABEL_CSV not found: {TRAIN_LABEL_CSV}")
    train_label_df = read_csv_robust(TRAIN_LABEL_CSV)
    print(f"[ok] Train label CSV loaded: {TRAIN_LABEL_CSV}")
    print(f"     rows={len(train_label_df)}, cols={len(train_label_df.columns)}")

    if TRAIN_LABEL_COL not in train_label_df.columns:
        raise RuntimeError(f"[fail] Training label CSV does not contain label column: {TRAIN_LABEL_COL}")

    # ----------------------------
    # Infer key columns / 自动识别关键列
    # ----------------------------
    train_feature_image_col = choose_image_col(train_feature_df, manual=MANUAL_IMAGE_COL_TRAIN_FEATURE)
    train_feature_nucleus_col = choose_nucleus_col(train_feature_df, manual=MANUAL_NUCLEUS_COL_TRAIN_FEATURE)

    train_label_image_col = choose_image_col(train_label_df, manual=MANUAL_IMAGE_COL_TRAIN_LABEL)
    train_label_nucleus_col = choose_nucleus_col(train_label_df, manual=MANUAL_NUCLEUS_COL_TRAIN_LABEL)

    print(f"[info] Train feature image col : {train_feature_image_col}")
    print(f"[info] Train feature nucleus col: {train_feature_nucleus_col}")
    print(f"[info] Train label image col   : {train_label_image_col}")
    print(f"[info] Train label nucleus col : {train_label_nucleus_col}")
    print(f"[info] Train label col         : {TRAIN_LABEL_COL}")

    if train_feature_nucleus_col is None:
        raise RuntimeError("[fail] Could not infer nucleus id column in TRAIN_FEATURE_CSV.")
    if train_label_nucleus_col is None:
        raise RuntimeError("[fail] Could not infer nucleus id column in TRAIN_LABEL_CSV.")

    # ----------------------------
    # Merge training features + labels / 合并训练特征与标签
    # ----------------------------
    train_df = merge_training_feature_and_label(
        feature_df=train_feature_df,
        label_df=train_label_df,
        feature_image_col=train_feature_image_col,
        feature_nucleus_col=train_feature_nucleus_col,
        label_image_col=train_label_image_col,
        label_nucleus_col=train_label_nucleus_col,
        label_col=TRAIN_LABEL_COL,
    )

    train_df[TRAIN_LABEL_COL] = train_df[TRAIN_LABEL_COL].map(normalize_training_label)
    train_df = train_df[train_df[TRAIN_LABEL_COL].isin([INTERNAL_CLUSTER_POSITIVE, INTERNAL_CLUSTER_NEGATIVE, OUTPUT_UNCERTAIN_LABEL])].copy()

    print("[info] Training label distribution after merge:")
    print(train_df[TRAIN_LABEL_COL].map(internal_to_output_label).value_counts(dropna=False).to_string())

    # ----------------------------
    # Load inference feature CSV / 读取推理特征表
    # ----------------------------
    if not INFER_FEATURE_CSV.exists():
        raise FileNotFoundError(f"[fail] INFER_FEATURE_CSV not found: {INFER_FEATURE_CSV}")
    infer_df = read_csv_robust(INFER_FEATURE_CSV)
    print(f"[ok] Inference feature CSV loaded: {INFER_FEATURE_CSV}")
    print(f"     rows={len(infer_df)}, cols={len(infer_df.columns)}")

    infer_image_col_name = choose_image_col(infer_df, manual=MANUAL_IMAGE_COL_INFER)
    infer_nucleus_col_name = choose_nucleus_col(infer_df, manual=MANUAL_NUCLEUS_COL_INFER)

    print(f"[info] Inference image col  : {infer_image_col_name}")
    print(f"[info] Inference nucleus col: {infer_nucleus_col_name}")

    if infer_image_col_name is None:
        raise RuntimeError("[fail] Could not infer image column in inference CSV.")
    if infer_nucleus_col_name is None:
        raise RuntimeError("[fail] Could not infer nucleus id column in inference CSV.")

    # ----------------------------
    # Final common feature selection / 最终公共特征选择
    # ----------------------------
    feature_cols, removed_common_cols = select_common_feature_columns(
        train_df=train_df,
        infer_df=infer_df,
        train_image_col=train_feature_image_col,
        train_nucleus_col=train_feature_nucleus_col,
        infer_image_col=infer_image_col_name,
        infer_nucleus_col=infer_nucleus_col_name,
        label_col=TRAIN_LABEL_COL,
    )
    if EXCLUDE_INTENSITY_FEATURES:
        feature_cols, removed_intensity_cols = filter_intensity_feature_columns(feature_cols)
        removed_common_cols = removed_common_cols + [f"{c} [intensity_held_out]" for c in removed_intensity_cols]
    feature_cols, dropped_px_cols = prefer_physical_feature_columns(feature_cols)
    removed_common_cols = removed_common_cols + dropped_px_cols
    feature_cols, unusable_feature_cols = filter_unusable_feature_columns(train_df, infer_df, feature_cols)
    removed_common_cols = removed_common_cols + unusable_feature_cols
    umap_feature_cols = list(feature_cols)
    train_df, infer_df, feature_cols, normalized_absolute_cols = apply_absolute_feature_normalization(
        train_df=train_df,
        infer_df=infer_df,
        feature_cols=feature_cols,
        train_image_col=train_feature_image_col,
        infer_image_col=infer_image_col_name,
        mode=ABSOLUTE_FEATURE_NORMALIZATION,
    )
    physical_feature_cols = [c for c in feature_cols if c.endswith("_um") or c.endswith("_um2") or "_per_um2" in c]

    print(f"[info] Final common feature count: {len(feature_cols)}")
    print("[info] Final feature columns used:")
    print(feature_cols)
    print(f"[info] Pipeline variant: {PIPELINE_VARIANT}")
    print(f"[info] Absolute feature normalization mode: {ABSOLUTE_FEATURE_NORMALIZATION}")
    print(f"[info] Inference UMAP mode: {INFER_UMAP_MODE}")
    print(f"[info] UMAP feature source: original common features before per-image normalization ({len(umap_feature_cols)} columns)")
    if EXCLUDE_INTENSITY_FEATURES:
        held_out_intensity_cols = [x.replace(" [intensity_held_out]", "") for x in removed_common_cols if x.endswith("[intensity_held_out]")]
        if held_out_intensity_cols:
            print(f"[info] Held out intensity features from model input: {len(held_out_intensity_cols)}")
            print("       first held-out examples:", held_out_intensity_cols[:20])
    if normalized_absolute_cols:
        print(f"[info] Absolute-scale features normalized within image: {len(normalized_absolute_cols)}")
        print("       first normalized examples:", normalized_absolute_cols[:20])
    print(f"[meta] Physical-unit feature count used: {len(physical_feature_cols)} / {len(feature_cols)}")
    if physical_feature_cols:
        print("[meta] Physical-unit features used:")
        print(physical_feature_cols)
    if dropped_px_cols:
        print(f"[info] Dropped pixel-space duplicates in favor of physical-unit columns: {len(dropped_px_cols)}")
        print("       first dropped examples:", dropped_px_cols[:20])

    if len(feature_cols) < MIN_COMMON_FEATURE_COUNT:
        raise RuntimeError(
            "[fail] Too few final common features after filtering.\n"
            f"Final common feature count = {len(feature_cols)}\n"
            f"Required minimum          = {MIN_COMMON_FEATURE_COUNT}"
        )

    if removed_common_cols:
        print(f"[info] Removed common non-feature/leakage columns: {len(removed_common_cols)}")
        print("       first removed examples:", removed_common_cols[:20])

    # ----------------------------
    # Build training set / 构建训练集
    # ----------------------------
    if TRAIN_WITH_ONLY_HARD_LABELS:
        train_use = train_df[train_df[TRAIN_LABEL_COL].isin([INTERNAL_CLUSTER_POSITIVE, INTERNAL_CLUSTER_NEGATIVE])].copy()
    else:
        train_use = train_df.copy()

    if len(train_use) == 0:
        raise RuntimeError("[fail] Training set is empty after label filtering.")

    y_bin = label_to_binary(train_use[TRAIN_LABEL_COL])
    valid_y = y_bin.notna()
    train_use = train_use.loc[valid_y].copy()
    y_bin = y_bin.loc[valid_y].astype(int)

    X = train_use[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
    umap_scaler = RobustScaler()
    X_imp_scaled = pd.DataFrame(
        umap_scaler.fit_transform(X_imp),
        columns=feature_cols,
        index=X.index,
    )

    print(f"[info] Final training rows used: {len(X_imp)}")
    print(f"[info] Final feature count used : {len(feature_cols)}")
    print(f"[info] Binary label counts      : {dict(pd.Series(y_bin).value_counts().sort_index())}")

    # ----------------------------
    # Split / 划分验证集
    # ----------------------------
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_imp,
            y_bin,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_bin,
        )
    except ValueError:
        print("[warn] Stratified split failed; fallback to non-stratified split.")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_imp,
            y_bin,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=None,
        )

    # ----------------------------
    # Train XGBoost / 训练 XGBoost
    # ----------------------------
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        gamma=XGB_GAMMA,
        scale_pos_weight=POSITIVE_CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        n_jobs=XGB_N_JOBS,
        tree_method="hist",
        missing=np.nan,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    print("[ok] XGBoost model trained.")

    # ----------------------------
    # Validation / 验证
    # ----------------------------
    p_valid = model.predict_proba(X_valid)[:, 1]
    pred_valid = np.array([predicted_label_from_prob(p) for p in p_valid])
    y_valid_label = np.where(y_valid.values == 1, OUTPUT_CLUSTER2_LABEL, OUTPUT_CLUSTER1_LABEL)

    metrics = {
        "deviated_threshold": DEVIATED_THRESHOLD,
        "undiff_threshold": UNDIFF_THRESHOLD,
        "valid_uncertain_fraction": float(np.mean(pred_valid == "uncertain")),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_valid, p_valid))
    except Exception:
        metrics["roc_auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(y_valid, p_valid))
    except Exception:
        metrics["average_precision"] = None

    hard_mask = pred_valid != "uncertain"
    if hard_mask.any():
        cm = confusion_matrix(
            y_valid_label[hard_mask],
            pred_valid[hard_mask],
            labels=[OUTPUT_CLUSTER1_LABEL, OUTPUT_CLUSTER2_LABEL],
        )
        cls_report = classification_report(
            y_valid_label[hard_mask],
            pred_valid[hard_mask],
            labels=[OUTPUT_CLUSTER1_LABEL, OUTPUT_CLUSTER2_LABEL],
            zero_division=0,
            output_dict=True,
        )
        metrics["confusion_matrix_excluding_uncertain"] = cm.tolist()
        metrics["classification_report_excluding_uncertain"] = cls_report
    else:
        metrics["confusion_matrix_excluding_uncertain"] = None
        metrics["classification_report_excluding_uncertain"] = None

    print("[info] Validation summary:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ----------------------------
    # Save model / 保存模型
    # ----------------------------
    model_path = OUTPUT_ROOT / MODEL_JSON_NAME
    model.save_model(str(model_path))
    print(f"[ok] Model saved: {model_path}")

    # ----------------------------
    # Save feature importance / 保存特征重要性
    # ----------------------------
    fi_df = save_xgb_gain_importance(
        model=model,
        feature_cols=feature_cols,
        out_csv=OUTPUT_ROOT / FEATURE_IMPORTANCE_CSV_NAME,
    )
    print(f"[ok] Feature importance saved: {OUTPUT_ROOT / FEATURE_IMPORTANCE_CSV_NAME}")
    print("[info] Top 20 feature importance by gain:")
    print(fi_df.head(20).to_string(index=False))

    # ----------------------------
    # Predict on inference data / 推理
    # ----------------------------
    X_infer = infer_df[feature_cols].copy()
    X_infer = X_infer.apply(pd.to_numeric, errors="coerce")
    X_infer = X_infer.replace([np.inf, -np.inf], np.nan)
    X_infer_imp = pd.DataFrame(
        imputer.transform(X_infer),
        columns=feature_cols,
        index=infer_df.index,
    )
    X_infer_scaled = pd.DataFrame(
        umap_scaler.transform(X_infer_imp),
        columns=feature_cols,
        index=infer_df.index,
    )

    X_infer_umap = infer_df[umap_feature_cols].copy()
    X_infer_umap = X_infer_umap.apply(pd.to_numeric, errors="coerce")
    X_infer_umap = X_infer_umap.replace([np.inf, -np.inf], np.nan)
    umap_imputer = SimpleImputer(strategy="median")
    X_infer_umap_imp = pd.DataFrame(
        umap_imputer.fit_transform(X_infer_umap),
        columns=umap_feature_cols,
        index=infer_df.index,
    )
    umap_scaler_global = RobustScaler()
    X_infer_umap_scaled = pd.DataFrame(
        umap_scaler_global.fit_transform(X_infer_umap_imp),
        columns=umap_feature_cols,
        index=infer_df.index,
    )

    p_dev = model.predict_proba(X_infer_imp)[:, 1]
    p_und = 1.0 - p_dev
    confidence = np.maximum(p_dev, p_und)
    pred_label = np.array([predicted_label_from_prob(p) for p in p_dev], dtype=object)

    # ----------------------------
    # Continuous deviation score / 连续偏离程度分数
    # raw probability 用于 hard label
    # deviation_score 用于 heatmap
    # ----------------------------
    if HEATMAP_REFERENCE_SOURCE == "validation_global":
        ref_prob_for_score = p_valid
    else:
        ref_prob_for_score = p_dev

    deviation_score_global, heatmap_info_global = compute_deviation_score(
        prob_values=p_dev,
        ref_prob_values=ref_prob_for_score,
    )

    pred_df = infer_df.copy()
    pred_df["cluster2_probability"] = p_dev
    pred_df["cluster1_probability"] = p_und
    pred_df["confidence"] = confidence
    pred_df["predicted_label"] = pred_label
    pred_df["deviation_score"] = deviation_score_global

    timepoint_ratio_csv = OUTPUT_ROOT / TIMEPOINT_RATIO_CSV_NAME
    timepoint_ratio_fig = OUTPUT_ROOT / TIMEPOINT_RATIO_FIG_NAME
    timepoint_ratio_df = save_timepoint_ratio_outputs(
        pred_df=pred_df,
        image_col=infer_image_col_name,
        label_col="predicted_label",
        out_csv=timepoint_ratio_csv,
        out_fig=timepoint_ratio_fig,
    )
    if len(timepoint_ratio_df) > 0:
        print(f"[ok] Timepoint ratio CSV saved: {timepoint_ratio_csv}")
        print(f"[ok] Timepoint ratio figure saved: {timepoint_ratio_fig}")

    pred_csv_path = OUTPUT_ROOT / PREDICTION_CSV_NAME
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"[ok] Prediction CSV saved: {pred_csv_path}")

    infer_umap_by_image_dir = OUTPUT_ROOT / INFER_UMAP_BY_IMAGE_DIRNAME
    infer_umap_done = False
    infer_umap_saved_images: List[str] = []
    if MAKE_INFER_UMAP and HAS_UMAP and len(X_infer_umap_scaled) >= 3:
        if INFER_UMAP_MODE == "global_fit_then_split":
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(INFER_UMAP_N_NEIGHBORS, max(2, len(X_infer_umap_scaled) - 1)),
                min_dist=INFER_UMAP_MIN_DIST,
                n_epochs=INFER_UMAP_N_EPOCHS,
                random_state=INFER_UMAP_RANDOM_STATE,
            )
            X_infer_umap_embed = reducer.fit_transform(X_infer_umap_scaled)
            pred_df["umap_1"] = X_infer_umap_embed[:, 0]
            pred_df["umap_2"] = X_infer_umap_embed[:, 1]
            shared_axis_meta = compute_square_axis_meta(pred_df, "umap_1", "umap_2")
            n_umap_images, infer_umap_saved_images = save_inference_umap_by_image(
                pred_df=pred_df,
                image_col=infer_image_col_name,
                nucleus_col=infer_nucleus_col_name,
                out_dir=infer_umap_by_image_dir,
                use_existing_embedding=True,
                shared_axis_meta=shared_axis_meta,
            )
        elif INFER_UMAP_MODE == "per_image_fit":
            n_umap_images, infer_umap_saved_images = save_inference_umap_by_image(
                pred_df=pred_df,
                feature_df_scaled=X_infer_umap_scaled,
                image_col=infer_image_col_name,
                nucleus_col=infer_nucleus_col_name,
                out_dir=infer_umap_by_image_dir,
            )
        else:
            raise ValueError(f"Unsupported INFER_UMAP_MODE: {INFER_UMAP_MODE}")

        infer_umap_done = n_umap_images > 0
        if infer_umap_done:
            print(f"[ok] Per-image inference UMAP saved for {n_umap_images} image(s): {infer_umap_by_image_dir}")
        else:
            print("[skip] Inference UMAP skipped because no image group had at least 3 nuclei.")
    elif MAKE_INFER_UMAP and not HAS_UMAP:
        print("[skip] Inference UMAP skipped because umap-learn is not installed.")
    elif MAKE_INFER_UMAP:
        print("[skip] Inference UMAP skipped because too few inference rows are available.")

    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"[ok] Prediction CSV updated: {pred_csv_path}")

    # ----------------------------
    # Save run info / 保存运行信息
    # ----------------------------
    run_info = {
        "train_feature_csv": str(TRAIN_FEATURE_CSV),
        "train_label_csv": str(TRAIN_LABEL_CSV),
        "train_label_col": TRAIN_LABEL_COL,
        "infer_feature_csv": str(INFER_FEATURE_CSV),
        "infer_image_dir": str(INFER_IMAGE_DIR),
        "infer_mask_dir": str(INFER_MASK_DIR),
        "output_root": str(OUTPUT_ROOT),
        "pipeline_variant": PIPELINE_VARIANT,
        "train_feature_image_col": train_feature_image_col,
        "train_feature_nucleus_col": train_feature_nucleus_col,
        "train_label_image_col": train_label_image_col,
        "train_label_nucleus_col": train_label_nucleus_col,
        "infer_image_col": infer_image_col_name,
        "infer_nucleus_col": infer_nucleus_col_name,
        "final_common_feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "physical_unit_feature_columns": physical_feature_cols,
        "exclude_intensity_features": EXCLUDE_INTENSITY_FEATURES,
        "absolute_feature_normalization": ABSOLUTE_FEATURE_NORMALIZATION,
        "normalized_absolute_source_columns": normalized_absolute_cols,
        "removed_common_nonfeature_columns": removed_common_cols,
        "dropped_pixel_space_duplicates": dropped_px_cols,
        "train_rows_total_after_merge": int(len(train_df)),
        "train_rows_used": int(len(train_use)),
        "train_label_distribution": {
            internal_to_output_label(str(k)): int(v) for k, v in train_df[TRAIN_LABEL_COL].value_counts().to_dict().items()
        },
        "output_label_explanation": OUTPUT_LABEL_EXPLANATION,
        "validation_metrics": metrics,
        "xgb_params": {
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "min_child_weight": XGB_MIN_CHILD_WEIGHT,
            "reg_alpha": XGB_REG_ALPHA,
            "reg_lambda": XGB_REG_LAMBDA,
            "gamma": XGB_GAMMA,
            "scale_pos_weight": POSITIVE_CLASS_WEIGHT,
            "random_seed": RANDOM_SEED,
            "tree_method": "hist",
        },
        "hard_label_thresholds": {
            "deviated_threshold": DEVIATED_THRESHOLD,
            "undiff_threshold": UNDIFF_THRESHOLD,
        },
        "output_probability_columns": {
            OUTPUT_CLUSTER2_LABEL: "cluster2_probability",
            OUTPUT_CLUSTER1_LABEL: "cluster1_probability",
        },
        "heatmap_score": {
            "mode": HEATMAP_SCORE_MODE,
            "reference_source": HEATMAP_REFERENCE_SOURCE,
            "stretch_space": HEATMAP_STRETCH_SPACE,
            "low_quantile": HEATMAP_LOW_QUANTILE,
            "high_quantile": HEATMAP_HIGH_QUANTILE,
            "gamma": HEATMAP_GAMMA,
            "per_image_stretch": PER_IMAGE_HEATMAP_STRETCH,
            "global_info": heatmap_info_global,
        },
        "inference_umap": {
            "enabled": MAKE_INFER_UMAP,
            "mode": INFER_UMAP_MODE,
            "feature_source": "original_common_features_before_per_image_normalization",
            "feature_columns": umap_feature_cols,
            "generated": infer_umap_done,
            "n_neighbors": INFER_UMAP_N_NEIGHBORS,
            "min_dist": INFER_UMAP_MIN_DIST,
            "per_image_output_dir": str(infer_umap_by_image_dir) if infer_umap_done else None,
            "saved_images": infer_umap_saved_images,
        },
        "timepoint_cluster_ratios": {
            "csv_path": str(timepoint_ratio_csv) if len(timepoint_ratio_df) > 0 else None,
            "figure_path": str(timepoint_ratio_fig) if len(timepoint_ratio_df) > 0 else None,
        },
        "feature_filters": {
            "exclude_absolute_position_features": EXCLUDE_ABSOLUTE_POSITION_FEATURES,
            "exclude_neighbor_density_features": EXCLUDE_NEIGHBOR_DENSITY_FEATURES,
        },
    }
    with open(OUTPUT_ROOT / MODEL_INFO_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(f"[ok] Run/model info saved: {OUTPUT_ROOT / MODEL_INFO_JSON_NAME}")

    # ----------------------------
    # Build image/mask index / 建立索引
    # ----------------------------
    image_index = build_file_index(
        INFER_IMAGE_DIR,
        exts=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    )
    mask_index = build_file_index(
        INFER_MASK_DIR,
        exts=[".npy"],
    )

    # ----------------------------
    # Draw overlays / 绘制 overlay
    # ----------------------------
    pred_df["__image_group_key__"] = pred_df[infer_image_col_name].map(normalize_stem_like)

    grouped: List[Tuple[str, object, pd.DataFrame]] = []
    for gkey, gdf in pred_df.groupby("__image_group_key__", dropna=False):
        if gkey == "":
            continue
        raw_name = gdf[infer_image_col_name].iloc[0]
        grouped.append((gkey, raw_name, gdf.copy()))

    print(f"[info] Unique inference image groups from CSV: {len(grouped)}")

    done_records = []
    skip_records = []

    for i, (img_key, raw_img_name, group) in enumerate(sorted(grouped, key=lambda x: x[0]), start=1):
        img_path = resolve_matching_file(raw_img_name, image_index)
        mask_path = resolve_matching_file(raw_img_name, mask_index)

        if img_path is None or mask_path is None:
            skip_records.append(
                {
                    "image_key": str(raw_img_name),
                    "resolved_image": None if img_path is None else str(img_path),
                    "resolved_mask": None if mask_path is None else str(mask_path),
                    "reason": "image_or_mask_not_found",
                }
            )
            print(f"[skip] {raw_img_name} -> image/mask not found")
            continue

        try:
            bg_rgb = read_image_as_rgb(img_path)
            mask_img = load_mask_npy(mask_path)
        except Exception as e:
            skip_records.append(
                {
                    "image_key": str(raw_img_name),
                    "resolved_image": str(img_path),
                    "resolved_mask": str(mask_path),
                    "reason": f"load_fail: {e}",
                }
            )
            print(f"[skip] {raw_img_name} -> load fail: {e}")
            continue

        if bg_rgb.shape[:2] != mask_img.shape[:2]:
            skip_records.append(
                {
                    "image_key": str(raw_img_name),
                    "resolved_image": str(img_path),
                    "resolved_mask": str(mask_path),
                    "reason": f"shape_mismatch image={bg_rgb.shape[:2]} mask={mask_img.shape[:2]}",
                }
            )
            print(f"[skip] {raw_img_name} -> shape mismatch")
            continue

        # 可选：每张图单独做 heatmap stretch
        if PER_IMAGE_HEATMAP_STRETCH:
            group_prob = group["cluster2_probability"].to_numpy(dtype=float)
            group_score, group_heatmap_info = compute_deviation_score(
                prob_values=group_prob,
                ref_prob_values=group_prob,
            )
            group = group.copy()
            group["deviation_score"] = group_score
        else:
            group_heatmap_info = heatmap_info_global

        hard_color = np.zeros_like(bg_rgb, dtype=np.uint8)
        heatmap_color = np.zeros_like(bg_rgb, dtype=np.uint8)
        valid_fill = np.zeros(mask_img.shape, dtype=bool)

        missing_label_count = 0
        painted_count = 0

        for _, row in group.iterrows():
            nucleus_label = row[infer_nucleus_col_name]
            try:
                nucleus_label = int(float(nucleus_label))
            except Exception:
                continue

            if nucleus_label <= 0:
                continue

            region = (mask_img == nucleus_label)
            if not region.any():
                missing_label_count += 1
                continue

            hard_rgb = hard_label_to_rgb(row["predicted_label"])
            heat_rgb = score_to_rgb(row["deviation_score"])

            hard_color[region] = hard_rgb
            heatmap_color[region] = heat_rgb
            valid_fill[region] = True
            painted_count += 1

        boundary = compute_boundaries(mask_img)

        if SAVE_HARD_OVERLAY:
            hard_overlay = blend_overlay(
                background_rgb=bg_rgb,
                color_layer=hard_color,
                valid_mask=valid_fill,
                alpha=HARD_OVERLAY_ALPHA,
                boundary_mask=boundary,
            )
            hard_out = hard_dir / f"{img_path.stem}_hard_overlay.png"
            try_save_rgb(hard_out, hard_overlay)

        if SAVE_HEATMAP_OVERLAY:
            heat_overlay = blend_overlay(
                background_rgb=bg_rgb,
                color_layer=heatmap_color,
                valid_mask=valid_fill,
                alpha=HEATMAP_OVERLAY_ALPHA,
                boundary_mask=boundary,
            )
            heat_out = heatmap_dir / f"{img_path.stem}_deviation_score_overlay.png"
            try_save_rgb(heat_out, heat_overlay)

        done_records.append(
            {
                "image_key": str(raw_img_name),
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "n_rows_in_csv_group": int(len(group)),
                "n_painted_nuclei": int(painted_count),
                "n_missing_mask_labels": int(missing_label_count),
                "per_image_heatmap_info": group_heatmap_info if PER_IMAGE_HEATMAP_STRETCH else None,
            }
        )

        print(
            f"[ok] {i:04d}/{len(grouped):04d} {img_path.name} "
            f"-> painted={painted_count}, missing_mask_labels={missing_label_count}"
        )

    with open(OUTPUT_ROOT / OVERLAY_LOG_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump({"done": done_records, "skipped": skip_records}, f, indent=2, ensure_ascii=False)

    if MINIMAL_OUTPUT_MODE:
        for extra_file in [
            OUTPUT_ROOT / MODEL_INFO_JSON_NAME,
            OUTPUT_ROOT / FEATURE_IMPORTANCE_CSV_NAME,
            OUTPUT_ROOT / MODEL_JSON_NAME,
            OUTPUT_ROOT / OVERLAY_LOG_JSON_NAME,
        ]:
            if extra_file.exists():
                extra_file.unlink()

    print("============================================================")
    print("[done] All finished.")
    print(f"[done] Prediction CSV      : {pred_csv_path}")
    print(f"[done] Hard overlays       : {hard_dir}")
    print(f"[done] Deviation heatmaps  : {heatmap_dir}")
    if infer_umap_done:
        print(f"[done] Per-image UMAP dir  : {infer_umap_by_image_dir}")
    print(f"[done] Skip count          : {len(skip_records)}")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
        sys.exit(1)
