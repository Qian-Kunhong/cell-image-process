from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


BASE05_PATH = Path(__file__).with_name("05_distinguish cell type.py")


def load_stage05_module():
    spec = importlib.util.spec_from_file_location("stage05_base", BASE05_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base module from: {BASE05_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage05 = load_stage05_module()


DEFAULT_INPUT_CSV = Path(stage05.TRAIN_FEATURE_CSV)
DEFAULT_LABEL_COL = stage05.TRAIN_LABEL_COL


@dataclass
class PreparedTrainingData:
    input_csv: Path
    raw_df: pd.DataFrame
    prepared_df: pd.DataFrame
    train_use: pd.DataFrame
    X_imp: pd.DataFrame
    y_bin: pd.Series
    feature_cols: list[str]
    removed_feature_cols: list[str]
    normalized_absolute_source_columns: list[str]
    image_col: Optional[str]
    nucleus_col: Optional[str]
    label_col: str
    label_counts: dict[str, int]
    imputer: SimpleImputer


@dataclass
class TrainedModelBundle:
    prepared: PreparedTrainingData
    model: Any
    metrics: dict[str, Any]


def _safe_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.to_dict().items()}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def _normalize_alias_base(feature_base: str) -> str:
    alias_map = {
        "adaptive_nb_area_mean_um2": "nb_area_mean_um2",
        "adaptive_nb_area_mean": "nb_area_mean",
        "adaptive_nb_circularity_mean": "nb_circularity_mean",
        "adaptive_nb_eccentricity_mean": "nb_eccentricity_mean",
        "adaptive_nb_aspect_ratio_mean": "nb_aspect_ratio_mean",
    }
    return alias_map.get(feature_base, feature_base)


def _canonical_semantic_feature(feature_col: str) -> tuple[str, bool, bool]:
    """
    Returns:
      - canonical key that ignores `__model` and normalizes adaptive alias names
      - whether this column has `__model`
      - whether this column is an alias name (adaptive_nb_*) for a base feature
    """
    parts = str(feature_col).split("__")
    base = parts[0]
    suffix_tokens = parts[1:]
    has_model = "model" in suffix_tokens
    normalized_base = _normalize_alias_base(base)
    is_alias_name = normalized_base != base

    suffix_wo_model = [t for t in suffix_tokens if t != "model"]
    canonical = normalized_base + "".join(f"__{t}" for t in suffix_wo_model)
    return canonical, has_model, is_alias_name


def prefer_model_feature_versions(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Semantic de-duplication of feature columns.

    Priority within each semantic group:
      1) keep `__model` variant over non-model variant
      2) keep normalized base name over alias name (e.g., `nb_*` over `adaptive_nb_*`)
      3) keep the earlier column if still tied
    """
    groups: dict[str, list[tuple[int, str, bool, bool]]] = {}
    for idx, col in enumerate(feature_cols):
        canonical, has_model, is_alias = _canonical_semantic_feature(col)
        groups.setdefault(canonical, []).append((idx, col, has_model, is_alias))

    kept_ordered: list[tuple[int, str]] = []
    removed: list[str] = []

    for canonical, entries in groups.items():
        # Lower score is preferred.
        best = min(entries, key=lambda e: (0 if e[2] else 1, 1 if e[3] else 0, e[0]))
        kept_ordered.append((best[0], best[1]))

        for entry in entries:
            if entry is best:
                continue
            _, col, has_model, _ = entry
            if best[2] and not has_model:
                removed.append(f"{col} [raw_or_alias_duplicate_of_model]")
            else:
                removed.append(f"{col} [semantic_duplicate]")

    kept_ordered.sort(key=lambda x: x[0])
    kept = [col for _, col in kept_ordered]
    return kept, removed


def _feature_preference_rank(col: str) -> tuple[int, int, int]:
    """
    Lower rank is preferred when duplicate signals are found.
    Priority:
      1) keep model-space feature (`__model`)
      2) keep image-relative feature (`__img_rel`)
      3) keep shorter name for readability
    """
    text = str(col)
    has_model = "__model" in text
    has_img_rel = "__img_rel" in text
    return (
        0 if has_model else 1,
        0 if has_img_rel else 1,
        len(text),
    )


def _allclose_with_nan(x: np.ndarray, y: np.ndarray, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    if x.shape != y.shape:
        return False
    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    if not np.array_equal(x_nan, y_nan):
        return False
    valid = ~x_nan
    if valid.sum() == 0:
        return True
    return bool(np.allclose(x[valid], y[valid], rtol=rtol, atol=atol))


def _is_log1p_equivalent(source: np.ndarray, target: np.ndarray, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    if source.shape != target.shape:
        return False
    src_nan = np.isnan(source)
    tgt_nan = np.isnan(target)
    if not np.array_equal(src_nan, tgt_nan):
        return False

    valid = ~src_nan
    if valid.sum() == 0:
        return False

    src = source[valid]
    tgt = target[valid]
    # log1p requires source >= 0 in this pipeline.
    if np.any(src < -1e-12):
        return False
    src = np.clip(src, 0.0, None)
    return bool(np.allclose(np.log1p(src), tgt, rtol=rtol, atol=atol))


def _detect_duplicate_relation(x: np.ndarray, y: np.ndarray) -> str | None:
    if _allclose_with_nan(x, y):
        return "value_duplicate"
    if _is_log1p_equivalent(x, y) or _is_log1p_equivalent(y, x):
        return "log1p_duplicate"
    return None


def drop_redundant_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    """
    Remove data-redundant features.

    Rules:
      - Drop exact / near-exact duplicate columns.
      - Drop log1p-mapped duplicates (e.g., one column is effectively log1p of another).
    """
    values: dict[str, np.ndarray] = {
        col: pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in feature_cols
    }

    kept: list[str] = []
    removed: list[str] = []

    for col in feature_cols:
        candidate = col
        candidate_removed = False

        idx = 0
        while idx < len(kept):
            kept_col = kept[idx]
            relation = _detect_duplicate_relation(values[candidate], values[kept_col])
            if relation is None:
                idx += 1
                continue

            if _feature_preference_rank(candidate) < _feature_preference_rank(kept_col):
                removed.append(f"{kept_col} [redundant_duplicate_of:{candidate}; relation:{relation}]")
                kept.pop(idx)
                continue

            removed.append(f"{candidate} [redundant_duplicate_of:{kept_col}; relation:{relation}]")
            candidate_removed = True
            break

        if not candidate_removed:
            kept.append(candidate)

    return kept, removed


def prepare_stage04_cluster_training_data(
    input_csv: Path | None = None,
    label_col: str = DEFAULT_LABEL_COL,
) -> PreparedTrainingData:
    input_csv = Path(input_csv) if input_csv is not None else DEFAULT_INPUT_CSV
    if not input_csv.exists():
        raise FileNotFoundError(f"Cluster CSV not found: {input_csv}")

    raw_df = stage05.read_csv_robust(input_csv)
    if len(raw_df) == 0:
        raise RuntimeError(f"Cluster CSV is empty: {input_csv}")
    if label_col not in raw_df.columns:
        raise KeyError(f"Label column `{label_col}` not found in: {input_csv}")

    image_col = stage05.choose_image_col(raw_df, manual=stage05.MANUAL_IMAGE_COL_TRAIN_FEATURE)
    nucleus_col = stage05.choose_nucleus_col(raw_df, manual=stage05.MANUAL_NUCLEUS_COL_TRAIN_FEATURE)
    if nucleus_col is None:
        raise RuntimeError("Could not infer nucleus id column from clustered CSV.")

    train_df = raw_df.copy()
    train_df[label_col] = train_df[label_col].map(stage05.normalize_training_label)
    allowed_labels = {
        stage05.INTERNAL_CLUSTER_POSITIVE,
        stage05.INTERNAL_CLUSTER_NEGATIVE,
        stage05.OUTPUT_UNCERTAIN_LABEL,
    }
    train_df = train_df[train_df[label_col].isin(allowed_labels)].copy()
    if len(train_df) == 0:
        raise RuntimeError("No usable rows remain after normalizing clustered labels.")

    feature_cols, removed_common_cols = stage05.select_common_feature_columns(
        train_df=train_df,
        infer_df=train_df,
        train_image_col=image_col,
        train_nucleus_col=nucleus_col,
        infer_image_col=image_col,
        infer_nucleus_col=nucleus_col,
        label_col=label_col,
    )

    if stage05.EXCLUDE_INTENSITY_FEATURES:
        feature_cols, removed_intensity_cols = stage05.filter_intensity_feature_columns(feature_cols)
        removed_common_cols = removed_common_cols + [f"{c} [intensity_held_out]" for c in removed_intensity_cols]

    feature_cols, dropped_px_cols = stage05.prefer_physical_feature_columns(feature_cols)
    removed_common_cols = removed_common_cols + dropped_px_cols

    feature_cols, unusable_feature_cols = stage05.filter_unusable_feature_columns(train_df, train_df, feature_cols)
    removed_feature_cols = removed_common_cols + unusable_feature_cols

    prepared_df, _, feature_cols, normalized_absolute_cols = stage05.apply_absolute_feature_normalization(
        train_df=train_df,
        infer_df=train_df.copy(),
        feature_cols=feature_cols,
        train_image_col=image_col,
        infer_image_col=image_col,
        mode=stage05.ABSOLUTE_FEATURE_NORMALIZATION,
    )

    feature_cols, removed_raw_dupes = prefer_model_feature_versions(list(feature_cols))
    removed_feature_cols = removed_feature_cols + removed_raw_dupes

    feature_cols, removed_redundant_cols = drop_redundant_feature_columns(prepared_df, list(feature_cols))
    removed_feature_cols = removed_feature_cols + removed_redundant_cols

    if len(feature_cols) < stage05.MIN_COMMON_FEATURE_COUNT:
        raise RuntimeError(
            "Too few usable feature columns remain after filtering. "
            f"Got {len(feature_cols)}, expected at least {stage05.MIN_COMMON_FEATURE_COUNT}."
        )

    if stage05.TRAIN_WITH_ONLY_HARD_LABELS:
        train_use = prepared_df[
            prepared_df[label_col].isin(
                [stage05.INTERNAL_CLUSTER_POSITIVE, stage05.INTERNAL_CLUSTER_NEGATIVE]
            )
        ].copy()
    else:
        train_use = prepared_df.copy()

    if len(train_use) == 0:
        raise RuntimeError("No training rows remain after hard-label filtering.")

    y_bin = stage05.label_to_binary(train_use[label_col])
    valid_y = y_bin.notna()
    train_use = train_use.loc[valid_y].copy()
    y_bin = y_bin.loc[valid_y].astype(int)

    if y_bin.nunique() < 2:
        raise RuntimeError("Cluster labels do not contain two valid classes for XGBoost analysis.")

    X = train_use[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    label_counts = _safe_int_dict(
        train_use[label_col].map(stage05.internal_to_output_label).value_counts(dropna=False)
    )

    return PreparedTrainingData(
        input_csv=input_csv,
        raw_df=raw_df,
        prepared_df=prepared_df,
        train_use=train_use,
        X_imp=X_imp,
        y_bin=y_bin,
        feature_cols=list(feature_cols),
        removed_feature_cols=list(removed_feature_cols),
        normalized_absolute_source_columns=list(normalized_absolute_cols),
        image_col=image_col,
        nucleus_col=nucleus_col,
        label_col=label_col,
        label_counts=label_counts,
        imputer=imputer,
    )


def build_stage05_xgb_classifier():
    return stage05.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=stage05.XGB_N_ESTIMATORS,
        max_depth=stage05.XGB_MAX_DEPTH,
        learning_rate=stage05.XGB_LEARNING_RATE,
        subsample=stage05.XGB_SUBSAMPLE,
        colsample_bytree=stage05.XGB_COLSAMPLE_BYTREE,
        min_child_weight=stage05.XGB_MIN_CHILD_WEIGHT,
        reg_alpha=stage05.XGB_REG_ALPHA,
        reg_lambda=stage05.XGB_REG_LAMBDA,
        gamma=stage05.XGB_GAMMA,
        scale_pos_weight=stage05.POSITIVE_CLASS_WEIGHT,
        random_state=stage05.RANDOM_SEED,
        n_jobs=stage05.XGB_N_JOBS,
        tree_method="hist",
        missing=np.nan,
    )


def compute_validation_metrics(model, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, Any]:
    if len(X_valid) == 0:
        return {}

    p_valid = model.predict_proba(X_valid)[:, 1]
    pred_valid = np.array([stage05.predicted_label_from_prob(p) for p in p_valid], dtype=object)
    y_valid_label = np.where(
        y_valid.to_numpy() == 1,
        stage05.OUTPUT_CLUSTER2_LABEL,
        stage05.OUTPUT_CLUSTER1_LABEL,
    )

    metrics: dict[str, Any] = {
        "validation_rows": int(len(X_valid)),
        "deviated_threshold": float(stage05.DEVIATED_THRESHOLD),
        "undiff_threshold": float(stage05.UNDIFF_THRESHOLD),
        "valid_uncertain_fraction": float(np.mean(pred_valid == stage05.OUTPUT_UNCERTAIN_LABEL)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_valid, p_valid))
    except Exception:
        metrics["roc_auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(y_valid, p_valid))
    except Exception:
        metrics["average_precision"] = None

    hard_mask = pred_valid != stage05.OUTPUT_UNCERTAIN_LABEL
    metrics["hard_prediction_rows"] = int(hard_mask.sum())

    if hard_mask.any():
        cm = confusion_matrix(
            y_valid_label[hard_mask],
            pred_valid[hard_mask],
            labels=[stage05.OUTPUT_CLUSTER1_LABEL, stage05.OUTPUT_CLUSTER2_LABEL],
        )
        metrics["confusion_matrix_excluding_uncertain"] = cm.tolist()
    else:
        metrics["confusion_matrix_excluding_uncertain"] = None

    return metrics


def fit_stage04_cluster_xgb(
    input_csv: Path | None = None,
    label_col: str = DEFAULT_LABEL_COL,
) -> TrainedModelBundle:
    prepared = prepare_stage04_cluster_training_data(input_csv=input_csv, label_col=label_col)

    split_kwargs = {
        "test_size": stage05.TEST_SIZE,
        "random_state": stage05.RANDOM_SEED,
    }
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            prepared.X_imp,
            prepared.y_bin,
            stratify=prepared.y_bin,
            **split_kwargs,
        )
    except ValueError:
        X_train, X_valid, y_train, y_valid = train_test_split(
            prepared.X_imp,
            prepared.y_bin,
            stratify=None,
            **split_kwargs,
        )

    model = build_stage05_xgb_classifier()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    metrics = compute_validation_metrics(model, X_valid, y_valid)

    return TrainedModelBundle(
        prepared=prepared,
        model=model,
        metrics=metrics,
    )
