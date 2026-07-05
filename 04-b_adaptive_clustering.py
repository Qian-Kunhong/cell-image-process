from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


BASE04_PATH = Path(__file__).with_name("04_unsupervised_learning.py")


def load_stage04_module():
    spec = importlib.util.spec_from_file_location("stage04_base", BASE04_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base module from: {BASE04_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage04 = load_stage04_module()


# =========================================================
# 04-b_adaptive_clustering.py
# ---------------------------------------------------------
# Auto-select cluster number with GMM model selection
# + reuse the same feature engineering / edge filtering logic as 04
# + neutral cluster naming (cluster_1, cluster_2, ...)
# + output 2D and 3D UMAP
# =========================================================


# =========================
# Paths / dataset
# =========================
SUZUI_ROOT = Path(r"F:\Suzui")
ANALYSIS_ROOT = SUZUI_ROOT / "analysis_out"
TRAINING_ROOT = SUZUI_ROOT / "training data"
TRAINING_SET_NAME = "SNL"

INPUT_CSV = ANALYSIS_ROOT / "features_training" / TRAINING_SET_NAME / "nucleus_features.csv"
INTENSITY_CSV = INPUT_CSV.parent / "nucleus_intensity_features.csv"
IMAGE_DIR = TRAINING_ROOT / TRAINING_SET_NAME
MASK_DIR = ANALYSIS_ROOT / "masks_training" / TRAINING_SET_NAME

OUT_DIR = INPUT_CSV.parent / "cluster_adaptive_auto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLUSTERED_CSV = OUT_DIR / "nucleus_features_clustered_auto.csv"
OUT_SELECTION_CSV = OUT_DIR / "cluster_model_selection.csv"
OUT_SELECTION_FIG = OUT_DIR / "cluster_model_selection.png"
OUT_CLUSTER_SUMMARY = OUT_DIR / "cluster_feature_summary.csv"
OUT_REVIEW_CSV = OUT_DIR / "cluster_review_samples.csv"
OUT_PCA_CSV = OUT_DIR / "pca_embedding.csv"
OUT_UMAP2_CSV = OUT_DIR / "umap2d_embedding.csv"
OUT_UMAP3_CSV = OUT_DIR / "umap3d_embedding.csv"
OUT_PCA_FIG = OUT_DIR / "pca_clusters.png"
OUT_UMAP2_FIG = OUT_DIR / "umap2d_clusters.png"
OUT_UMAP3_FIG = OUT_DIR / "umap3d_clusters.png"
OUT_RUN_INFO = OUT_DIR / "run_info.json"


# =========================
# Auto clustering config
# =========================
FORCE_N_CLUSTERS = None
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
MODEL_SELECTION_METRIC = "bic_penalized"
# options: "bic_penalized", "aic_penalized", "bic", "aic", "silhouette"

ENABLE_FRAGMENTATION_PENALTY = True
PREFERRED_MAX_CLUSTERS = 5
SMALL_CLUSTER_FRACTION = 0.08
TINY_CLUSTER_FRACTION = 0.05
EXCESS_CLUSTER_PENALTY = 12.0
SMALL_CLUSTER_DEFICIT_WEIGHT = 250.0
TINY_CLUSTER_DEFICIT_WEIGHT = 500.0

GMM_COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42
N_PCA_COMPONENTS_FOR_MODEL = 5

MAKE_UMAP_2D = True
MAKE_UMAP_3D = True
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10
UMAP_RANDOM_STATE = 42

N_REVIEW_PER_CLUSTER = 80
SAVE_EDGE_DEBUG_COLONY_SUPPORT = False


def configure_stage04_globals():
    stage04.INPUT_CSV = INPUT_CSV
    stage04.INTENSITY_CSV = INTENSITY_CSV
    stage04.IMAGE_DIR = IMAGE_DIR
    stage04.ORIG_MASK_DIR = MASK_DIR
    stage04.QC_KEEP_MASK_DIR = MASK_DIR
    stage04.OUT_DIR = OUT_DIR
    stage04.EDGE_DEBUG_DIR = OUT_DIR / "edge_debug"
    stage04.EDGE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stage04.SAVE_EDGE_DEBUG_COLONY_SUPPORT = SAVE_EDGE_DEBUG_COLONY_SUPPORT
    stage04.RANDOM_STATE = RANDOM_STATE
    stage04.N_PCA_COMPONENTS_FOR_CLUSTERING = N_PCA_COMPONENTS_FOR_MODEL


def choose_candidate_cluster_counts(n_fit: int) -> List[int]:
    if FORCE_N_CLUSTERS is not None:
        if FORCE_N_CLUSTERS < 2:
            raise ValueError("FORCE_N_CLUSTERS must be >= 2.")
        if FORCE_N_CLUSTERS >= n_fit:
            raise ValueError(f"FORCE_N_CLUSTERS={FORCE_N_CLUSTERS} must be smaller than fit sample count={n_fit}.")
        return [int(FORCE_N_CLUSTERS)]

    max_allowed = min(MAX_CLUSTERS, max(2, n_fit - 1))
    if max_allowed < MIN_CLUSTERS:
        raise ValueError(f"Not enough fit samples ({n_fit}) for cluster search range {MIN_CLUSTERS}..{MAX_CLUSTERS}.")
    return list(range(MIN_CLUSTERS, max_allowed + 1))


def compute_fragmentation_penalty(cluster_counts: np.ndarray) -> dict:
    counts = np.asarray(cluster_counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return {
            "cluster_size_min_frac": np.nan,
            "small_cluster_count": 0,
            "tiny_cluster_count": 0,
            "excess_cluster_penalty": 0.0,
            "small_cluster_penalty": 0.0,
            "tiny_cluster_penalty": 0.0,
            "fragmentation_penalty": 0.0,
        }

    frac = counts / total
    small_deficit = np.clip(SMALL_CLUSTER_FRACTION - frac, a_min=0.0, a_max=None)
    tiny_deficit = np.clip(TINY_CLUSTER_FRACTION - frac, a_min=0.0, a_max=None)

    excess_cluster_penalty = EXCESS_CLUSTER_PENALTY * max(0, len(counts) - PREFERRED_MAX_CLUSTERS)
    small_cluster_penalty = SMALL_CLUSTER_DEFICIT_WEIGHT * float(small_deficit.sum())
    tiny_cluster_penalty = TINY_CLUSTER_DEFICIT_WEIGHT * float(tiny_deficit.sum())
    total_penalty = excess_cluster_penalty + small_cluster_penalty + tiny_cluster_penalty

    return {
        "cluster_size_min_frac": float(frac.min()),
        "small_cluster_count": int((frac < SMALL_CLUSTER_FRACTION).sum()),
        "tiny_cluster_count": int((frac < TINY_CLUSTER_FRACTION).sum()),
        "excess_cluster_penalty": float(excess_cluster_penalty),
        "small_cluster_penalty": float(small_cluster_penalty),
        "tiny_cluster_penalty": float(tiny_cluster_penalty),
        "fragmentation_penalty": float(total_penalty),
    }


def evaluate_cluster_candidates(X_fit: np.ndarray, X_all: np.ndarray, candidate_ks: List[int]) -> tuple[pd.DataFrame, GaussianMixture, int]:
    rows = []
    fitted_models: Dict[int, GaussianMixture] = {}

    for k in candidate_ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=GMM_COVARIANCE_TYPE,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X_fit)
        fitted_models[k] = gmm

        pred_fit = gmm.predict(X_fit)
        n_unique = len(np.unique(pred_fit))

        row = {
            "n_clusters": k,
            "bic": float(gmm.bic(X_fit)),
            "aic": float(gmm.aic(X_fit)),
            "fit_log_likelihood_mean": float(gmm.score(X_fit)),
            "fit_cluster_count_observed": int(n_unique),
            "fit_cluster_size_min": int(pd.Series(pred_fit).value_counts().min()),
            "fit_cluster_size_max": int(pd.Series(pred_fit).value_counts().max()),
        }

        fit_counts = pd.Series(pred_fit).value_counts().to_numpy(dtype=int)
        penalty_info = compute_fragmentation_penalty(fit_counts) if ENABLE_FRAGMENTATION_PENALTY else {
            "cluster_size_min_frac": float(fit_counts.min() / max(fit_counts.sum(), 1)),
            "small_cluster_count": 0,
            "tiny_cluster_count": 0,
            "excess_cluster_penalty": 0.0,
            "small_cluster_penalty": 0.0,
            "tiny_cluster_penalty": 0.0,
            "fragmentation_penalty": 0.0,
        }
        row.update(penalty_info)
        row["bic_penalized"] = row["bic"] + row["fragmentation_penalty"]
        row["aic_penalized"] = row["aic"] + row["fragmentation_penalty"]

        if n_unique >= 2:
            row["silhouette"] = float(silhouette_score(X_fit, pred_fit))
            row["calinski_harabasz"] = float(calinski_harabasz_score(X_fit, pred_fit))
            row["davies_bouldin"] = float(davies_bouldin_score(X_fit, pred_fit))
        else:
            row["silhouette"] = np.nan
            row["calinski_harabasz"] = np.nan
            row["davies_bouldin"] = np.nan

        pred_all = gmm.predict(X_all)
        row["all_cluster_size_min"] = int(pd.Series(pred_all).value_counts().min())
        row["all_cluster_size_max"] = int(pd.Series(pred_all).value_counts().max())
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)

    if MODEL_SELECTION_METRIC == "bic_penalized":
        best_k = int(result_df.sort_values(["bic_penalized", "bic", "n_clusters"], ascending=[True, True, True]).iloc[0]["n_clusters"])
    elif MODEL_SELECTION_METRIC == "aic_penalized":
        best_k = int(result_df.sort_values(["aic_penalized", "aic", "n_clusters"], ascending=[True, True, True]).iloc[0]["n_clusters"])
    elif MODEL_SELECTION_METRIC == "bic":
        best_k = int(result_df.sort_values(["bic", "aic", "n_clusters"], ascending=[True, True, True]).iloc[0]["n_clusters"])
    elif MODEL_SELECTION_METRIC == "aic":
        best_k = int(result_df.sort_values(["aic", "bic", "n_clusters"], ascending=[True, True, True]).iloc[0]["n_clusters"])
    elif MODEL_SELECTION_METRIC == "silhouette":
        valid = result_df["silhouette"].notna()
        if not valid.any():
            raise RuntimeError("No valid silhouette scores available for model selection.")
        best_k = int(result_df.loc[valid].sort_values(["silhouette", "bic"], ascending=[False, True]).iloc[0]["n_clusters"])
    else:
        raise ValueError(f"Unsupported MODEL_SELECTION_METRIC: {MODEL_SELECTION_METRIC}")

    return result_df, fitted_models[best_k], best_k


def reorder_clusters_by_size(raw_clusters: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
    counts = pd.Series(raw_clusters).value_counts().sort_values(ascending=False)
    ordered_raw_ids = [int(x) for x in counts.index.tolist()]
    raw_to_rank = {raw_id: rank for rank, raw_id in enumerate(ordered_raw_ids)}

    cluster_rank = np.array([raw_to_rank[int(c)] for c in raw_clusters], dtype=int)
    cluster_label = np.array([f"cluster_{rank + 1}" for rank in cluster_rank], dtype=object)
    reordered_probs = probs[:, ordered_raw_ids]
    return cluster_rank, cluster_label, ordered_raw_ids, reordered_probs


def make_group_summary(df_out: pd.DataFrame, feature_cols: list[str], group_col: str) -> pd.DataFrame:
    rows = []
    for group_name, sub in df_out.groupby(group_col, dropna=False):
        row = {group_col: group_name, "n": int(len(sub))}
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
            parts.append(sub.copy())
        else:
            parts.append(sub.sample(n=n_each, random_state=seed))
    return pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()


def make_color_map(labels: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(labels)}


def save_selection_curve(df_sel: pd.DataFrame, out_path: Path, best_k: int, selection_metric: str):
    fig, ax1 = plt.subplots(figsize=(9, 5.6))
    x = df_sel["n_clusters"].to_numpy(dtype=int)

    left_lines = []
    left_labels = []

    if "bic" in df_sel.columns:
        line_bic, = ax1.plot(
            x,
            df_sel["bic"],
            marker="o",
            linewidth=2.2,
            color="#1f77b4",
            label="BIC (lower is better)",
        )
        left_lines.append(line_bic)
        left_labels.append(line_bic.get_label())

    if "aic" in df_sel.columns:
        line_aic, = ax1.plot(
            x,
            df_sel["aic"],
            marker="o",
            linewidth=2.2,
            color="#ff7f0e",
            label="AIC (lower is better)",
        )
        left_lines.append(line_aic)
        left_labels.append(line_aic.get_label())

    if "bic_penalized" in df_sel.columns and selection_metric == "bic_penalized":
        line_bic_pen, = ax1.plot(
            x,
            df_sel["bic_penalized"],
            marker="D",
            linewidth=2.0,
            linestyle="--",
            color="#0b4f8a",
            label="Penalized BIC (selection score)",
        )
        left_lines.append(line_bic_pen)
        left_labels.append(line_bic_pen.get_label())

    if "aic_penalized" in df_sel.columns and selection_metric == "aic_penalized":
        line_aic_pen, = ax1.plot(
            x,
            df_sel["aic_penalized"],
            marker="D",
            linewidth=2.0,
            linestyle="--",
            color="#a04d00",
            label="Penalized AIC (selection score)",
        )
        left_lines.append(line_aic_pen)
        left_labels.append(line_aic_pen.get_label())

    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Information criterion value")
    ax1.set_xticks(x.tolist())
    ax1.grid(True, axis="y", alpha=0.25)

    right_lines = []
    right_labels = []
    ax2 = None
    if "silhouette" in df_sel.columns and df_sel["silhouette"].notna().any():
        ax2 = ax1.twinx()
        line_sil, = ax2.plot(
            x,
            df_sel["silhouette"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color="purple",
            label="Silhouette score (higher is better)",
        )
        ax2.set_ylabel("Silhouette score")
        right_lines.append(line_sil)
        right_labels.append(line_sil.get_label())

    ax1.axvline(best_k, color="gray", linestyle=":", linewidth=1.8)
    ax1.annotate(
        f"Selected k = {best_k}\nby {selection_metric.upper()}",
        xy=(best_k, ax1.get_ylim()[1]),
        xytext=(8, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.9},
    )

    if "fragmentation_penalty" in df_sel.columns and ENABLE_FRAGMENTATION_PENALTY:
        best_row = df_sel.loc[df_sel["n_clusters"] == best_k].iloc[0]
        penalty_text = (
            f"Penalty at selected k: {best_row['fragmentation_penalty']:.2f}\n"
            f"small clusters: {int(best_row['small_cluster_count'])}, "
            f"tiny clusters: {int(best_row['tiny_cluster_count'])}"
        )
        ax1.text(
            0.02,
            0.03,
            penalty_text,
            transform=ax1.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "lightgray", "alpha": 0.9},
        )

    fig.legend(left_lines + right_lines, left_labels + right_labels, loc="upper right", frameon=True)
    ax1.set_title("Adaptive clustering model selection across candidate k")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_2d_scatter(df_plot: pd.DataFrame, x_col: str, y_col: str, label_col: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 8))
    labels = [str(x) for x in sorted(df_plot[label_col].dropna().astype(str).unique().tolist())]
    color_map = make_color_map(labels)
    for name, sub in df_plot.groupby(label_col, dropna=False):
        plt.scatter(
            sub[x_col],
            sub[y_col],
            s=8,
            alpha=0.65,
            color=color_map.get(str(name), "gray"),
            label=str(name),
        )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(markerscale=2, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_3d_scatter(df_plot: pd.DataFrame, xyz_cols: List[str], label_col: str, title: str, out_path: Path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    labels = [str(x) for x in sorted(df_plot[label_col].dropna().astype(str).unique().tolist())]
    color_map = make_color_map(labels)
    x_col, y_col, z_col = xyz_cols

    for name, sub in df_plot.groupby(label_col, dropna=False):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            sub[z_col],
            s=8,
            alpha=0.55,
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
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_umap_embeddings(X_weighted_all: np.ndarray) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    umap2_df = None
    umap3_df = None

    if not stage04.HAS_UMAP:
        warnings.warn("umap-learn is not installed. 2D/3D UMAP outputs will be skipped.")
        return None, None

    if MAKE_UMAP_2D:
        reducer2 = stage04.umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )
        emb2 = reducer2.fit_transform(X_weighted_all)
        umap2_df = pd.DataFrame({"umap2_1": emb2[:, 0], "umap2_2": emb2[:, 1]})

    if MAKE_UMAP_3D:
        reducer3 = stage04.umap.UMAP(
            n_components=3,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )
        emb3 = reducer3.fit_transform(X_weighted_all)
        umap3_df = pd.DataFrame({"umap3_1": emb3[:, 0], "umap3_2": emb3[:, 1], "umap3_3": emb3[:, 2]})

    return umap2_df, umap3_df


def main():
    configure_stage04_globals()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df = stage04.add_feature_aliases(df)
    if len(df) == 0:
        raise RuntimeError("Input CSV is empty.")

    features, log1p_cols, feature_weights = stage04.resolve_feature_specs(df, stage04.FEATURE_SPECS, "INPUT_CSV")
    print("[info] Resolved adaptive-clustering features:")
    print(features)

    keep_meta_cols = [c for c in stage04.PREFERRED_META_COLS if c in df.columns]
    df_meta = df[keep_meta_cols].copy()

    edge_df = stage04.compute_edge_metrics_for_all_rows(df_meta)
    fit_mask = stage04.make_inner_fit_mask(edge_df)
    print(f"[info] Inner-fit nuclei used for model selection: {int(fit_mask.sum())} / {len(fit_mask)}")

    df_work = df.copy()
    transformed_feature_cols = []
    for col in features:
        new_col = f"{col}__model"
        if col in log1p_cols:
            df_work[new_col] = stage04.safe_log1p(df_work[col])
        else:
            df_work[new_col] = pd.to_numeric(df_work[col], errors="coerce")
        transformed_feature_cols.append(new_col)

    X_df = df_work[transformed_feature_cols].copy()
    _, _, pca, X_weighted_all, X_pca_all, _ = stage04.fit_transform_pipeline_on_inner(
        X_df,
        fit_mask,
        feature_weights,
    )

    X_fit_pca = X_pca_all[fit_mask]
    candidate_ks = choose_candidate_cluster_counts(len(X_fit_pca))
    print(f"[info] Candidate cluster counts: {candidate_ks}")

    selection_df, best_model, best_k = evaluate_cluster_candidates(X_fit_pca, X_pca_all, candidate_ks)
    selection_df.to_csv(OUT_SELECTION_CSV, index=False, encoding="utf-8-sig")
    save_selection_curve(selection_df, OUT_SELECTION_FIG, best_k=best_k, selection_metric=MODEL_SELECTION_METRIC)
    print(f"[saved] {OUT_SELECTION_CSV}")
    print(f"[saved] {OUT_SELECTION_FIG}")
    print(f"[info] Selected n_clusters = {best_k} by {MODEL_SELECTION_METRIC}")

    gmm_cluster_raw = best_model.predict(X_pca_all)
    gmm_prob_raw = best_model.predict_proba(X_pca_all)
    gmm_max_prob = gmm_prob_raw.max(axis=1)

    cluster_rank, cluster_label, ordered_raw_ids, gmm_prob = reorder_clusters_by_size(gmm_cluster_raw, gmm_prob_raw)

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    kmeans.fit(X_pca_all[fit_mask])
    kmeans_raw = kmeans.predict(X_pca_all)
    kmeans_rank = pd.Series(kmeans_raw).rank(method="dense").astype(int).to_numpy() - 1
    kmeans_label = np.array([f"cluster_{int(x) + 1}" for x in kmeans_rank], dtype=object)

    umap2_df, umap3_df = compute_umap_embeddings(X_weighted_all)

    df_out = df.copy()
    for col in transformed_feature_cols:
        df_out[col] = df_work[col]

    df_out["pca_1"] = X_pca_all[:, 0]
    df_out["pca_2"] = X_pca_all[:, 1]
    if X_pca_all.shape[1] >= 3:
        df_out["pca_3"] = X_pca_all[:, 2]

    if umap2_df is not None:
        df_out["umap2_1"] = umap2_df["umap2_1"]
        df_out["umap2_2"] = umap2_df["umap2_2"]
        umap2_df = pd.concat([df_out[keep_meta_cols], umap2_df, pd.Series(cluster_label, name="cluster_label")], axis=1)
        umap2_df.to_csv(OUT_UMAP2_CSV, index=False, encoding="utf-8-sig")
        save_2d_scatter(df_out.assign(cluster_label=cluster_label), "umap2_1", "umap2_2", "cluster_label", "Adaptive clustering UMAP 2D", OUT_UMAP2_FIG)
        print(f"[saved] {OUT_UMAP2_CSV}")
        print(f"[saved] {OUT_UMAP2_FIG}")

    if umap3_df is not None:
        df_out["umap3_1"] = umap3_df["umap3_1"]
        df_out["umap3_2"] = umap3_df["umap3_2"]
        df_out["umap3_3"] = umap3_df["umap3_3"]
        umap3_df = pd.concat([df_out[keep_meta_cols], umap3_df, pd.Series(cluster_label, name="cluster_label")], axis=1)
        umap3_df.to_csv(OUT_UMAP3_CSV, index=False, encoding="utf-8-sig")
        save_3d_scatter(df_out.assign(cluster_label=cluster_label), ["umap3_1", "umap3_2", "umap3_3"], "cluster_label", "Adaptive clustering UMAP 3D", OUT_UMAP3_FIG)
        print(f"[saved] {OUT_UMAP3_CSV}")
        print(f"[saved] {OUT_UMAP3_FIG}")

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
    df_out["adaptive_gmm_cluster_raw"] = gmm_cluster_raw
    df_out["adaptive_cluster_rank"] = cluster_rank
    df_out["adaptive_cluster_label"] = cluster_label
    df_out["adaptive_cluster_confidence"] = gmm_max_prob
    df_out["adaptive_cluster_count_selected"] = int(best_k)
    for idx in range(best_k):
        df_out[f"adaptive_prob_cluster_{idx + 1}"] = gmm_prob[:, idx]

    df_out["adaptive_kmeans_cluster_raw"] = kmeans_raw
    df_out["adaptive_kmeans_label"] = kmeans_label

    df_out.to_csv(OUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_CLUSTERED_CSV}")

    pca_cols = keep_meta_cols + ["pca_1", "pca_2", "adaptive_cluster_label", "adaptive_cluster_confidence", "fit_used_for_model", "is_edge_band", "is_inner_fit"]
    if "pca_3" in df_out.columns:
        pca_cols.append("pca_3")
    df_out[pca_cols].to_csv(OUT_PCA_CSV, index=False, encoding="utf-8-sig")
    save_2d_scatter(df_out, "pca_1", "pca_2", "adaptive_cluster_label", "Adaptive clustering PCA 2D", OUT_PCA_FIG)
    print(f"[saved] {OUT_PCA_CSV}")
    print(f"[saved] {OUT_PCA_FIG}")

    summary_df = make_group_summary(df_out, features, "adaptive_cluster_label")
    summary_df.to_csv(OUT_CLUSTER_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_CLUSTER_SUMMARY}")

    review_cols = keep_meta_cols + features + [
        "adaptive_cluster_label",
        "adaptive_cluster_confidence",
        "fit_used_for_model",
        "is_edge_band",
        "is_inner_fit",
        "boundary_distance_um",
        "pixel_size_mean_um",
        "pca_1",
        "pca_2",
    ]
    if "pca_3" in df_out.columns:
        review_cols.append("pca_3")
    if umap2_df is not None:
        review_cols += ["umap2_1", "umap2_2"]
    if umap3_df is not None:
        review_cols += ["umap3_1", "umap3_2", "umap3_3"]
    review_df = sample_for_review(df_out[review_cols], "adaptive_cluster_label", N_REVIEW_PER_CLUSTER, seed=RANDOM_STATE)
    review_df.to_csv(OUT_REVIEW_CSV, index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_REVIEW_CSV}")

    run_info = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUT_DIR),
        "training_set_name": TRAINING_SET_NAME,
        "feature_columns": features,
        "log1p_columns": sorted(log1p_cols),
        "feature_weights": feature_weights,
        "selection_metric": MODEL_SELECTION_METRIC,
        "fragmentation_penalty": {
            "enabled": ENABLE_FRAGMENTATION_PENALTY,
            "preferred_max_clusters": PREFERRED_MAX_CLUSTERS,
            "small_cluster_fraction": SMALL_CLUSTER_FRACTION,
            "tiny_cluster_fraction": TINY_CLUSTER_FRACTION,
            "excess_cluster_penalty": EXCESS_CLUSTER_PENALTY,
            "small_cluster_deficit_weight": SMALL_CLUSTER_DEFICIT_WEIGHT,
            "tiny_cluster_deficit_weight": TINY_CLUSTER_DEFICIT_WEIGHT,
        },
        "candidate_cluster_counts": candidate_ks,
        "selected_n_clusters": int(best_k),
        "gmm_covariance_type": GMM_COVARIANCE_TYPE,
        "random_state": RANDOM_STATE,
        "inner_fit_count": int(fit_mask.sum()),
        "total_count": int(len(df_out)),
        "cluster_size_counts": df_out["adaptive_cluster_label"].value_counts(dropna=False).to_dict(),
        "ordered_raw_cluster_ids_by_size": ordered_raw_ids,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "outputs": {
            "clustered_csv": str(OUT_CLUSTERED_CSV),
            "selection_csv": str(OUT_SELECTION_CSV),
            "selection_figure": str(OUT_SELECTION_FIG),
            "cluster_summary_csv": str(OUT_CLUSTER_SUMMARY),
            "review_csv": str(OUT_REVIEW_CSV),
            "pca_csv": str(OUT_PCA_CSV),
            "pca_figure": str(OUT_PCA_FIG),
            "umap2_csv": str(OUT_UMAP2_CSV) if umap2_df is not None else None,
            "umap2_figure": str(OUT_UMAP2_FIG) if umap2_df is not None else None,
            "umap3_csv": str(OUT_UMAP3_CSV) if umap3_df is not None else None,
            "umap3_figure": str(OUT_UMAP3_FIG) if umap3_df is not None else None,
        },
    }
    with open(OUT_RUN_INFO, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_RUN_INFO}")

    print("\nAdaptive cluster counts:")
    print(df_out["adaptive_cluster_label"].value_counts(dropna=False).to_string())
    print(f"\nSelected cluster count = {best_k}")
    print(f"Model selection metric = {MODEL_SELECTION_METRIC}")


if __name__ == "__main__":
    main()
