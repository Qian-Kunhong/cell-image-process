from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

from double_work_utils import (
    build_pair_records,
    load_module_from_path,
    save_score_intensity_relationship,
    save_preview_image,
    sibling_single_work_dir,
)


# =========================================================
# 04_cluster_dapi_oct4.py
# ---------------------------------------------------------
# Cluster with DAPI-derived morphology/neighborhood features,
# then compare the resulting DAPI deviation score against
# Oct-4 mean intensity.
# =========================================================

SUZUI_ROOT = Path(globals().get("SUZUI_ROOT", r"F:\Suzui"))
ANALYSIS_ROOT = Path(globals().get("ANALYSIS_ROOT", SUZUI_ROOT / "analysis_out"))
TRAINING_ROOT = Path(globals().get("TRAINING_ROOT", SUZUI_ROOT / "training data"))
TRAINING_SET_NAME = globals().get("TRAINING_SET_NAME", "SNL")

DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", TRAINING_ROOT / TRAINING_SET_NAME))
OCT4_IMAGE_DIR = Path(globals().get("OCT4_IMAGE_DIR", DAPI_IMAGE_DIR))
INPUT_CSV = Path(
    globals().get(
        "INPUT_CSV",
        ANALYSIS_ROOT / "features_training_double" / TRAINING_SET_NAME / "nucleus_features.csv",
    )
)
INTENSITY_CSV = Path(globals().get("INTENSITY_CSV", INPUT_CSV.parent / "nucleus_intensity_features.csv"))
MASK_DIR = Path(
    globals().get(
        "MASK_DIR",
        ANALYSIS_ROOT / "masks_training_double" / TRAINING_SET_NAME,
    )
)
QC_KEEP_MASK_DIR = Path(globals().get("QC_KEEP_MASK_DIR", INPUT_CSV.parent / "qc_training" / "qc_keep_masks"))
TMP_OUT_DIR_NAME = globals().get("TMP_OUT_DIR_NAME", "_cluster_tmp_double")

ORIENTATION_POSITIVE_COLS = {
    "nn1_distance_um": 1.0,
    "knn6_distance_mean_um": 1.0,
    "knn6_distance_std_um": 0.7,
    "adaptive_nb_area_mean_um2": 0.8,
    "adaptive_nb_eccentricity_mean": 0.8,
    "adaptive_nb_aspect_ratio_mean": 0.8,
}
ORIENTATION_NEGATIVE_COLS = {
    "local_density_per_um2": 1.0,
    "adaptive_nb_circularity_mean": 0.7,
    "fixed_neighbor_count": 1.0,
}
FINAL_LABEL_COLORS = {
    "cluster1": "#3cb65a",
    "cluster2": "#d63a3a",
    "uncertain": "#e7b93a",
}


def configure_base04(base_module) -> None:
    base_module.SUZUI_ROOT = SUZUI_ROOT
    base_module.ANALYSIS_ROOT = ANALYSIS_ROOT
    base_module.TRAINING_ROOT = TRAINING_ROOT
    base_module.TRAINING_SET_NAME = TRAINING_SET_NAME

    base_module.INPUT_CSV = INPUT_CSV
    base_module.INTENSITY_CSV = INTENSITY_CSV
    base_module.IMAGE_DIR = DAPI_IMAGE_DIR
    base_module.ORIG_MASK_DIR = MASK_DIR
    base_module.QC_KEEP_MASK_DIR = QC_KEEP_MASK_DIR

    out_dir = INPUT_CSV.parent / TMP_OUT_DIR_NAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_module.OUT_DIR = out_dir
    base_module.MINIMAL_OUTPUT_MODE = True
    base_module.KEEP_EXTENDED_OUTPUT_DIR = True
    base_module.FINAL_CLUSTERED_CSV = INPUT_CSV
    base_module.FINAL_CLUSTER_FIG = INPUT_CSV.parent / "umap2d_clusters.png"

    base_module.OUT_CLUSTERED_CSV = out_dir / "nucleus_features_qc_clustered.csv"
    base_module.OUT_FEATURE_SUMMARY_RAW = out_dir / "cluster_feature_summary_raw_gmm.csv"
    base_module.OUT_FEATURE_SUMMARY = out_dir / "cluster_feature_summary.csv"
    base_module.OUT_REVIEW_SAMPLES = out_dir / "cluster_review_samples.csv"
    base_module.OUT_PCA_EMBEDDING = out_dir / "pca_embedding.csv"
    base_module.OUT_UMAP_EMBEDDING = out_dir / "umap_embedding.csv"
    base_module.OUT_UMAP2_EMBEDDING = out_dir / "umap2d_embedding.csv"
    base_module.OUT_UMAP3_EMBEDDING = out_dir / "umap3d_embedding.csv"
    base_module.OUT_FIG_PCA_RAW = out_dir / "pca_clusters_raw_gmm.png"
    base_module.OUT_FIG_PCA = out_dir / "pca_clusters.png"
    base_module.OUT_FIG_UMAP_RAW = out_dir / "umap_clusters_raw_gmm.png"
    base_module.OUT_FIG_UMAP = out_dir / "umap_clusters.png"
    base_module.OUT_FIG_UMAP2_RAW = out_dir / "umap2d_clusters_raw_gmm.png"
    base_module.OUT_FIG_UMAP2 = INPUT_CSV.parent / "umap2d_clusters.png"
    base_module.OUT_FIG_UMAP3_RAW = out_dir / "umap3d_clusters_raw_gmm.png"
    base_module.OUT_FIG_UMAP3 = INPUT_CSV.parent / "umap3d_clusters.png"
    base_module.OUT_RUN_INFO = out_dir / "run_info.txt"
    base_module.OUT_RUN_INFO_JSON = out_dir / "run_info.json"
    base_module.OUT_FEATURE_INFO = out_dir / "selected_features.txt"
    base_module.OUT_DEVIATION_INTENSITY_CSV = out_dir / "deviation_score_vs_oct4_mean_intensity.csv"
    base_module.OUT_DEVIATION_INTENSITY_FIG = INPUT_CSV.parent / "deviation_score_vs_oct4_mean_intensity.png"
    base_module.REVIEW_OVERLAY_DIR = out_dir / "review_overlay_tif"
    base_module.REVIEW_CLUSTER_MASK_DIR = out_dir / "review_cluster_masks"
    base_module.REVIEW_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    base_module.REVIEW_CLUSTER_MASK_DIR.mkdir(parents=True, exist_ok=True)


def infer_old_deviated_cluster(df: pd.DataFrame) -> int:
    cluster_ids = sorted(pd.to_numeric(df["gmm_cluster_raw"], errors="coerce").dropna().astype(int).unique().tolist())
    if not cluster_ids:
        raise RuntimeError("No gmm_cluster_raw values found.")
    diff_map = {}
    for cluster_id in cluster_ids:
        prob_col = f"gmm_prob_cluster{cluster_id}"
        if prob_col not in df.columns:
            continue
        diff_map[cluster_id] = float(
            np.nanmean(np.abs(df["gmm_prob_deviated_raw"].to_numpy(dtype=float) - df[prob_col].to_numpy(dtype=float)))
        )
    if not diff_map:
        raise RuntimeError("Could not infer old deviated cluster from probability columns.")
    return min(diff_map, key=diff_map.get)


def resolve_cluster_orientation(df: pd.DataFrame) -> dict:
    cluster_ids = sorted(pd.to_numeric(df["gmm_cluster_raw"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(cluster_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 GMM clusters, got {cluster_ids}")

    mean_df = df.groupby("gmm_cluster_raw").mean(numeric_only=True)
    scores = {cluster_id: 0.0 for cluster_id in cluster_ids}
    feature_details = []

    for col, weight in ORIENTATION_POSITIVE_COLS.items():
        if col not in mean_df.columns:
            continue
        vals = mean_df.loc[cluster_ids, col].astype(float)
        vmin = float(vals.min())
        vmax = float(vals.max())
        span = max(vmax - vmin, 1e-12)
        per_cluster = {}
        for cluster_id in cluster_ids:
            rel = (float(vals.loc[cluster_id]) - vmin) / span
            scores[cluster_id] += weight * rel
            per_cluster[str(cluster_id)] = float(vals.loc[cluster_id])
        feature_details.append({"feature": col, "direction": "higher_is_more_deviated", "weight": weight, "means": per_cluster})

    for col, weight in ORIENTATION_NEGATIVE_COLS.items():
        if col not in mean_df.columns:
            continue
        vals = mean_df.loc[cluster_ids, col].astype(float)
        vmin = float(vals.min())
        vmax = float(vals.max())
        span = max(vmax - vmin, 1e-12)
        per_cluster = {}
        for cluster_id in cluster_ids:
            rel = (vmax - float(vals.loc[cluster_id])) / span
            scores[cluster_id] += weight * rel
            per_cluster[str(cluster_id)] = float(vals.loc[cluster_id])
        feature_details.append({"feature": col, "direction": "lower_is_more_deviated", "weight": weight, "means": per_cluster})

    new_dev_cluster = max(scores, key=scores.get)
    new_undiff_cluster = min(scores, key=scores.get)
    old_dev_cluster = infer_old_deviated_cluster(df)
    old_undiff_cluster = [c for c in cluster_ids if c != old_dev_cluster][0]

    return {
        "old_deviated_cluster_raw": int(old_dev_cluster),
        "old_undifferentiated_cluster_raw": int(old_undiff_cluster),
        "new_deviated_cluster_raw": int(new_dev_cluster),
        "new_undifferentiated_cluster_raw": int(new_undiff_cluster),
        "cluster_orientation_scores": {str(k): float(v) for k, v in scores.items()},
        "orientation_feature_details": feature_details,
        "was_swapped": bool(new_dev_cluster != old_dev_cluster),
    }


def relabel_by_orientation(df: pd.DataFrame, orientation_info: dict, base04) -> pd.DataFrame:
    out = df.copy()
    dev_cluster = int(orientation_info["new_deviated_cluster_raw"])
    undiff_cluster = int(orientation_info["new_undifferentiated_cluster_raw"])

    dev_col = f"gmm_prob_cluster{dev_cluster}"
    undiff_col = f"gmm_prob_cluster{undiff_cluster}"
    out["gmm_prob_deviated_raw"] = pd.to_numeric(out[dev_col], errors="coerce")
    out["gmm_prob_undifferentiated_raw"] = pd.to_numeric(out[undiff_col], errors="coerce")
    out["gmm_prob_margin_dev_minus_undiff_raw"] = out["gmm_prob_deviated_raw"] - out["gmm_prob_undifferentiated_raw"]
    out["deviated_score"] = out["gmm_prob_deviated_raw"]

    labels = []
    for p_dev, p_undiff in zip(out["gmm_prob_deviated_raw"], out["gmm_prob_undifferentiated_raw"]):
        if float(p_dev) >= float(base04.FULL_DEV_MIN):
            labels.append("cluster2")
        elif float(p_undiff) >= float(base04.FULL_UNDIFF_MIN):
            labels.append("cluster1")
        else:
            labels.append("uncertain")
    out["final_state_label"] = labels
    out["resolved_deviated_cluster_raw"] = dev_cluster
    out["resolved_undifferentiated_cluster_raw"] = undiff_cluster
    return out


def refresh_final_figures(base04, df_out: pd.DataFrame) -> None:
    save_scatter_figure_clustered(
        df_out,
        "pca_1",
        "pca_2",
        "final_state_label",
        f"PCA scatter final labels ({base04.LABELING_MODE})",
        base04.OUT_FIG_PCA,
    )
    if {"umap2_1", "umap2_2"}.issubset(df_out.columns):
        save_scatter_figure_clustered(
            df_out,
            "umap2_1",
            "umap2_2",
            "final_state_label",
            f"UMAP 2D scatter final labels ({base04.LABELING_MODE})",
            base04.OUT_FIG_UMAP2,
        )
        if base04.OUT_FIG_UMAP2.resolve() != base04.FINAL_CLUSTER_FIG.resolve():
            shutil.copy2(base04.OUT_FIG_UMAP2, base04.FINAL_CLUSTER_FIG)
    if {"umap3_1", "umap3_2", "umap3_3"}.issubset(df_out.columns):
        save_scatter_figure_3d_clustered(
            df_out,
            ["umap3_1", "umap3_2", "umap3_3"],
            "final_state_label",
            f"UMAP 3D scatter final labels ({base04.LABELING_MODE})",
            base04.OUT_FIG_UMAP3,
        )


def save_scatter_figure_clustered(
    df_embed: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 8))
    for name, sub in df_embed.groupby(group_col, dropna=False):
        label = str(name)
        plt.scatter(
            sub[x_col],
            sub[y_col],
            s=8,
            alpha=0.75,
            label=label,
            color=FINAL_LABEL_COLORS.get(label, "#808080"),
        )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(markerscale=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_scatter_figure_3d_clustered(
    df_embed: pd.DataFrame,
    xyz_cols: list[str],
    group_col: str,
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    x_col, y_col, z_col = xyz_cols
    for name, sub in df_embed.groupby(group_col, dropna=False):
        label = str(name)
        ax.scatter(
            sub[x_col],
            sub[y_col],
            sub[z_col],
            s=8,
            alpha=0.75,
            label=label,
            color=FINAL_LABEL_COLORS.get(label, "#808080"),
            depthshade=False,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.legend(markerscale=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def refresh_review_samples(base04, df_out: pd.DataFrame) -> pd.DataFrame:
    if base04.OUT_REVIEW_SAMPLES.exists():
        existing_cols = pd.read_csv(base04.OUT_REVIEW_SAMPLES, nrows=1).columns.tolist()
        review_cols = [c for c in existing_cols if c in df_out.columns]
    else:
        review_cols = [c for c in df_out.columns if c not in {"resolved_deviated_cluster_raw", "resolved_undifferentiated_cluster_raw"}]

    review_source = df_out.loc[:, review_cols].copy()
    review_df = base04.sample_for_review(
        review_source,
        "final_state_label",
        base04.N_REVIEW_PER_CLUSTER,
        seed=base04.RANDOM_STATE,
    )
    review_df.to_csv(base04.OUT_REVIEW_SAMPLES, index=False, encoding="utf-8-sig")
    return review_df


def refresh_sampled_review_assets(base04, review_df: pd.DataFrame) -> None:
    if base04.REVIEW_OVERLAY_DIR.exists():
        for p in base04.REVIEW_OVERLAY_DIR.iterdir():
            if p.is_file():
                p.unlink()
    if base04.REVIEW_CLUSTER_MASK_DIR.exists():
        for p in base04.REVIEW_CLUSTER_MASK_DIR.iterdir():
            if p.is_file():
                p.unlink()

    for image_name, sub in review_df.groupby("image_name"):
        orig_mask_path = base04.find_matching_orig_mask(image_name)
        qc_keep_mask_path = base04.find_matching_qc_keep_mask(image_name)
        img_path = base04.find_matching_image(image_name)
        if orig_mask_path is None or qc_keep_mask_path is None or img_path is None:
            continue

        orig_mask = np.load(orig_mask_path)
        qc_keep_mask = np.load(qc_keep_mask_path)
        keep_binary = qc_keep_mask > 0
        img_gray = base04.ensure_2d_image(tiff.imread(img_path))

        dev_labels = set(pd.to_numeric(sub.loc[sub["final_state_label"] == "deviated", "label"], errors="coerce").dropna().astype(int).tolist())
        undiff_labels = set(pd.to_numeric(sub.loc[sub["final_state_label"] == "undifferentiated", "label"], errors="coerce").dropna().astype(int).tolist())
        uncertain_labels = set(pd.to_numeric(sub.loc[sub["final_state_label"] == "uncertain", "label"], errors="coerce").dropna().astype(int).tolist())

        deviated_mask = np.where(np.isin(orig_mask, list(dev_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
        undiff_mask = np.where(np.isin(orig_mask, list(undiff_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
        uncertain_mask = np.where(np.isin(orig_mask, list(uncertain_labels)) & keep_binary, orig_mask, 0).astype(np.int32)

        overlay = base04.make_label_overlay(
            img_gray=img_gray,
            deviated_mask=deviated_mask,
            undiff_mask=undiff_mask,
            uncertain_mask=uncertain_mask,
        )
        tiff.imwrite(str(base04.REVIEW_OVERLAY_DIR / f"{image_name}_final_label_label_overlay.tif"), overlay)

        np.save(base04.REVIEW_CLUSTER_MASK_DIR / f"{image_name}_final_label_deviated_mask.npy", base04.relabel_compact(deviated_mask))
        np.save(base04.REVIEW_CLUSTER_MASK_DIR / f"{image_name}_final_label_undifferentiated_mask.npy", base04.relabel_compact(undiff_mask))
        np.save(base04.REVIEW_CLUSTER_MASK_DIR / f"{image_name}_final_label_uncertain_mask.npy", base04.relabel_compact(uncertain_mask))


def refresh_summary_and_run_info(base04, df_out: pd.DataFrame, orientation_info: dict) -> None:
    run_info = {}
    if base04.OUT_RUN_INFO_JSON.exists():
        with open(base04.OUT_RUN_INFO_JSON, "r", encoding="utf-8") as f:
            run_info = json.load(f)
    feature_cols = run_info.get("feature_columns", [])
    if feature_cols:
        summary_df = base04.make_group_summary(df_out, feature_cols, "final_state_label")
        summary_df.to_csv(base04.OUT_FEATURE_SUMMARY, index=False, encoding="utf-8-sig")

    run_info["cluster_orientation_resolution"] = orientation_info
    with open(base04.OUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)


def make_state_masks(df_sub: pd.DataFrame, orig_mask: np.ndarray, keep_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cluster2_names = {"deviated", "cluster2"}
    cluster1_names = {"undifferentiated", "cluster1"}
    uncertain_names = {"uncertain"}

    dev_labels = set(
        pd.to_numeric(
            df_sub.loc[df_sub["final_state_label"].isin(cluster2_names), "label"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )
    undiff_labels = set(
        pd.to_numeric(
            df_sub.loc[df_sub["final_state_label"].isin(cluster1_names), "label"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )
    uncertain_labels = set(
        pd.to_numeric(
            df_sub.loc[df_sub["final_state_label"].isin(uncertain_names), "label"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )

    deviated_mask = np.where(np.isin(orig_mask, list(dev_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
    undiff_mask = np.where(np.isin(orig_mask, list(undiff_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
    uncertain_mask = np.where(np.isin(orig_mask, list(uncertain_labels)) & keep_binary, orig_mask, 0).astype(np.int32)
    return deviated_mask, undiff_mask, uncertain_mask


def save_full_overlays(base04, df_out: pd.DataFrame) -> list[Path]:
    pair_records, _ = build_pair_records(
        dapi_dir=DAPI_IMAGE_DIR,
        oct4_dir=OCT4_IMAGE_DIR,
        mask_dir=MASK_DIR,
        out_report_csv=None,
    )
    pair_map = {record.image_name: record for record in pair_records}
    saved_paths: list[Path] = []

    for image_name, sub in df_out.groupby("image_name"):
        record = pair_map.get(image_name)
        if record is None:
            continue

        orig_mask_path = record.mask_path
        qc_keep_mask_path = QC_KEEP_MASK_DIR / f"{image_name}_qc_keep_mask.npy"
        use_keep_mask = qc_keep_mask_path if qc_keep_mask_path.exists() else orig_mask_path
        if not orig_mask_path.exists() or not use_keep_mask.exists():
            continue

        orig_mask = np.load(orig_mask_path)
        qc_keep_mask = np.load(use_keep_mask)
        keep_binary = qc_keep_mask > 0

        deviated_mask, undiff_mask, uncertain_mask = make_state_masks(sub, orig_mask, keep_binary)

        dapi_img = base04.ensure_2d_image(tiff.imread(record.dapi_path))
        dapi_overlay = base04.make_label_overlay(
            img_gray=dapi_img,
            deviated_mask=deviated_mask,
            undiff_mask=undiff_mask,
            uncertain_mask=uncertain_mask,
        )
        dapi_out = INPUT_CSV.parent / f"{image_name}_overlay_on_dapi.png"
        save_preview_image(dapi_overlay, out_png=dapi_out)
        saved_paths.append(dapi_out)

        oct4_img = base04.ensure_2d_image(tiff.imread(record.oct4_path))
        oct4_overlay = base04.make_label_overlay(
            img_gray=oct4_img,
            deviated_mask=deviated_mask,
            undiff_mask=undiff_mask,
            uncertain_mask=uncertain_mask,
        )
        oct4_out = INPUT_CSV.parent / f"{image_name}_overlay_on_oct4.png"
        save_preview_image(oct4_overlay, out_png=oct4_out)
        saved_paths.append(oct4_out)

    return saved_paths


def cleanup_extra_outputs(base04) -> None:
    legacy_dir = INPUT_CSV.parent / f"cluster_neighbor_innerfit_{base04.LABELING_MODE}_double"
    cleanup_paths = [
        INPUT_CSV.parent / "cluster_umap.png",
        INPUT_CSV.parent / "deviation_score_vs_oct4_mean_intensity.csv",
        INPUT_CSV.parent / "deviation_score_vs_oct4_mean_intensity_summary.json",
        INPUT_CSV.parent / "deviation_score_vs_mean_intensity.csv",
        INPUT_CSV.parent / "deviation_score_vs_mean_intensity.png",
    ]
    for path in cleanup_paths:
        if path.exists():
            path.unlink()

    for path in [base04.OUT_DIR, legacy_dir]:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def build_score_intensity_df(scored_df: pd.DataFrame) -> pd.DataFrame:
    intensity_df = pd.read_csv(INTENSITY_CSV)
    rel_df = scored_df.loc[:, ["image_name", "label", "final_state_label", "deviated_score"]].merge(
        intensity_df.loc[:, ["image_name", "label", "mean_intensity"]],
        on=["image_name", "label"],
        how="inner",
    )
    rel_df["deviated_score"] = pd.to_numeric(rel_df["deviated_score"], errors="coerce")
    rel_df["mean_intensity"] = pd.to_numeric(rel_df["mean_intensity"], errors="coerce")
    rel_df = rel_df.loc[rel_df["deviated_score"].notna() & rel_df["mean_intensity"].notna()].copy()
    return rel_df


def save_oct4_intensity_by_cluster_figure(rel_df: pd.DataFrame, out_path: Path) -> None:
    order = [label for label in ["cluster1", "uncertain", "cluster2"] if label in rel_df["final_state_label"].unique()]
    groups = [rel_df.loc[rel_df["final_state_label"] == label, "mean_intensity"].to_numpy(dtype=float) for label in order]
    positions = np.arange(1, len(order) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.6},
        whiskerprops={"color": "#404040", "linewidth": 1.1},
        capprops={"color": "#404040", "linewidth": 1.1},
        boxprops={"edgecolor": "#404040", "linewidth": 1.1},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#ffffff")
        patch.set_alpha(0.92)

    rng = np.random.default_rng(42)
    for pos, label, values in zip(positions, order, groups):
        if len(values) == 0:
            continue
        sample_n = min(len(values), 500)
        idx = rng.choice(len(values), size=sample_n, replace=False)
        sampled = values[idx]
        jitter = rng.normal(0, 0.07, size=sample_n)
        ax.scatter(
            np.full(sample_n, pos) + jitter,
            sampled,
            s=14,
            alpha=0.28,
            color=FINAL_LABEL_COLORS.get(label, "#808080"),
            edgecolors="none",
        )

    tick_labels = []
    for label, values in zip(order, groups):
        tick_labels.append(f"{label}\n(n={len(values)})")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Oct-4 mean intensity")
    ax.set_title("Oct-4 mean intensity by cluster")
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def trim_mean_intensity_outliers_iqr(rel_df: pd.DataFrame, whisker: float = 1.5) -> pd.DataFrame:
    if len(rel_df) == 0:
        return rel_df.copy()
    q1 = float(rel_df["mean_intensity"].quantile(0.25))
    q3 = float(rel_df["mean_intensity"].quantile(0.75))
    iqr = q3 - q1
    low = q1 - whisker * iqr
    high = q3 + whisker * iqr
    return rel_df.loc[rel_df["mean_intensity"].between(low, high)].copy()


def save_cluster2_fraction_by_intensity_bin_figure(rel_df: pd.DataFrame, out_path: Path, *, trim_outliers: bool = False) -> None:
    work_df = rel_df.loc[rel_df["final_state_label"].isin(["cluster1", "cluster2", "uncertain"])].copy()
    if len(work_df) == 0:
        return
    if trim_outliers:
        work_df = trim_mean_intensity_outliers_iqr(work_df)
        if len(work_df) == 0:
            return

    bin_count = 10
    bins = np.linspace(float(work_df["mean_intensity"].min()), float(work_df["mean_intensity"].max()), bin_count + 1)
    bins = np.unique(bins)
    if len(bins) < 3:
        return
    work_df["intensity_bin"] = pd.cut(work_df["mean_intensity"], bins=bins, include_lowest=True)
    rows = []
    for _, sub in work_df.groupby("intensity_bin", observed=False):
        if len(sub) == 0:
            continue
        rows.append(
            {
                "bin_center": float(sub["mean_intensity"].mean()),
                "cluster2_fraction": float((sub["final_state_label"] == "cluster2").mean()),
                "n": int(len(sub)),
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(7.8, 6.2))
    ax1.plot(
        summary["bin_center"],
        summary["cluster2_fraction"],
        marker="o",
        linewidth=2.0,
        color=FINAL_LABEL_COLORS["cluster2"],
    )
    ax1.set_xlabel("Oct-4 mean intensity")
    ax1.set_ylabel("Cluster2 fraction", color=FINAL_LABEL_COLORS["cluster2"])
    ax1.tick_params(axis="y", labelcolor=FINAL_LABEL_COLORS["cluster2"])
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(axis="y", alpha=0.18, linewidth=0.6)

    ax2 = ax1.twinx()
    widths = np.diff(bins)[: len(summary)] * 0.85
    ax2.bar(
        summary["bin_center"],
        summary["n"],
        width=widths,
        color="#bfbfbf",
        alpha=0.25,
        edgecolor="none",
    )
    ax2.set_ylabel("Cells per bin", color="#666666")
    ax2.tick_params(axis="y", labelcolor="#666666")

    title = "Cluster2 fraction vs Oct-4 intensity bin"
    if trim_outliers:
        title += " (outliers removed)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = sibling_single_work_dir(Path(__file__))
    base04_path = base_dir / "04_unsupervised_learning.py"
    base04 = load_module_from_path("base04_double_cluster", base04_path)
    configure_base04(base04)

    print("Running double-staining 04 clustering:")
    print(f"- DAPI feature CSV : {INPUT_CSV}")
    print(f"- Oct-4 intensity  : {INTENSITY_CSV}")
    print(f"- DAPI image dir   : {DAPI_IMAGE_DIR}")
    print(f"- Mask dir         : {MASK_DIR}")

    base04.main()

    alias_duplicates = [
        base04.OUT_UMAP_EMBEDDING,
        base04.OUT_FIG_UMAP_RAW,
        base04.OUT_FIG_UMAP,
    ]
    for alias_path in alias_duplicates:
        if alias_path.exists():
            alias_path.unlink()

    clustered_csv = base04.FINAL_CLUSTERED_CSV if base04.FINAL_CLUSTERED_CSV.exists() else base04.OUT_CLUSTERED_CSV
    clustered_df = pd.read_csv(clustered_csv)
    orientation_info = resolve_cluster_orientation(clustered_df)
    corrected_df = relabel_by_orientation(clustered_df, orientation_info, base04)

    corrected_df.to_csv(clustered_csv, index=False, encoding="utf-8-sig")
    if base04.OUT_CLUSTERED_CSV.exists() and base04.OUT_CLUSTERED_CSV != clustered_csv:
        corrected_df.to_csv(base04.OUT_CLUSTERED_CSV, index=False, encoding="utf-8-sig")

    refresh_final_figures(base04, corrected_df)
    refresh_summary_and_run_info(base04, corrected_df, orientation_info)
    overlay_paths = save_full_overlays(base04, corrected_df)

    relation_csv = base04.OUT_DEVIATION_INTENSITY_CSV
    relation_fig = base04.OUT_DEVIATION_INTENSITY_FIG
    relation_json = base04.OUT_DIR / "deviation_score_vs_oct4_mean_intensity_summary.json"

    summary = save_score_intensity_relationship(
        scored_df=corrected_df,
        intensity_csv=INTENSITY_CSV,
        out_csv=relation_csv,
        out_fig=relation_fig,
        out_summary_json=relation_json,
        score_candidates=["deviated_score", "gmm_prob_deviated_raw"],
        intensity_col="mean_intensity",
        state_candidates=["final_state_label", "state_label"],
        score_label="DAPI deviation score",
        intensity_label="Oct-4 mean intensity",
        title_prefix="DAPI deviation score vs Oct-4 mean intensity",
    )
    rel_df = build_score_intensity_df(corrected_df)
    intensity_violin_fig = INPUT_CSV.parent / "oct4_mean_intensity_by_cluster.png"
    cluster2_bin_fig = INPUT_CSV.parent / "cluster2_fraction_by_oct4_intensity_bin.png"
    cluster2_bin_no_outlier_fig = INPUT_CSV.parent / "cluster2_fraction_by_oct4_intensity_bin_no_outliers.png"
    save_oct4_intensity_by_cluster_figure(rel_df, intensity_violin_fig)
    save_cluster2_fraction_by_intensity_bin_figure(rel_df, cluster2_bin_fig)
    save_cluster2_fraction_by_intensity_bin_figure(rel_df, cluster2_bin_no_outlier_fig, trim_outliers=True)

    cleanup_extra_outputs(base04)

    print(f"[ok] UMAP 2D saved: {base04.OUT_FIG_UMAP2}")
    print(f"[ok] UMAP 3D saved: {base04.OUT_FIG_UMAP3}")
    for overlay_path in overlay_paths:
        print(f"[ok] Overlay saved: {overlay_path}")
    print(f"[ok] Oct-4 validation figure saved: {relation_fig}")
    print(f"[ok] Oct-4 intensity by cluster saved: {intensity_violin_fig}")
    print(f"[ok] Cluster2 fraction by intensity bin saved: {cluster2_bin_fig}")
    print(f"[ok] Cluster2 fraction by intensity bin (no outliers) saved: {cluster2_bin_no_outlier_fig}")
    print("Color legend:")
    print("cluster1 = green")
    print("cluster2 = red")
    print("uncertain = yellow")
    print("[info] Cluster orientation resolution:")
    print(json.dumps(orientation_info, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
