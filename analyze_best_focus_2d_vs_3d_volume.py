from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models
from scipy.stats import pearsonr, spearmanr
from skimage import measure, segmentation


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


def _data_root() -> Path:
    """数据根从 naruchit_pipeline/config/project.json 读；可用 NARUCHIT_DATA_ROOT 覆盖。"""
    override = os.environ.get("NARUCHIT_DATA_ROOT")
    if override:
        return Path(override)
    cfg = WORKSPACE / "naruchit_pipeline" / "config" / "project.json"
    if not cfg.exists():
        raise RuntimeError(f"找不到配置 {cfg}，也没有设置 NARUCHIT_DATA_ROOT")
    return Path(json.loads(cfg.read_text(encoding="utf-8-sig"))["data_root"])


RAW_DIR = _data_root() / "03_Raw data" / "Immunostaining" / "Confocal - YAP"
ANALYSIS_ROOT = (
    WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "dapi_3d" / "yap"
)
INPUT_3D = ANALYSIS_ROOT / "analysis_summary" / "nucleus_3d_yap_cell_metrics.csv"
MAX_SLICE_RESULTS = ANALYSIS_ROOT / "area_volume_correlations" / "condition_feature_volume_correlations.csv"
OUTPUT_DIR = ANALYSIS_ROOT / "best_focus_2d_vs_3d"
MASK_2D_DIR = OUTPUT_DIR / "masks_2d"
QC_DIR = OUTPUT_DIR / "qc"

XY_UM = 0.17578
SAMPLE_ORDER = [
    "D20D", "D30D", "D31D", "D33D", "D40D",
    "D41D", "D43D", "D50D", "D51D", "D53D",
]
AREA_LIMITS_UM2 = (30.0, 260.0)
MIN_MATCH_FRACTION_2D = 0.50
MIN_MATCH_IOU = 0.30
CELLPOSE_DIAMETER_PX = 55.0

FEATURE_LABELS = {
    "area_um2": "Area",
    "perimeter_crofton_um": "Perimeter",
    "equivalent_diameter_um": "Equivalent diameter",
    "major_axis_um": "Major axis",
    "minor_axis_um": "Minor axis",
    "aspect_ratio": "Aspect ratio",
    "circularity": "Circularity",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "extent": "Extent",
    "dapi_mean_norm": "DAPI mean (normalized)",
    "dapi_std_norm": "DAPI SD (normalized)",
    "dapi_cv": "DAPI CV",
    "dapi_iqr_norm": "DAPI IQR (normalized)",
    "dapi_entropy_bits": "DAPI intensity entropy",
}
FEATURES = list(FEATURE_LABELS)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError(f"Expected ZCYX metadata, got axes={axes}, shape={array.shape}")
    if array.ndim == 4 and array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def tenengrad_focus(image: np.ndarray, sigma: float = 1.5, crop: float = 0.10) -> float:
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, [1, 99])
    normalized = np.clip((image - p1) / (p99 - p1 + 1e-6), 0, 1)
    blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=sigma, sigmaY=sigma)
    height, width = blurred.shape
    central = blurred[
        int(height * crop):int(height * (1 - crop)),
        int(width * crop):int(width * (1 - crop)),
    ]
    gx = cv2.Sobel(central, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(central, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def robust_normalize(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size == 0:
        return np.zeros_like(image, dtype=np.float32), 0.0, 1.0
    low, high = np.percentile(positive, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    return normalized, float(low), float(high)


def entropy_bits(values: np.ndarray, bins: int = 16) -> float:
    if values.size < 2:
        return np.nan
    counts, _ = np.histogram(np.clip(values, 0, 1), bins=bins, range=(0, 1))
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def segment_plane(model: models.CellposeModel, sample: str, plane: np.ndarray) -> np.ndarray:
    path = MASK_2D_DIR / f"{sample}_best_focus_2d_mask.npz"
    if path.exists():
        return np.load(path)["labels"].astype(np.int32)
    masks, _, _ = model.eval(
        plane,
        channel_axis=None,
        z_axis=None,
        normalize={"lowhigh": None, "percentile": [1.0, 99.8], "normalize": True},
        diameter=CELLPOSE_DIAMETER_PX,
        cellprob_threshold=0.0,
        min_size=300,
        bsize=256,
    )
    labels = np.asarray(masks, dtype=np.int32)
    np.savez_compressed(path, labels=labels)
    return labels


def border_labels(labels: np.ndarray) -> set[int]:
    ids = set(np.unique(labels[0])) | set(np.unique(labels[-1]))
    ids |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    ids.discard(0)
    return ids


def extract_and_match(
    sample: str,
    labels_2d: np.ndarray,
    labels_3d_plane: np.ndarray,
    image_raw: np.ndarray,
    accepted_3d: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_norm, _, _ = robust_normalize(image_raw)
    accepted_labels = set(accepted_3d["label"].astype(int))
    excluded_border = border_labels(labels_2d)
    candidates = []
    rejected = []

    for region in measure.regionprops(labels_2d, intensity_image=image_raw):
        area_um2 = float(region.area * XY_UM**2)
        reason = None
        if region.label in excluded_border:
            reason = "2d_border"
        elif not (AREA_LIMITS_UM2[0] <= area_um2 <= AREA_LIMITS_UM2[1]):
            reason = "2d_area_qc"

        coords = region.coords
        overlap_values = labels_3d_plane[coords[:, 0], coords[:, 1]]
        overlap_values = overlap_values[np.isin(overlap_values, list(accepted_labels))]
        if overlap_values.size:
            labels, counts = np.unique(overlap_values, return_counts=True)
            best_index = int(np.argmax(counts))
            label_3d = int(labels[best_index])
            overlap_px = int(counts[best_index])
            cross_section_px = int(np.sum(labels_3d_plane == label_3d))
            match_fraction_2d = overlap_px / float(region.area)
            match_iou = overlap_px / float(region.area + cross_section_px - overlap_px)
        else:
            label_3d = 0
            overlap_px = 0
            match_fraction_2d = 0.0
            match_iou = 0.0

        if reason is None and label_3d == 0:
            reason = "no_accepted_3d_overlap"
        if reason is None and match_fraction_2d < MIN_MATCH_FRACTION_2D:
            reason = "low_2d_overlap_fraction"
        if reason is None and match_iou < MIN_MATCH_IOU:
            reason = "low_iou"

        perimeter_um = float(region.perimeter_crofton * XY_UM)
        major_um = float(region.axis_major_length * XY_UM)
        minor_um = float(region.axis_minor_length * XY_UM)
        values_norm = image_norm[coords[:, 0], coords[:, 1]]
        values_raw = image_raw[coords[:, 0], coords[:, 1]].astype(float)
        q25, q75 = np.percentile(values_norm, [25, 75])
        raw_mean = float(np.mean(values_raw))
        raw_std = float(np.std(values_raw, ddof=1)) if values_raw.size > 1 else 0.0
        row = {
            "sample": sample,
            "label_2d": int(region.label),
            "label_3d": label_3d,
            "overlap_px": overlap_px,
            "match_fraction_2d": match_fraction_2d,
            "match_iou": match_iou,
            "area_um2": area_um2,
            "perimeter_crofton_um": perimeter_um,
            "equivalent_diameter_um": float(region.equivalent_diameter_area * XY_UM),
            "major_axis_um": major_um,
            "minor_axis_um": minor_um,
            "aspect_ratio": major_um / minor_um if minor_um > 0 else np.nan,
            "circularity": float(4 * np.pi * area_um2 / perimeter_um**2) if perimeter_um > 0 else np.nan,
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "extent": float(region.extent),
            "dapi_mean_norm": float(np.mean(values_norm)),
            "dapi_std_norm": float(np.std(values_norm, ddof=1)) if values_norm.size > 1 else 0.0,
            "dapi_cv": raw_std / raw_mean if raw_mean > 0 else np.nan,
            "dapi_iqr_norm": float(q75 - q25),
            "dapi_entropy_bits": entropy_bits(values_norm),
            "rejection_reason": reason,
        }
        if reason is None:
            candidates.append(row)
        else:
            rejected.append(row)

    candidates_frame = pd.DataFrame(candidates)
    if len(candidates_frame):
        candidates_frame = candidates_frame.sort_values(
            ["match_iou", "match_fraction_2d"], ascending=False
        )
        duplicate_3d = candidates_frame.duplicated("label_3d", keep="first")
        duplicate_rows = candidates_frame[duplicate_3d].copy()
        if len(duplicate_rows):
            duplicate_rows["rejection_reason"] = "duplicate_match_to_3d"
            rejected.extend(duplicate_rows.to_dict("records"))
        candidates_frame = candidates_frame[~duplicate_3d].copy()
    return candidates_frame, pd.DataFrame(rejected)


def safe_correlation(frame: pd.DataFrame, feature: str) -> dict:
    pair = frame[[feature, "volume_um3"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 4 or pair[feature].nunique() < 2:
        return {"n": len(pair), "spearman_rho": np.nan, "pearson_r": np.nan}
    return {
        "n": int(len(pair)),
        "spearman_rho": float(spearmanr(pair[feature], pair["volume_um3"]).statistic),
        "pearson_r": float(pearsonr(pair[feature], pair["volume_um3"]).statistic),
    }


def stratified_rho(frame: pd.DataFrame, feature: str) -> float:
    pair = frame[["sample", feature, "volume_um3"]].dropna()
    pair = pair.groupby("sample", observed=True).filter(lambda group: len(group) >= 4)
    if len(pair) < 4:
        return np.nan
    x = pair.groupby("sample", observed=True)[feature].rank(pct=True)
    y = pair.groupby("sample", observed=True)["volume_um3"].rank(pct=True)
    return float(pearsonr(x, y).statistic)


def condition_centered_pearson(frame: pd.DataFrame, feature: str) -> float:
    pair = frame[["sample", feature, "volume_um3"]].replace([np.inf, -np.inf], np.nan).dropna()
    pair = pair.groupby("sample", observed=True).filter(lambda group: len(group) >= 4)
    if len(pair) < 4:
        return np.nan
    x = pair[feature] - pair.groupby("sample", observed=True)[feature].transform("mean")
    y = pair["volume_um3"] - pair.groupby("sample", observed=True)["volume_um3"].transform("mean")
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(pearsonr(x, y).statistic)


def correlation_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    condition_rows = []
    for feature in FEATURES:
        overall_rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                **safe_correlation(cells, feature),
                "condition_stratified_spearman_rho": stratified_rho(cells, feature),
                "condition_centered_pearson_r": condition_centered_pearson(cells, feature),
            }
        )
        for sample in SAMPLE_ORDER:
            subset = cells[cells["sample"] == sample]
            condition_rows.append(
                {
                    "sample": sample,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    **safe_correlation(subset, feature),
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values(
        "condition_stratified_spearman_rho", key=lambda series: series.abs(), ascending=False
    )
    return overall, pd.DataFrame(condition_rows)


def save_qc(sample: str, image: np.ndarray, labels_2d: np.ndarray, labels_3d_plane: np.ndarray, focus_z: int) -> None:
    normalized, _, _ = robust_normalize(image)
    overlay_2d = segmentation.mark_boundaries(normalized, labels_2d, color=(0, 1, 1), mode="thick")
    overlay_3d = segmentation.mark_boundaries(normalized, labels_3d_plane, color=(1, 0, 0), mode="thick")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Best-focus DAPI, z={focus_z}")
    axes[1].imshow(overlay_2d)
    axes[1].set_title("Independent 2D Cellpose masks")
    axes[2].imshow(overlay_3d)
    axes[2].set_title("3D labels intersecting this plane")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(sample)
    fig.savefig(QC_DIR / f"{sample}_best_focus_qc.png", dpi=180)
    plt.close(fig)


def plot_results(
    cells: pd.DataFrame,
    overall: pd.DataFrame,
    by_condition: pd.DataFrame,
    focus_scores: pd.DataFrame,
) -> None:
    area = overall.set_index("feature").loc["area_um2"]
    fig, ax = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    colors = dict(zip(SAMPLE_ORDER, plt.get_cmap("tab10").colors))
    for sample in SAMPLE_ORDER:
        subset = cells[cells["sample"] == sample]
        ax.scatter(subset["area_um2"], subset["volume_um3"], s=28, alpha=0.7, color=colors[sample], label=sample)
    ax.set_xlabel("Best-focus 2D nuclear area (µm²)")
    ax.set_ylabel("Matched 3D nuclear volume (µm³)")
    ax.set_title(
        f"Independent best-focus 2D segmentation\n"
        f"pooled Spearman ρ={area['spearman_rho']:.3f}; stratified ρ={area['condition_stratified_spearman_rho']:.3f}"
    )
    ax.legend(ncol=2, fontsize=8, title="Sample")
    fig.savefig(OUTPUT_DIR / "best_focus_area_vs_3d_volume.png", dpi=220)
    plt.close(fig)

    area_condition = by_condition[by_condition["feature"] == "area_um2"].set_index("sample")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    for ax, sample in zip(axes.flat, SAMPLE_ORDER):
        subset = cells[cells["sample"] == sample]
        stats = area_condition.loc[sample]
        ax.scatter(subset["area_um2"], subset["volume_um3"], s=24, alpha=0.7, color="#2563eb")
        rho_text = "NA" if not np.isfinite(stats["spearman_rho"]) else f"{stats['spearman_rho']:.3f}"
        ax.set_title(f"{sample}  n={int(stats['n'])}\nSpearman ρ={rho_text}")
        ax.set_xlabel("2D area (µm²)")
        ax.set_ylabel("3D volume (µm³)")
    fig.suptitle("Best-focus area–volume association within each sample", fontsize=16)
    fig.savefig(OUTPUT_DIR / "best_focus_area_volume_by_condition.png", dpi=220)
    plt.close(fig)

    matrix = by_condition.pivot(index="feature_label", columns="sample", values="spearman_rho")
    matrix = matrix.reindex([FEATURE_LABELS[feature] for feature in FEATURES])
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.55 else "black")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Best-focus 2D features versus 3D nuclear volume")
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_label("Within-condition Spearman ρ")
    fig.savefig(OUTPUT_DIR / "best_focus_feature_volume_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), constrained_layout=True)
    for ax, sample in zip(axes.flat, SAMPLE_ORDER):
        subset = focus_scores[focus_scores["sample"] == sample]
        ax.plot(subset["z_index"], subset["tenengrad_score"], marker="o", markersize=3)
        chosen = subset[subset["selected"]].iloc[0]
        ax.axvline(chosen["z_index"], color="#dc2626", linestyle="--")
        ax.set_title(f"{sample}: best z={int(chosen['z_index'])}")
        ax.set_xlabel("Z index")
        ax.set_ylabel("Tenengrad")
    fig.suptitle("Field-level best-focus selection", fontsize=16)
    fig.savefig(OUTPUT_DIR / "focus_score_curves.png", dpi=200)
    plt.close(fig)

    if MAX_SLICE_RESULTS.exists():
        old = pd.read_csv(MAX_SLICE_RESULTS)
        old = old[old["feature"] == "max_xy_area_um2"][["sample", "spearman_rho"]].rename(columns={"spearman_rho": "nucleus_specific_max_slice"})
        new = area_condition.reset_index()[["sample", "spearman_rho"]].rename(columns={"spearman_rho": "field_best_focus"})
        compare = old.merge(new, on="sample")
        compare.to_csv(OUTPUT_DIR / "max_slice_vs_best_focus_comparison.csv", index=False, encoding="utf-8-sig")
        x = np.arange(len(compare))
        width = 0.38
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        ax.bar(x - width / 2, compare["nucleus_specific_max_slice"], width, label="Nucleus-specific maximum slice", color="#94a3b8")
        ax.bar(x + width / 2, compare["field_best_focus"], width, label="One best-focus plane per field", color="#2563eb")
        ax.set_xticks(x, compare["sample"])
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Within-sample Spearman ρ")
        ax.set_title("Area–volume correlation: optimistic maximum slice versus realistic best-focus plane")
        ax.legend(frameon=False)
        fig.savefig(OUTPUT_DIR / "max_slice_vs_best_focus_comparison.png", dpi=220)
        plt.close(fig)


def write_report(
    cells: pd.DataFrame,
    overall: pd.DataFrame,
    by_condition: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> None:
    area = overall.set_index("feature").loc["area_um2"]
    area_conditions = by_condition[by_condition["feature"] == "area_um2"].set_index("sample")
    condition_lines = []
    for sample in SAMPLE_ORDER:
        stats = area_conditions.loc[sample]
        summary = sample_summary.set_index("sample").loc[sample]
        rho = "NA" if not np.isfinite(stats["spearman_rho"]) else f"{stats['spearman_rho']:.3f}"
        condition_lines.append(
            f"| {sample} | {int(summary['best_focus_z'])} | {int(summary['n_2d_objects'])} | "
            f"{int(summary['n_matched'])} | {summary['match_rate']:.1%} | {rho} | {stats['pearson_r']:.3f} |"
        )
    top = overall.reindex(overall["condition_stratified_spearman_rho"].abs().sort_values(ascending=False).index).head(8)
    top_lines = [
        f"| {row.feature_label} | {row.spearman_rho:.3f} | {row.pearson_r:.3f} | "
        f"{row.condition_stratified_spearman_rho:.3f} | {row.condition_centered_pearson_r:.3f} |"
        for row in top.itertuples(index=False)
    ]
    report = f"""# 最佳焦平面二维核特征与三维核体积

## 方法修正

旧分析为每个细胞核分别选择面积最大的Z截面，因此面积—体积关系包含明显的几何必然性。本分析改为：

1. 每个OIB视野通过Tenengrad清晰度评分选择一个统一的最佳焦平面；
2. 只在该平面使用Cellpose进行独立二维分割；
3. 通过空间重叠将二维核匹配到三维核；
4. 使用独立二维掩膜计算面积和其他特征。

## 主要结果

- 成功匹配二维—三维细胞核：**{len(cells)}**个。
- 最佳焦平面二维面积与三维体积，总体Spearman ρ = **{area['spearman_rho']:.3f}**。
- 总体Pearson r = **{area['pearson_r']:.3f}**。
- 去除样本/条件整体偏移后的分层ρ = **{area['condition_stratified_spearman_rho']:.3f}**。
- 条件中心化Pearson r = **{area['condition_centered_pearson_r']:.3f}**。

## 每个条件

| 样本 | 最佳Z层 | 2D分割核数 | 成功匹配 | 匹配率 | Spearman ρ | Pearson r |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(condition_lines)}

## 与三维体积关系最强的二维特征

| 二维特征 | 总体Spearman ρ | 总体Pearson r | 条件分层Spearman ρ | 条件中心化Pearson r |
|---|---:|---:|---:|---:|
{chr(10).join(top_lines)}

## 解释

- 该结果比“每个核最大截面”更接近真实单层Hoechst成像。
- 一个焦平面不一定经过所有细胞核的中心，因此面积—体积相关性预期会下降；这种下降正是需要测量的现实误差。
- 每个条件目前仍只有一个OIB视野，细胞核属于技术子样本。结果用于方法验证，不能当作生物学重复。
- 下一步应在独立活细胞Hoechst图像中使用相同选焦和二维分割流程，并按培养批次进行验证。
"""
    (OUTPUT_DIR / "analysis_report_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_2D_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    metrics_3d = pd.read_csv(INPUT_3D)
    accepted = metrics_3d[metrics_3d["qc_pass_uniform"]].copy()
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    matched_frames = []
    rejected_frames = []
    focus_rows = []
    sample_rows = []

    for sample in SAMPLE_ORDER:
        print(f"Processing {sample}...", flush=True)
        with oiffile.OifFile(RAW_DIR / f"{sample}.oib") as handle:
            stack = ensure_zcyx(handle.asarray(), handle.axes)
        dapi = stack[:, 0].astype(np.float32)
        scores = np.asarray([tenengrad_focus(plane) for plane in dapi])
        focus_z = int(np.argmax(scores))
        for z_index, score in enumerate(scores):
            focus_rows.append(
                {
                    "sample": sample,
                    "z_index": z_index,
                    "tenengrad_score": float(score),
                    "selected": z_index == focus_z,
                }
            )

        labels_2d = segment_plane(model, sample, dapi[focus_z])
        labels_3d = np.load(ANALYSIS_ROOT / sample / "dapi_3d_masks.npz")["raw_labels"].astype(np.int32)
        sample_accepted = accepted[accepted["sample"] == sample]
        matched, rejected = extract_and_match(
            sample,
            labels_2d,
            labels_3d[focus_z],
            dapi[focus_z],
            sample_accepted,
        )
        if len(matched):
            matched_frames.append(matched)
        if len(rejected):
            rejected_frames.append(rejected)
        save_qc(sample, dapi[focus_z], labels_2d, labels_3d[focus_z], focus_z)
        n_objects = int(labels_2d.max())
        sample_rows.append(
            {
                "sample": sample,
                "best_focus_z": focus_z,
                "n_z_planes": int(len(dapi)),
                "best_focus_score": float(scores[focus_z]),
                "second_focus_score": float(np.partition(scores, -2)[-2]) if len(scores) > 1 else np.nan,
                "n_2d_objects": n_objects,
                "n_matched": int(len(matched)),
                "match_rate": float(len(matched) / n_objects) if n_objects else np.nan,
                "n_accepted_3d_total": int(len(sample_accepted)),
                "n_accepted_3d_intersecting_plane": int(sample_accepted["label"].isin(np.unique(labels_3d[focus_z])).sum()),
            }
        )

    matched = pd.concat(matched_frames, ignore_index=True)
    rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    metadata_columns = [
        "sample", "label", "volume_um3", "surface_area_um2", "height_um",
        "effective_thickness_um", "sphericity", "culture_day", "paper_day", "regimen", "regimen_cn",
    ]
    cells = matched.merge(
        accepted[metadata_columns],
        left_on=["sample", "label_3d"],
        right_on=["sample", "label"],
        how="inner",
        validate="one_to_one",
    )
    overall, by_condition = correlation_tables(cells)
    focus_scores = pd.DataFrame(focus_rows)
    sample_summary = pd.DataFrame(sample_rows)

    cells.to_csv(OUTPUT_DIR / "matched_best_focus_2d_to_3d_nuclei.csv", index=False, encoding="utf-8-sig")
    rejected.to_csv(OUTPUT_DIR / "rejected_2d_matches.csv", index=False, encoding="utf-8-sig")
    focus_scores.to_csv(OUTPUT_DIR / "focus_scores_by_z.csv", index=False, encoding="utf-8-sig")
    sample_summary.to_csv(OUTPUT_DIR / "sample_focus_and_matching_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_DIR / "overall_best_focus_correlations.csv", index=False, encoding="utf-8-sig")
    by_condition.to_csv(OUTPUT_DIR / "condition_best_focus_correlations.csv", index=False, encoding="utf-8-sig")

    plot_results(cells, overall, by_condition, focus_scores)
    write_report(cells, overall, by_condition, sample_summary)
    settings = {
        "focus_method": "Tenengrad, sigma=1.5, central 80% crop",
        "one_focus_plane_per_oib": True,
        "independent_2d_segmentation": "Cellpose cpsam",
        "cellpose_diameter_px": CELLPOSE_DIAMETER_PX,
        "area_limits_um2": AREA_LIMITS_UM2,
        "minimum_match_fraction_2d": MIN_MATCH_FRACTION_2D,
        "minimum_match_iou": MIN_MATCH_IOU,
    }
    (OUTPUT_DIR / "analysis_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    area = overall.set_index("feature").loc["area_um2"]
    print(f"Matched nuclei: {len(cells)}", flush=True)
    print(
        f"Best-focus area-volume: pooled rho={area['spearman_rho']:.3f}, "
        f"stratified rho={area['condition_stratified_spearman_rho']:.3f}",
        flush=True,
    )
    print(sample_summary[["sample", "best_focus_z", "n_2d_objects", "n_matched", "match_rate"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
