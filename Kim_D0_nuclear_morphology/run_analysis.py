from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import jensenshannon
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
MASK_DIR = OUT / "masks"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"

DATA_DIR = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\IN Cell\D0_3_17_21_E-cad N-cad\10x\D0"
)
SAMPLES = {
    "HA00": {"dose_nM": 0, "path": DATA_DIR / "HA00_DAPI.tif"},
    "HA01": {"dose_nM": 1, "path": DATA_DIR / "HA01_DAPI.tif"},
    "HA05": {"dose_nM": 5, "path": DATA_DIR / "HA05_DAPI.tif"},
    "HA10": {"dose_nM": 10, "path": DATA_DIR / "HA10_DAPI.tif"},
}

RANDOM_SEED = 20261020
MAX_FIT_PER_SAMPLE = 3500
K_CANDIDATES = range(2, 7)
GMM_SEEDS = [7, 17, 29, 43, 71]

# The primary model deliberately uses geometry only. Intensity/texture are retained
# as descriptive outputs but cannot dominate clusters when acquisition brightness differs.
MODEL_FEATURES = [
    "log_area_px",
    "circularity",
    "eccentricity",
    "solidity",
    "extent",
    "log_aspect_ratio",
    "perimeter_norm",
]
DISPLAY_FEATURES = [
    "area_px",
    "circularity",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "intensity_cv",
]


def ensure_dirs() -> None:
    for p in [OUT, MASK_DIR, FIG_DIR, TABLE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def normalize_image(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99.7])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def segment_all() -> None:
    missing = [name for name in SAMPLES if not (MASK_DIR / f"{name}_mask.npy").exists()]
    if not missing:
        print("[segment] all masks already exist")
        return

    import torch
    from cellpose import models

    use_gpu = bool(torch.cuda.is_available())
    print(f"[segment] loading Cellpose-SAM; gpu={use_gpu}")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    for name in missing:
        img = tifffile.imread(SAMPLES[name]["path"])
        norm = normalize_image(img)
        rgb = np.repeat(norm[..., None], 3, axis=2)
        masks, _, _ = model.eval(
            rgb,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        masks = masks.astype(np.int32)
        np.save(MASK_DIR / f"{name}_mask.npy", masks)
        print(f"[segment] {name}: raw objects={int(masks.max())}")


def touches_border(p, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = p.bbox
    return r0 == 0 or c0 == 0 or r1 == shape[0] or c1 == shape[1]


def extract_features() -> pd.DataFrame:
    all_rows: list[dict] = []
    qc_rows: list[dict] = []
    for name, meta in SAMPLES.items():
        img_raw = tifffile.imread(meta["path"])
        img = normalize_image(img_raw)
        mask = np.load(MASK_DIR / f"{name}_mask.npy")
        raw_props = regionprops(mask, intensity_image=img)
        areas = np.array([p.area for p in raw_props if not touches_border(p, img.shape)], dtype=float)
        if len(areas) < 20:
            raise RuntimeError(f"Too few segmented nuclei in {name}: {len(areas)}")
        q1, q3 = np.percentile(areas, [25, 75])
        iqr = q3 - q1
        lower = max(25.0, q1 - 2.5 * iqr)
        upper = min(2500.0, q3 + 3.5 * iqr)

        kept_labels = []
        reason_counts = {"border": 0, "area": 0, "shape": 0}
        for p in raw_props:
            if touches_border(p, img.shape):
                reason_counts["border"] += 1
                continue
            if not (lower <= p.area <= upper):
                reason_counts["area"] += 1
                continue
            if p.solidity < 0.65 or p.minor_axis_length <= 1 or p.perimeter <= 0:
                reason_counts["shape"] += 1
                continue

            pix = p.image_intensity[p.image]
            mean_i = float(np.mean(pix))
            std_i = float(np.std(pix))
            area = float(p.area)
            perim = float(p.perimeter)
            aspect = float(p.major_axis_length / max(p.minor_axis_length, 1e-6))
            row = {
                "sample": name,
                "dose_nM": int(meta["dose_nM"]),
                "label": int(p.label),
                "centroid_y": float(p.centroid[0]),
                "centroid_x": float(p.centroid[1]),
                "bbox_y0": int(p.bbox[0]),
                "bbox_x0": int(p.bbox[1]),
                "bbox_y1": int(p.bbox[2]),
                "bbox_x1": int(p.bbox[3]),
                "area_px": area,
                "perimeter_px": perim,
                "equivalent_diameter_px": float(p.equivalent_diameter_area),
                "major_axis_px": float(p.major_axis_length),
                "minor_axis_px": float(p.minor_axis_length),
                "aspect_ratio": aspect,
                "circularity": float(4 * math.pi * area / (perim * perim)),
                "eccentricity": float(p.eccentricity),
                "solidity": float(p.solidity),
                "extent": float(p.extent),
                "mean_intensity_norm": mean_i,
                "std_intensity_norm": std_i,
                "intensity_cv": float(std_i / max(mean_i, 1e-6)),
                "intensity_q25_norm": float(np.percentile(pix, 25)),
                "intensity_median_norm": float(np.median(pix)),
                "intensity_q75_norm": float(np.percentile(pix, 75)),
                "log_area_px": float(np.log1p(area)),
                "log_aspect_ratio": float(np.log(max(aspect, 1e-6))),
                "perimeter_norm": float(perim / max(np.sqrt(area), 1e-6)),
            }
            all_rows.append(row)
            kept_labels.append(int(p.label))

        qc_rows.append(
            {
                "sample": name,
                "dose_nM": meta["dose_nM"],
                "raw_objects": len(raw_props),
                "kept_objects": len(kept_labels),
                "kept_fraction": len(kept_labels) / max(len(raw_props), 1),
                "area_lower_px": lower,
                "area_upper_px": upper,
                **{f"removed_{k}": v for k, v in reason_counts.items()},
            }
        )

        qc_mask = np.where(np.isin(mask, kept_labels), mask, 0).astype(np.int32)
        np.save(MASK_DIR / f"{name}_mask_qc.npy", qc_mask)
        print(f"[features] {name}: kept {len(kept_labels)}/{len(raw_props)}")

    df = pd.DataFrame(all_rows)
    pd.DataFrame(qc_rows).to_csv(TABLE_DIR / "segmentation_qc.csv", index=False)
    return df


def balanced_indices(df: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idxs = []
    n_each = min(MAX_FIT_PER_SAMPLE, int(df.groupby("sample").size().min()))
    for _, sub in df.groupby("sample", sort=True):
        idxs.extend(rng.choice(sub.index.to_numpy(), size=n_each, replace=False).tolist())
    return np.asarray(sorted(idxs), dtype=int)


def fit_joint_model(df: pd.DataFrame):
    fit_idx = balanced_indices(df, RANDOM_SEED)
    X_raw = df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    X_raw = X_raw.fillna(X_raw.loc[fit_idx].median())
    scaler = RobustScaler(quantile_range=(10, 90)).fit(X_raw.loc[fit_idx])
    X_scaled = scaler.transform(X_raw)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_SEED).fit(X_scaled[fit_idx])
    X_pca = pca.transform(X_scaled)
    X_fit = X_pca[fit_idx]

    rng = np.random.default_rng(RANDOM_SEED)
    sil_idx = rng.choice(len(X_fit), size=min(5000, len(X_fit)), replace=False)
    selection_rows = []
    best_models = {}
    for k in K_CANDIDATES:
        models_k = []
        predictions = []
        for seed in GMM_SEEDS:
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-5,
                n_init=2,
                max_iter=500,
                random_state=seed,
            ).fit(X_fit)
            pred = model.predict(X_fit)
            models_k.append(model)
            predictions.append(pred)
        bics = np.array([m.bic(X_fit) for m in models_k])
        best_i = int(np.argmin(bics))
        best_model = models_k[best_i]
        pred_best = predictions[best_i]
        pair_aris = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pair_aris.append(adjusted_rand_score(predictions[i], predictions[j]))
        counts = np.bincount(pred_best, minlength=k)
        min_fraction = float(counts.min() / counts.sum())
        sil = float(silhouette_score(X_fit[sil_idx], pred_best[sil_idx]))
        selection_rows.append(
            {
                "k": k,
                "bic": float(bics[best_i]),
                "silhouette": sil,
                "stability_ari": float(np.mean(pair_aris)),
                "min_cluster_fraction": min_fraction,
            }
        )
        best_models[k] = best_model

    sel = pd.DataFrame(selection_rows)
    eligible = sel[sel["min_cluster_fraction"] >= 0.03].copy()
    if eligible.empty:
        eligible = sel.copy()
    eligible["rank_bic"] = eligible["bic"].rank(method="min", ascending=True)
    eligible["rank_silhouette"] = eligible["silhouette"].rank(method="min", ascending=False)
    eligible["rank_stability"] = eligible["stability_ari"].rank(method="min", ascending=False)
    eligible["rank_sum"] = eligible[["rank_bic", "rank_silhouette", "rank_stability"]].sum(axis=1)
    best_k = int(eligible.sort_values(["rank_sum", "k"]).iloc[0]["k"])
    sel = sel.merge(eligible[["k", "rank_bic", "rank_silhouette", "rank_stability", "rank_sum"]], how="left", on="k")
    sel["selected"] = sel["k"] == best_k

    model = best_models[best_k]
    raw_cluster = model.predict(X_pca)
    probs = model.predict_proba(X_pca)
    # Stable neutral labels: order components from smaller to larger median nuclear area.
    med_area = pd.Series(df["area_px"].to_numpy()).groupby(raw_cluster).median().sort_values()
    raw_order = med_area.index.to_list()
    raw_to_rank = {int(raw): i + 1 for i, raw in enumerate(raw_order)}
    clusters = np.array([f"cluster_{raw_to_rank[int(x)]}" for x in raw_cluster], dtype=object)
    ordered_probs = np.column_stack([probs[:, int(raw)] for raw in raw_order])

    out = df.copy()
    out["pca_1"] = X_pca[:, 0]
    out["pca_2"] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0
    out["cluster"] = clusters
    out["cluster_confidence"] = probs.max(axis=1)
    for i in range(best_k):
        out[f"prob_cluster_{i+1}"] = ordered_probs[:, i]
    return out, sel, scaler, pca, model, fit_idx, best_k


def cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    prof = df.groupby("cluster")[DISPLAY_FEATURES].median()
    all_med = df[DISPLAY_FEATURES].median()
    all_iqr = df[DISPLAY_FEATURES].quantile(0.75) - df[DISPLAY_FEATURES].quantile(0.25)
    z = (prof - all_med) / all_iqr.replace(0, np.nan)
    return z.fillna(0)


def summarize_samples(df: pd.DataFrame, best_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [f"cluster_{i}" for i in range(1, best_k + 1)]
    rows = []
    block_rows = []
    for (sample, dose), sub in df.groupby(["sample", "dose_nM"], sort=True):
        props = sub["cluster"].value_counts(normalize=True).reindex(labels, fill_value=0.0)
        entropy = float(-(props[props > 0] * np.log(props[props > 0])).sum() / np.log(best_k))
        centroid = sub[["pca_1", "pca_2"]].median().to_numpy()
        dispersion = float(np.median(np.linalg.norm(sub[["pca_1", "pca_2"]].to_numpy() - centroid, axis=1)))
        row = {
            "sample": sample,
            "dose_nM": dose,
            "n_nuclei": len(sub),
            "normalized_cluster_entropy": entropy,
            "simpson_diversity": float(1 - np.square(props).sum()),
            "pca_dispersion_median": dispersion,
            "area_cv": float(sub["area_px"].std() / sub["area_px"].mean()),
            "circularity_iqr": float(sub["circularity"].quantile(0.75) - sub["circularity"].quantile(0.25)),
        }
        row.update({f"fraction_{lab}": float(props[lab]) for lab in labels})
        rows.append(row)

        overall = props.to_numpy() + 1e-9
        for by in range(4):
            for bx in range(4):
                tile = sub[
                    (sub["centroid_y"] >= by * 512)
                    & (sub["centroid_y"] < (by + 1) * 512)
                    & (sub["centroid_x"] >= bx * 512)
                    & (sub["centroid_x"] < (bx + 1) * 512)
                ]
                tile_props = tile["cluster"].value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy() + 1e-9
                block_rows.append(
                    {
                        "sample": sample,
                        "dose_nM": dose,
                        "block_y": by,
                        "block_x": bx,
                        "n_nuclei": len(tile),
                        "js_distance_to_image": float(jensenshannon(tile_props, overall)),
                        **{f"fraction_{labels[i]}": float(tile_props[i] / tile_props.sum()) for i in range(best_k)},
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(block_rows)


def save_figures(df: pd.DataFrame, sel: pd.DataFrame, best_k: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    palette = dict(zip([f"cluster_{i}" for i in range(1, best_k + 1)], plt.get_cmap("tab10").colors[:best_k]))
    dose_order = [0, 1, 5, 10]

    # Raw-image montage
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (name, meta) in zip(axes, SAMPLES.items()):
        ax.imshow(normalize_image(tifffile.imread(meta["path"])), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name} ({meta['dose_nM']} nM)")
        ax.axis("off")
    fig.suptitle("D0 DAPI images used in the joint analysis")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_input_images.png", dpi=180)
    plt.close(fig)

    # Model selection
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(sel["k"], sel["bic"], "o-", color="#1f77b4")
    axes[0].set_title("BIC (lower is better)")
    axes[1].plot(sel["k"], sel["silhouette"], "o-", color="#d95f02")
    axes[1].set_title("Silhouette (higher is better)")
    axes[2].plot(sel["k"], sel["stability_ari"], "o-", color="#1b9e77")
    axes[2].set_title("Seed stability, ARI")
    for ax in axes:
        ax.axvline(best_k, color="black", linestyle="--", alpha=.65)
        ax.set_xlabel("Number of clusters")
        ax.set_xticks(sel["k"])
    fig.suptitle(f"Data-driven model selection: selected k = {best_k}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_model_selection.png", dpi=200)
    plt.close(fig)

    # PCA views (balanced plotting)
    rng = np.random.default_rng(RANDOM_SEED)
    plot_idx = []
    for _, sub in df.groupby("sample"):
        plot_idx.extend(rng.choice(sub.index, min(2500, len(sub)), replace=False))
    p = df.loc[plot_idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for cluster, sub in p.groupby("cluster"):
        axes[0].scatter(sub["pca_1"], sub["pca_2"], s=9, alpha=.5, linewidths=0, color=palette[cluster], label=cluster)
    dose_colors = dict(zip(dose_order, plt.get_cmap("viridis")(np.linspace(.1, .9, len(dose_order)))))
    for dose, sub in p.groupby("dose_nM"):
        axes[1].scatter(sub["pca_1"], sub["pca_2"], s=9, alpha=.45, linewidths=0, color=dose_colors[dose], label=str(dose))
    axes[0].set_title("Same feature space, colored by cluster")
    axes[1].set_title("Same feature space, colored by HA condition")
    for ax in axes:
        ax.legend(title="Cluster" if ax is axes[0] else "HA (nM)", frameon=True, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_joint_pca.png", dpi=220)
    plt.close(fig)

    # Cluster fractions
    props = pd.crosstab(df["dose_nM"], df["cluster"], normalize="index").reindex(dose_order).fillna(0)
    props = props.reindex(columns=[f"cluster_{i}" for i in range(1, best_k + 1)], fill_value=0)
    ax = props.plot(kind="bar", stacked=True, figsize=(8.5, 5), color=[palette[c] for c in props.columns], width=.75)
    ax.set_xlabel("HA condition (nM)")
    ax.set_ylabel("Fraction of QC-passed nuclei")
    ax.set_title("Nuclear morphology composition by sample")
    ax.legend(title="Joint morphology cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_cluster_fractions.png", dpi=220)
    plt.close()

    # Cluster profile heatmap
    z = cluster_profiles(df)
    fig, ax = plt.subplots(figsize=(9, max(3.2, best_k * .65)))
    vmax = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))
    im = ax.imshow(z.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(z.columns)), labels=z.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(z.index)), labels=z.index)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(z.iloc[i, j]) > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, label="Median difference / pooled IQR")
    ax.set_title("What each neutral cluster looks like")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_cluster_profiles.png", dpi=220)
    plt.close(fig)

    # Cluster overlays
    cmap = ListedColormap([(0, 0, 0, 0)] + [(*palette[f"cluster_{i}"], .85) for i in range(1, best_k + 1)])
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (name, meta) in zip(axes.flat, SAMPLES.items()):
        img = normalize_image(tifffile.imread(meta["path"]))
        mask = np.load(MASK_DIR / f"{name}_mask_qc.npy")
        map_label = dict(zip(df.loc[df["sample"] == name, "label"], df.loc[df["sample"] == name, "cluster"]))
        cluster_mask = np.zeros(mask.shape, dtype=np.uint8)
        for lab, cl in map_label.items():
            cluster_mask[mask == int(lab)] = int(str(cl).split("_")[-1])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(cluster_mask == 0, cluster_mask), cmap=cmap, vmin=0, vmax=best_k, alpha=.42, interpolation="nearest")
        ax.set_title(f"{name} ({meta['dose_nM']} nM); n={len(map_label):,}")
        ax.axis("off")
    fig.suptitle("Clusters mapped back to the original DAPI images")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_cluster_overlays.png", dpi=180)
    plt.close(fig)

    # Representative nuclei montage
    fig, axes = plt.subplots(best_k, 8, figsize=(12, 1.55 * best_k), squeeze=False)
    for ci in range(1, best_k + 1):
        lab = f"cluster_{ci}"
        candidates = df[df["cluster"] == lab].sort_values("cluster_confidence", ascending=False)
        # Spread examples across conditions while prioritizing confident assignments.
        picks = candidates.groupby("sample", group_keys=False).head(2).head(8)
        for j, (_, r) in enumerate(picks.iterrows()):
            img = normalize_image(tifffile.imread(SAMPLES[r["sample"]]["path"]))
            cy, cx = int(round(r["centroid_y"])), int(round(r["centroid_x"]))
            half = 24
            crop = np.zeros((2 * half, 2 * half), dtype=float)
            y0, y1 = max(0, cy-half), min(img.shape[0], cy+half)
            x0, x1 = max(0, cx-half), min(img.shape[1], cx+half)
            crop[:y1-y0, :x1-x0] = img[y0:y1, x0:x1]
            axes[ci-1, j].imshow(crop, cmap="gray", vmin=0, vmax=1)
            axes[ci-1, j].set_title(r["sample"], fontsize=7)
        for j in range(8):
            axes[ci-1, j].axis("off")
        axes[ci-1, 0].set_ylabel(lab, rotation=0, labelpad=33, va="center", fontsize=9)
    fig.suptitle("High-confidence examples from each morphology cluster")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_representative_nuclei.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, sel: pd.DataFrame, sample_summary: pd.DataFrame, block_summary: pd.DataFrame, best_k: int, pca: PCA) -> None:
    labels = [f"cluster_{i}" for i in range(1, best_k + 1)]
    props = pd.crosstab(df["dose_nM"], df["cluster"], normalize="index").reindex([0, 1, 5, 10]).fillna(0)
    props = props.reindex(columns=labels, fill_value=0)
    dominant = {int(dose): str(row.idxmax()) for dose, row in props.iterrows()}
    largest_change = None
    if len(props) >= 2:
        delta = props.loc[10] - props.loc[0]
        c = str(delta.abs().idxmax())
        largest_change = (c, float(delta[c]))
    spatial = block_summary.groupby("sample")["js_distance_to_image"].mean().sort_values()
    profiles = cluster_profiles(df)
    profile_text = []
    for cl, row in profiles.iterrows():
        hi = row.sort_values(ascending=False).head(2).index.tolist()
        lo = row.sort_values().head(2).index.tolist()
        profile_text.append(f"- {cl}: relatively high {', '.join(hi)}; relatively low {', '.join(lo)}.")

    lines = [
        "# D0 nuclear morphology analysis",
        "",
        "## Question",
        "",
        "Can nuclei from the four D0 HA conditions be represented in one shared morphology space, and do the samples differ in morphology composition or within-image heterogeneity?",
        "",
        "## Data and workflow",
        "",
        "- Inputs: four published 2048×2048, 16-bit DAPI images: HA 0, 1, 5 and 10 nM.",
        "- Unit of analysis: one QC-passed nucleus becomes one geometry feature vector.",
        "- All four samples were pooled for a single balanced joint model; no sample was clustered separately.",
        "- Primary clusters use geometry only (area, circularity, eccentricity, solidity, extent, aspect ratio and normalized perimeter). Intensity is retained only for description.",
        f"- Model selection compared k=2–6 by BIC, silhouette and repeat-seed stability; k={best_k} was selected.",
        f"- PCA retained {len(pca.explained_variance_ratio_)} components and {100*pca.explained_variance_ratio_.sum():.1f}% of scaled feature variance.",
        "",
        "## Main descriptive results",
        "",
        f"- QC-passed nuclei: {len(df):,} total (" + ", ".join(f"{r.sample}: {int(r.n_nuclei):,}" for r in sample_summary.itertuples()) + ").",
        "- Dominant cluster by condition: " + ", ".join(f"{dose} nM → {dominant[dose]}" for dose in [0, 1, 5, 10]) + ".",
    ]
    if largest_change:
        direction = "increased" if largest_change[1] > 0 else "decreased"
        lines.append(f"- The largest 0→10 nM composition change was {largest_change[0]} ({direction} by {abs(largest_change[1])*100:.1f} percentage points).")
    lines += [
        f"- Lowest mean spatial block-to-image divergence: {spatial.index[0]}; highest: {spatial.index[-1]}. This is a within-image uniformity check, not a biological replicate test.",
        "",
        "## Neutral cluster descriptions",
        "",
        *profile_text,
        "",
        "## Interpretation boundary",
        "",
        "This dataset supports a descriptive image-analysis result: the four D0 images occupy a shared nuclear morphology space but differ in the proportions and spatial distribution of neutral morphology clusters. Because there is one image per condition here, nuclei and 4×4 image blocks are not independent biological replicates. Therefore no condition-level p-value or causal biological claim is reported.",
        "",
        "## Files to inspect first",
        "",
        "1. `figures/06_cluster_overlays.png` — segmentation/clusters returned to the source images.",
        "2. `figures/04_cluster_fractions.png` — direct comparison of morphology composition.",
        "3. `figures/05_cluster_profiles.png` and `figures/07_representative_nuclei.png` — what the clusters mean visually.",
        "4. `tables/sample_summary.csv` — sample-level outputs.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    for name, meta in SAMPLES.items():
        if not meta["path"].exists():
            raise FileNotFoundError(meta["path"])
    segment_all()
    df = extract_features()
    df, sel, scaler, pca, model, fit_idx, best_k = fit_joint_model(df)
    sample_summary, block_summary = summarize_samples(df, best_k)

    df.to_csv(TABLE_DIR / "nucleus_features_clustered.csv", index=False)
    sel.to_csv(TABLE_DIR / "model_selection.csv", index=False)
    sample_summary.to_csv(TABLE_DIR / "sample_summary.csv", index=False)
    block_summary.to_csv(TABLE_DIR / "spatial_block_summary.csv", index=False)
    cluster_profiles(df).to_csv(TABLE_DIR / "cluster_profiles_robust_z.csv")
    pd.crosstab(df["dose_nM"], df["cluster"], normalize="index").to_csv(TABLE_DIR / "cluster_proportions.csv")

    save_figures(df, sel, best_k)
    write_report(df, sel, sample_summary, block_summary, best_k, pca)
    run_info = {
        "selected_k": best_k,
        "n_nuclei": len(df),
        "fit_nuclei": len(fit_idx),
        "model_features": MODEL_FEATURES,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "random_seed": RANDOM_SEED,
        "input_files": {k: str(v["path"]) for k, v in SAMPLES.items()},
    }
    (OUT / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"[done] selected k={best_k}; nuclei={len(df)}; output={OUT}")


if __name__ == "__main__":
    main()
