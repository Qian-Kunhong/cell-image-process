"""12 表型分割网格图：行=条件，列=时间点，DAPI 底图 + GMM 上色。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure, segmentation
from skimage.segmentation import find_boundaries
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rp.paths import data_path, legacy_path  # noqa: E402

DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))
import oiffile

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

RAW_DIR = data_path("03_Raw data", "Immunostaining", "Confocal - YAP")
BEST_FOCUS_DIR = legacy_path(
    "analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap",
    "best_focus_2d_vs_3d",
)
MASK_DIR = BEST_FOCUS_DIR / "masks_2d"
FOCUS_CSV = BEST_FOCUS_DIR / "focus_scores_by_z.csv"

# 网格
ROWS = ["Without HA\n(control)", "With HA\n(single)", "With HA\n(repeated)"]
COLS = ["Day −2", "Day −1", "Day 0"]
SAMPLES = {
    (0, 0): "D30D", (0, 1): "D40D", (0, 2): "D50D",
    (1, 0): "D31D", (1, 1): "D41D", (1, 2): "D51D",
    (2, 0): "D33D", (2, 1): "D43D", (2, 2): "D53D",
}

# 读所有样本的特征，统一 GMM
focus = pd.read_csv(FOCUS_CSV)
all_features = []
sample_data = {}

for (i, j), sample in SAMPLES.items():
    oib_path = RAW_DIR / f"{sample}.oib"
    mask_path = MASK_DIR / f"{sample}_best_focus_2d_mask.npz"
    if not oib_path.exists() or not mask_path.exists():
        print(f"Missing: {sample}", file=sys.stderr)
        continue

    best_z = focus[(focus["sample"] == sample) & (focus["selected"] == True)]["z_index"].values[0]

    with oiffile.OifFile(oib_path) as handle:
        stack = handle.asarray()
    if stack.ndim == 4 and stack.shape[0] <= 4 and stack.shape[1] > 4:
        stack = np.transpose(stack, (1, 0, 2, 3))
    dapi_best = stack[best_z, 0].astype(np.float32)

    mask = np.load(mask_path)["labels"]

    # 提取特征
    regions = measure.regionprops(mask)
    rows = []
    for r in regions:
        rows.append({
            "sample": sample,
            "label": r.label,
            "area_um2": r.area * 0.17578**2,
            "circularity": 4 * np.pi * r.area / (r.perimeter**2) if r.perimeter > 0 else 0,
            "elongation": r.eccentricity,
            "solidity": r.solidity,
            "centroid_y": r.centroid[0],
            "centroid_x": r.centroid[1],
        })
    df = pd.DataFrame(rows)

    # 邻居特征
    coords = df[["centroid_y", "centroid_x"]].values
    for k in range(len(df)):
        d = np.sqrt(np.sum((coords - coords[k])**2, axis=1))
        d[k] = np.inf
        nn_idx = np.argsort(d)[:6]
        df.loc[k, "nn_distance_um"] = d[nn_idx[0]] * 0.17578
        df.loc[k, "neighbor_circularity_std"] = df.loc[nn_idx, "circularity"].std()

    all_features.append(df)
    sample_data[sample] = {"dapi": dapi_best, "mask": mask, "df": df}

all_df = pd.concat(all_features, ignore_index=True)

# 统一 GMM
gmm_features = all_df[["area_um2", "circularity", "elongation", "nn_distance_um",
                       "neighbor_circularity_std"]].values
gmm = GaussianMixture(n_components=3, random_state=42)
all_df["cluster"] = gmm.fit_predict(gmm_features)
cluster_area = all_df.groupby("cluster")["area_um2"].mean().sort_values()
cluster_map = {old: new for new, old in enumerate(cluster_area.index)}
all_df["cluster"] = all_df["cluster"].map(cluster_map)
print(f"Total nuclei: {len(all_df)}, clusters: {all_df['cluster'].value_counts().sort_index().to_dict()}", file=sys.stderr)

# 配色
PHENOTYPE_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
PHENOTYPE_NAMES = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3"}

# 画图
fig, axes = plt.subplots(3, 3, figsize=(12, 11))
fig.subplots_adjust(hspace=0.12, wspace=0.05, left=0.12, right=0.98, top=0.92, bottom=0.08)

# 列标题
for j, col in enumerate(COLS):
    axes[0, j].set_title(col, fontsize=14, fontweight="bold", pad=8)

# 行标题
for i, row in enumerate(ROWS):
    axes[i, 0].set_ylabel(row, fontsize=11, fontweight="bold", rotation=0, labelpad=50, va="center")

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        sample = SAMPLES.get((i, j))
        if sample not in sample_data:
            ax.axis("off")
            continue

        data = sample_data[sample]
        dapi = data["dapi"]
        mask = data["mask"]
        df = data["df"]

        # DAPI 归一化
        lo, hi = np.percentile(dapi[dapi > 0], (0.5, 99.7))
        dapi_display = np.clip((dapi - lo) / (hi - lo + 1e-6), 0, 1)

        # 画 DAPI 灰度底
        ax.imshow(dapi_display, cmap="gray", vmin=0, vmax=1)

        # 叠加表型颜色
        for _, row in df.iterrows():
            cl = all_df[(all_df["sample"] == sample) & (all_df["label"] == row.label)]["cluster"].values[0]
            mask_i = mask == row.label
            # 半透明叠加
            color = to_rgb(PHENOTYPE_COLORS[cl])
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask_i] = [color[0], color[1], color[2], 0.6]
            ax.imshow(overlay)

        # 黑色分隔线
        boundaries = find_boundaries(mask, mode="inner")
        ax.contour(boundaries, colors="black", linewidths=0.3)

        ax.set_xticks([])
        ax.set_yticks([])

# 图例
handles = [Line2D([], [], marker="o", linestyle="none", color=PHENOTYPE_COLORS[k],
                  markersize=10, label=PHENOTYPE_NAMES[k]) for k in sorted(PHENOTYPE_COLORS)]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle("DAPI cluster assignment across HA conditions",
             fontsize=16, fontweight="bold", y=0.97)

out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "cluster_grid.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}", file=sys.stderr)
