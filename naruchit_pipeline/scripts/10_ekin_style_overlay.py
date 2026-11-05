"""10 按 Ekin make_ppt_figures.py 的三面板代码复刻。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, measure, segmentation
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

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})

SAMPLE = "D50D"
OIB_PATH = data_path("03_Raw data", "Immunostaining", "Confocal - YAP", f"{SAMPLE}.oib")
BEST_FOCUS_DIR = legacy_path(
    "analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap",
    "best_focus_2d_vs_3d",
)
MASK_PATH = BEST_FOCUS_DIR / "masks_2d" / f"{SAMPLE}_best_focus_2d_mask.npz"
FOCUS_CSV = BEST_FOCUS_DIR / "focus_scores_by_z.csv"

# 读最佳焦平面
focus = pd.read_csv(FOCUS_CSV)
best_z = focus[(focus["sample"] == SAMPLE) & (focus["selected"] == True)]["z_index"].values[0]

# 读 OIB DAPI
with oiffile.OifFile(OIB_PATH) as handle:
    stack = handle.asarray()
if stack.ndim == 4 and stack.shape[0] <= 4 and stack.shape[1] > 4:
    stack = np.transpose(stack, (1, 0, 2, 3))
dapi = stack[:, 0].astype(np.float32)
dapi_best = dapi[best_z]

# 显示范围（和 Ekin 一样：百分位）
lo, hi = np.percentile(dapi_best[dapi_best > 0], (0.5, 99.7))

# 读 mask
mask_data = np.load(MASK_PATH)
labels = mask_data["labels"]
print(f"Mask: {len(np.unique(labels)) - 1} nuclei", file=sys.stderr)

# 提取特征 + GMM（和 Ekin 一样的流程）
regions = measure.regionprops(labels)
rows = []
for r in regions:
    rows.append({
        "label": r.label,
        "area_um2": r.area * 0.17578**2,
        "perimeter_um": r.perimeter * 0.17578,
        "circularity": 4 * np.pi * r.area / (r.perimeter**2) if r.perimeter > 0 else 0,
        "elongation": r.eccentricity,
        "solidity": r.solidity,
        "centroid_y": r.centroid[0],
        "centroid_x": r.centroid[1],
    })
features = pd.DataFrame(rows)

# 邻居特征
coords = features[["centroid_y", "centroid_x"]].values
for i, row in features.iterrows():
    d = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
    d[i] = np.inf
    nn_idx = np.argsort(d)[:6]
    features.loc[i, "nn_distance_um"] = d[nn_idx[0]] * 0.17578
    features.loc[i, "neighbor_circularity_std"] = features.loc[nn_idx, "circularity"].std()
    features.loc[i, "local_crowding"] = features.loc[nn_idx, "area_um2"].sum() / (np.pi * (d[nn_idx[0]] * 0.17578)**2 + 1e-6)

# GMM 3 簇
gmm_features = features[["area_um2", "circularity", "elongation", "nn_distance_um",
                         "neighbor_circularity_std", "local_crowding"]].values
gmm = GaussianMixture(n_components=3, random_state=42)
features["cluster"] = gmm.fit_predict(gmm_features)
cluster_area = features.groupby("cluster")["area_um2"].mean().sort_values()
cluster_map = {old: new for new, old in enumerate(cluster_area.index)}
features["cluster"] = features["cluster"].map(cluster_map)
print(f"Clusters: {features['cluster'].value_counts().sort_index().to_dict()}", file=sys.stderr)

# Ekin 配色；标签只表示无监督簇。
PHENOTYPE_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
PHENOTYPE_NAMES = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3"}

# === 严格按 Ekin 代码画三面板 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# Panel 1: DAPI input
axes[0].imshow(dapi_best, cmap="gray", vmin=lo, vmax=hi)
axes[0].set_title("1\nDAPI input", fontsize=16, fontweight="bold", pad=8)

# Panel 2: Nuclear segmentation
axes[1].imshow(dapi_best, cmap="gray", vmin=lo, vmax=hi)
boundaries = find_boundaries(labels, mode="outer")
axes[1].imshow(np.ma.masked_where(~boundaries, boundaries), cmap="autumn", alpha=.9)
axes[1].set_title("2\nNuclear segmentation", fontsize=16, fontweight="bold", pad=8)

# Panel 3: GMM phenotype
phenotype_canvas = np.zeros((*labels.shape, 3), dtype=float)
for row in features.itertuples():
    phenotype_canvas[labels == int(row.label)] = to_rgb(PHENOTYPE_COLORS[row.cluster])
# 内部黑色分隔线（Ekin 加的）
phenotype_canvas[find_boundaries(labels, mode="inner")] = 0.0
axes[2].imshow(phenotype_canvas)
axes[2].set_title("3\nGMM clusters 1–3", fontsize=16, fontweight="bold", pad=8)

for ax in axes:
    ax.axis("off")

handles = [Line2D([], [], marker="o", linestyle="none", color=PHENOTYPE_COLORS[k],
                  label=PHENOTYPE_NAMES[k]) for k in sorted(PHENOTYPE_COLORS)]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11)

fig.suptitle("D50D (No HA, Day 0): DAPI cluster assignment",
             fontsize=18, fontweight="bold", y=1.01)
fig.subplots_adjust(left=.025, right=.985, top=.84, bottom=.15, wspace=.035)

out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "ekin_style_overlay.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}", file=sys.stderr)
