"""09 用已有 2D mask 绘制三面板图。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import exposure, measure, segmentation
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

# 读最佳焦平面 z 索引
focus = pd.read_csv(FOCUS_CSV)
best_z = focus[(focus["sample"] == SAMPLE) & (focus["selected"] == True)]["z_index"].values[0]
print(f"Best z-slice: {best_z}", file=sys.stderr)

# 读 OIB DAPI z 栈
with oiffile.OifFile(OIB_PATH) as handle:
    stack = handle.asarray()
    axes = handle.axes
if stack.ndim == 4 and stack.shape[0] <= 4 and stack.shape[1] > 4:
    stack = np.transpose(stack, (1, 0, 2, 3))
dapi = stack[:, 0].astype(np.float32)
print(f"DAPI z-stack: {dapi.shape}", file=sys.stderr)

# 最佳焦平面
dapi_best = dapi[best_z]
nonzero = dapi_best[dapi_best > 0]
lo, hi = np.percentile(nonzero, (0.5, 99.7))
dapi_display = exposure.rescale_intensity(dapi_best, in_range=(lo, hi), out_range=(0, 1))

# 读已有 2D mask
mask_data = np.load(MASK_PATH)
labels = mask_data["labels"]
print(f"2D mask labels: {len(np.unique(labels)) - 1}", file=sys.stderr)

# 提取 2D 特征
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

# === 三面板 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# 1. DAPI input
axes[0].imshow(dapi_display, cmap="gray")
axes[0].set_title("1  DAPI input", fontsize=14, fontweight="bold", loc="left")
axes[0].axis("off")

# 2. Nuclear segmentation
axes[1].imshow(dapi_display, cmap="gray")
boundary = segmentation.mark_boundaries(dapi_display, labels, color=(1, 0, 0), mode="thick")
axes[1].imshow(boundary)
axes[1].set_title("2  Nuclear segmentation", fontsize=14, fontweight="bold", loc="left")
axes[1].axis("off")

# 3. GMM clusters
color_map = {0: (0.22, 0.44, 0.72, 1.0),
             1: (0.90, 0.62, 0.0, 1.0),
             2: (0.30, 0.68, 0.30, 1.0)}
color_mask = np.zeros((*labels.shape, 4))
for row in features.itertuples():
    color_mask[labels == row.label] = color_map[row.cluster]

axes[2].imshow(np.zeros_like(dapi_display), cmap="gray", vmin=0, vmax=1)
axes[2].imshow(color_mask)
axes[2].set_title("3  GMM clusters 1–3", fontsize=14, fontweight="bold", loc="left")
axes[2].axis("off")

from matplotlib.patches import Patch
legend = [
    Patch(facecolor="#3870B8", label="Cluster 1"),
    Patch(facecolor="#E69E00", label="Cluster 2"),
    Patch(facecolor="#4DB04D", label="Cluster 3"),
]
axes[2].legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, fontsize=9, frameon=False)

fig.tight_layout()
out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "segmentation_2d_overlay.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}", file=sys.stderr)
