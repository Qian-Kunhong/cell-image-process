"""08 用现有 Cellpose 结果绘制三面板：DAPI → 红轮廓 → GMM 上色。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure, measure, segmentation

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
MASK_PATH = legacy_path(
    "analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap", SAMPLE,
    "dapi_3d_masks.npz",
)
METRICS_PATH = legacy_path(
    "analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap", SAMPLE,
    "nucleus_3d_metrics.csv",
)

# 读 DAPI z 栈
with oiffile.OifFile(OIB_PATH) as handle:
    stack = handle.asarray()
    axes = handle.axes
print(f"OIB shape: {stack.shape}, axes: {axes}", file=sys.stderr)

# 转成 ZCYX（原代码的 ensure_zcyx 逻辑）
if stack.ndim == 4 and stack.shape[0] <= 4 and stack.shape[1] > 4:
    stack = np.transpose(stack, (1, 0, 2, 3))
print(f"Transposed ZCYX shape: {stack.shape}", file=sys.stderr)
dapi = stack[:, 0].astype(np.float32)

print(f"DAPI z-stack shape: {dapi.shape}", file=sys.stderr)

# XY 最大投影
dapi_xy = dapi.max(axis=0)
# 对比度增强
nonzero = dapi_xy[dapi_xy > 0]
lo, hi = np.percentile(nonzero, (0.5, 99.7))
dapi_display = exposure.rescale_intensity(dapi_xy, in_range=(lo, hi), out_range=(0, 1))

# 读 mask（已过滤的 complete nuclei）
mask_data = np.load(MASK_PATH)
labels_3d = mask_data["labels"]
# 最大投影保留 label ID（不是二值）
labels_2d = labels_3d.max(axis=0)
print(f"2D projected labels: {len(np.unique(labels_2d)) - 1}", file=sys.stderr)

# 读 metrics 做 GMM
metrics = pd.read_csv(METRICS_PATH, encoding="utf-8-sig")
print(f"Metrics: {len(metrics)} nuclei", file=sys.stderr)

# 简单 GMM：用 volume + elongation 两特征分 3 簇
from sklearn.mixture import GaussianMixture
features = metrics[["volume_um3", "pca_elongation_1_over_3", "max_xy_area_um2"]].values
gmm = GaussianMixture(n_components=3, random_state=42)
metrics["cluster"] = gmm.fit_predict(features)
# 按 volume 排序以稳定显示编号，不赋予细胞类型含义。
cluster_vol = metrics.groupby("cluster")["volume_um3"].mean().sort_values()
cluster_map = {old: new for new, old in enumerate(cluster_vol.index)}
metrics["cluster"] = metrics["cluster"].map(cluster_map)
print(f"Cluster sizes: {metrics['cluster'].value_counts().sort_index().to_dict()}", file=sys.stderr)

# 给 2D 投影的每个核上色（用 3D metrics 的 label 对应）
# labels_2d_labeled 是 2D 投影重新标记的，和 3D label 不直接对应
# 用 centroid 位置匹配
# 简单方法：按 3D mask 的 max projection 直接上色
color_mask = np.zeros((*labels_3d.shape[1:], 4))
for row in metrics.itertuples():
    lab = row.label
    mask_i = labels_3d == lab
    # XY 投影
    mask_xy = mask_i.any(axis=0)
    cat = row.cluster
    colors = {0: (0.22, 0.44, 0.72, 1.0),
              1: (0.90, 0.62, 0.0, 1.0),
              2: (0.30, 0.68, 0.30, 1.0)}
    color_mask[mask_xy] = colors[cat]

# === 三面板 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# 1. DAPI input
axes[0].imshow(dapi_display, cmap="gray")
axes[0].set_title("1  DAPI input", fontsize=14, fontweight="bold", loc="left")
axes[0].axis("off")

# 2. Nuclear segmentation（红轮廓）
axes[1].imshow(dapi_display, cmap="gray")
boundary_overlay = segmentation.mark_boundaries(dapi_display, labels_2d,
                                                color=(1, 0, 0), mode="thick")
axes[1].imshow(boundary_overlay)
axes[1].set_title("2  Nuclear segmentation", fontsize=14, fontweight="bold", loc="left")
axes[1].axis("off")

# 3. GMM clusters
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
out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "segmentation_3d_overlay.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}", file=sys.stderr)
