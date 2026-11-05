"""05c 英文标注版：簇比例 + YAP 箱线图。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rp.paths import data_path, legacy_path  # noqa: E402

YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
COND_CSV = YAP_ROOT_3D / "analysis_summary" / "condition_3d_summary.csv"
YAP_ROOT_2D = data_path("03_Raw data", "Immunostaining", "YAP quantification")
OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin"

# --- 数据加载 ---
cond = pd.read_csv(COND_CSV, encoding="utf-8-sig")
cond = cond[["sample", "culture_day", "paper_day", "regimen"]].copy()
regimen_en = {"shared_baseline": "Baseline", "no_HA": "No HA", "single_HA": "Single", "repeated_HA": "Repeated"}
cond["regimen_en"] = cond["regimen"].map(regimen_en)

FEATURES = [
    "volume_um3", "height_um",
    "bbox_y_um", "bbox_x_um",
    "pca_axis_1_um", "pca_axis_2_um", "pca_axis_3_um",
    "pca_elongation_1_over_3",
    "max_xy_area_um2",
    "equivalent_sphere_diameter_um",
]
frames = []
for sample in cond["sample"]:
    f = YAP_ROOT_3D / sample / "nucleus_3d_metrics.csv"
    if not f.exists():
        continue
    frames.append(pd.read_csv(f, encoding="utf-8-sig"))
morph = pd.concat(frames, ignore_index=True).merge(cond, on="sample", how="left")

X = morph[FEATURES].values
X = np.nan_to_num(X, nan=0.0)
Xs = StandardScaler().fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type="full")
morph["cluster"] = gmm.fit_predict(Xs)

# 簇按编号中性命名：GMM 是无监督结果，不给尺寸/形状类生物学解释
PALETTE = ["#3870B8", "#E69E00", "#4DB04D"]
cluster_names = {c: f"Cluster {c + 1}" for c in range(3)}
cluster_colors = {c: PALETTE[c] for c in range(3)}

# --- YAP 2D ---
yap_frames = []
for sample in cond["sample"]:
    sample_dir = YAP_ROOT_2D / sample
    if not sample_dir.exists():
        continue
    ana = list(sample_dir.glob(f"Ana_*/{sample}_Nuclei.csv"))
    if not ana:
        continue
    df = pd.read_csv(ana[0], encoding="utf-8-sig")
    df["sample"] = sample
    yap_frames.append(df[["sample", "Math_NucByRingYAP"]])
yap = pd.concat(yap_frames, ignore_index=True).merge(cond, on="sample", how="left")

# --- 图 2：簇比例堆叠柱（英文） ---
pivot = morph.groupby(["paper_day", "regimen_en", "cluster"]).size().unstack(fill_value=0)
pivot = pivot.div(pivot.sum(axis=1), axis=0)
pivot = pivot.sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(pivot))
for c in range(3):
    if c in pivot.columns:
        ax.bar(range(len(pivot)), pivot[c].values, bottom=bottom,
               color=cluster_colors[c], label=cluster_names[c])
        bottom += pivot[c].values
ax.set_xticks(range(len(pivot)))
ax.set_xticklabels([f"{idx[1]}\nDay {int(idx[0])}" for idx in pivot.index], fontsize=9)
ax.set_ylabel("Cell fraction")
ax.set_title("Nuclear morphology cluster proportions by condition")
ax.legend(fontsize=9, loc="upper right")
ax.axvline(x=0.5, color="gray", ls="--", alpha=0.4)
ax.axvline(x=3.5, color="gray", ls="--", alpha=0.4)
ax.axvline(x=6.5, color="gray", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "cluster_proportions_en.png", dpi=180, bbox_inches="tight")
plt.close()

# --- 图 4：YAP 箱线图（英文） ---
fig, ax = plt.subplots(figsize=(10, 5))
order_day = [-3, -2, -1, 0]
order_regimen = ["Baseline", "No HA", "Single", "Repeated"]
colors_reg = {"Baseline": "#999999", "No HA": "#4C72B0", "Single": "#DD8452", "Repeated": "#C44E52"}

positions = []
box_data = []
box_colors = []
xtick_labels = []
pos = 0
for day in order_day:
    for reg in order_regimen:
        sub = yap[(yap["paper_day"] == day) & (yap["regimen_en"] == reg)]["Math_NucByRingYAP"].dropna()
        if len(sub) == 0:
            pos += 1
            continue
        positions.append(pos)
        box_data.append(sub.values)
        box_colors.append(colors_reg[reg])
        xtick_labels.append(f"{reg}\nDay {day}")
        pos += 1
    pos += 0.5

bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for k in ("whiskers", "caps", "medians"):
    for line in bp[k]:
        line.set_color("black")
ax.set_xticks(positions)
ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45, ha="right")
ax.set_ylabel("log2 YAP nuclear/perinuclear ratio")
ax.set_title("YAP N/C ratio by condition (CellProfiler 2D slices)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "yap_boxplot_en.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"Done. English figures saved to {OUT_DIR}", file=sys.stderr)
