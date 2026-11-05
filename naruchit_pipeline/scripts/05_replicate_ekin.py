"""05 复刻 Ekin 分析范式：UMAP 聚类 + 簇比例 + 效应量 + YAP 箱线图。

输入：
- 973 核 3D 形态表（旧仓 nucleus_3d_metrics.csv）
- 2821 核 2D YAP 表（CellProfiler 输出）

输出：
- UMAP_3clusters.png
- cluster_proportions.png
- ha_effect_forest.png
- yap_boxplot.png
"""
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
import umap

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rp.paths import data_path, legacy_path  # noqa: E402

YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
COND_CSV = YAP_ROOT_3D / "analysis_summary" / "condition_3d_summary.csv"
YAP_ROOT_2D = data_path("03_Raw data", "Immunostaining", "YAP quantification")
OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. 数据加载 ---
cond = pd.read_csv(COND_CSV, encoding="utf-8-sig")
cond = cond[["sample", "culture_day", "paper_day", "regimen"]].copy()
regimen_cn = {"shared_baseline": "基线", "no_HA": "不暴露", "single_HA": "单次", "repeated_HA": "每日"}
cond["regimen_cn"] = cond["regimen"].map(regimen_cn)

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
print(f"morph: {len(morph)}", file=sys.stderr)

# --- 2. GMM 3 簇 + UMAP ---
X = morph[FEATURES].values
X = np.nan_to_num(X, nan=0.0)
Xs = StandardScaler().fit_transform(X)

gmm = GaussianMixture(n_components=3, random_state=42, covariance_type="full")
morph["cluster"] = gmm.fit_predict(Xs)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
emb = reducer.fit_transform(Xs)
morph["umap1"] = emb[:, 0]
morph["umap2"] = emb[:, 1]

# 给簇命名：看每个簇的中位数
cluster_summary = morph.groupby("cluster")[FEATURES].median()
print("Cluster centroids (median):", file=sys.stderr)
print(cluster_summary.to_string(), file=sys.stderr)

# 簇按编号中性命名：GMM 是无监督结果，不给"小/大/长"这类生物学解释
cluster_means = morph.groupby("cluster")[["volume_um3", "pca_elongation_1_over_3"]].mean()
print("\nCluster means:", file=sys.stderr)
print(cluster_means.to_string(), file=sys.stderr)

PALETTE = ["#3870B8", "#E69E00", "#4DB04D"]
cluster_names = {c: f"Cluster {c + 1}" for c in range(3)}
cluster_colors = {c: PALETTE[c] for c in range(3)}

print("\nCluster names:", cluster_names, file=sys.stderr)

# --- 图 1：UMAP ---
fig, ax = plt.subplots(figsize=(7, 6))
for c in range(3):
    sub = morph[morph["cluster"] == c]
    ax.scatter(sub["umap1"], sub["umap2"], s=8, alpha=0.6,
               label=cluster_names[c], color=cluster_colors[c])
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP embedding of single-nucleus morphology (973 nuclei, 10 conditions)")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "umap_3clusters.png", dpi=180, bbox_inches="tight")
plt.close()

# --- 图 2：簇比例堆叠柱 ---
# 按 day × regimen 分组
pivot = morph.groupby(["paper_day", "regimen_cn", "cluster"]).size().unstack(fill_value=0)
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
ax.set_xticklabels([f"{idx[1]}\nday {int(idx[0])}" for idx in pivot.index], fontsize=9)
ax.set_ylabel("Cell fraction")
ax.set_title("Nuclear morphology cluster proportions by condition")
ax.legend(fontsize=9, loc="upper right")
ax.axvline(x=0.5, color="gray", ls="--", alpha=0.4)
ax.axvline(x=3.5, color="gray", ls="--", alpha=0.4)
ax.axvline(x=6.5, color="gray", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "cluster_proportions.png", dpi=180, bbox_inches="tight")
plt.close()

# --- 图 3：效应量森林图 ---
# 对每个形态特征，算 HA（单次/每日）vs 不暴露 的差异，用 IQR 标准化
features_to_plot = [
    ("equivalent_sphere_diameter_um", "Nuclear diameter"),
    ("volume_um3", "Nuclear volume"),
    ("pca_elongation_1_over_3", "Elongation"),
    ("height_um", "Height (z)"),
    ("max_xy_area_um2", "Max XY area"),
]

# 只看 day 0（分化起点）
day0 = morph[morph["paper_day"] == 0]
ctrl = day0[day0["regimen_cn"] == "不暴露"]

effects = []
for col, label in features_to_plot:
    ctrl_median = ctrl[col].median()
    ctrl_iqr = ctrl[col].quantile(0.75) - ctrl[col].quantile(0.25)
    if ctrl_iqr == 0:
        continue
    for reg in ["单次", "每日"]:
        ha = day0[day0["regimen_cn"] == reg]
        effect = (ha[col].median() - ctrl_median) / ctrl_iqr
        effects.append({"feature": label, "regimen": reg, "effect": effect})

eff_df = pd.DataFrame(effects)
fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(features_to_plot))
for i, (col, label) in enumerate(features_to_plot):
    sub = eff_df[eff_df["feature"] == label]
    for _, row in sub.iterrows():
        color = "#DD8452" if row["regimen"] == "单次" else "#C44E52"
        offset = 0.15 if row["regimen"] == "单次" else -0.15
        ax.scatter(row["effect"], i + offset, color=color, s=60, zorder=3)
        ax.plot([0, row["effect"]], [i + offset, i + offset], color=color, alpha=0.5, lw=2)
ax.axvline(x=0, color="black", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f[1] for f in features_to_plot])
ax.set_xlabel("HA effect relative to no-HA control (IQR units, day 0)")
ax.set_title("HA effect on nuclear morphology (day 0)")
# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#DD8452', markersize=8, label='Single HA'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#C44E52', markersize=8, label='Repeated HA'),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "ha_effect_forest.png", dpi=180, bbox_inches="tight")
plt.close()

# --- 图 4：YAP N/C 箱线图（按条件） ---
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

fig, ax = plt.subplots(figsize=(10, 5))
order_day = [-3, -2, -1, 0]
order_regimen = ["基线", "不暴露", "单次", "每日"]
colors_reg = {"基线": "#999999", "不暴露": "#4C72B0", "单次": "#DD8452", "每日": "#C44E52"}

positions = []
box_data = []
box_colors = []
xtick_labels = []
pos = 0
for day in order_day:
    for reg in order_regimen:
        sub = yap[(yap["paper_day"] == day) & (yap["regimen_cn"] == reg)]["Math_NucByRingYAP"].dropna()
        if len(sub) == 0:
            pos += 1
            continue
        positions.append(pos)
        box_data.append(sub.values)
        box_colors.append(colors_reg[reg])
        xtick_labels.append(f"{reg}\nday{day}")
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
fig.savefig(OUT_DIR / "yap_boxplot.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to: {OUT_DIR}", file=sys.stderr)
