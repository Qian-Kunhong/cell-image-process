"""05b 修正森林图（菱形中位数）+ phenotype×YAP 条件级散点。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rp.paths import data_path, legacy_path  # noqa: E402

YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
COND_CSV = YAP_ROOT_3D / "analysis_summary" / "condition_3d_summary.csv"
YAP_ROOT_2D = data_path("03_Raw data", "Immunostaining", "YAP quantification")
OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin"

# --- 加载数据 ---
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

# GMM 3 簇
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
X = morph[FEATURES].values
X = np.nan_to_num(X, nan=0.0)
Xs = StandardScaler().fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type="full")
morph["cluster"] = gmm.fit_predict(Xs)

# 簇按编号中性命名：GMM 是无监督结果，不给"小/大/长"这类生物学解释
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

# --- 图 3 修正：森林图（菱形中位数） ---
features_to_plot = [
    ("equivalent_sphere_diameter_um", "Nuclear diameter"),
    ("volume_um3", "Nuclear volume"),
    ("pca_elongation_1_over_3", "Elongation"),
    ("height_um", "Height (z)"),
    ("max_xy_area_um2", "Max XY area"),
]

day0 = morph[morph["paper_day"] == 0]
ctrl = day0[day0["regimen_cn"] == "不暴露"]

fig, ax = plt.subplots(figsize=(9, 4.5))
y_pos = np.arange(len(features_to_plot))

for i, (col, label) in enumerate(features_to_plot):
    ctrl_median = ctrl[col].median()
    ctrl_q25 = ctrl[col].quantile(0.25)
    ctrl_q75 = ctrl[col].quantile(0.75)
    ctrl_iqr = ctrl_q75 - ctrl_q25
    if ctrl_iqr == 0:
        continue

    for reg, color, offset in [("单次", "#C44E52", 0.15), ("每日", "#8172B3", -0.15)]:
        ha = day0[day0["regimen_cn"] == reg]
        ha_med = ha[col].median()
        ha_q25 = ha[col].quantile(0.25)
        ha_q75 = ha[col].quantile(0.75)

        # 效应量 = (ha_med - ctrl_med) / ctrl_iqr
        eff_med = (ha_med - ctrl_median) / ctrl_iqr
        eff_low = (ha_q25 - ctrl_median) / ctrl_iqr
        eff_high = (ha_q75 - ctrl_median) / ctrl_iqr

        y = i + offset
        # 水平线 = IQR 范围
        ax.plot([eff_low, eff_high], [y, y], color=color, lw=1.5, alpha=0.6)
        # 菱形 = 中位数
        ax.scatter(eff_med, y, marker="D", s=80, color=color, zorder=5, edgecolors="white", linewidths=0.5)

ax.axvline(x=0, color="black", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f[1] for f in features_to_plot])
ax.set_xlabel("HA effect relative to no-HA control (IQR units, day 0)")
ax.set_title("HA effect on nuclear morphology (day 0)")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#C44E52', markersize=9, label='Single HA (day -3)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#8172B3', markersize=9, label='Repeated HA (day -3→0)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "ha_effect_forest.png", dpi=180, bbox_inches="tight")
plt.close()

# --- 图 5：cluster × YAP（条件级） ---
# 每条件算：三个无监督簇的比例 + YAP 中位数
cond_pheno = morph.groupby(["paper_day", "regimen_cn", "cluster"]).size().unstack(fill_value=0)
cond_pheno = cond_pheno.div(cond_pheno.sum(axis=1), axis=0)
cond_pheno.columns = [cluster_names.get(c, f"Cluster {c + 1}") for c in cond_pheno.columns]
cond_pheno = cond_pheno.reset_index()

cond_yap = yap.groupby(["paper_day", "regimen_cn"])["Math_NucByRingYAP"].median().reset_index()
cond_yap = cond_yap.rename(columns={"Math_NucByRingYAP": "yap_median"})

cond_df = cond_pheno.merge(cond_yap, on=["paper_day", "regimen_cn"], how="inner")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
pheno_cols = [c for c in cond_df.columns if c.startswith("P")]
pheno_colors = {cluster_names[c]: cluster_colors[c] for c in range(3)}

for ax, pheno_col in zip(axes, pheno_cols):
    for _, row in cond_df.iterrows():
        day = row["paper_day"]
        reg = row["regimen_cn"]
        if reg == "基线":
            marker = "s"
            size = 120
        elif reg == "不暴露":
            marker = "o"
            size = 80
        elif reg == "单次":
            marker = "^"
            size = 80
        else:
            marker = "D"
            size = 80
        color = "#999999" if reg == "基线" else "#4C72B0" if reg == "不暴露" else "#DD8452" if reg == "单次" else "#C44E52"
        ax.scatter(row[pheno_col], row["yap_median"], marker=marker, s=size,
                   color=color, edgecolors="black", linewidths=0.5, zorder=3)
        # 标注
        ax.annotate(f"d{int(day)}", (row[pheno_col], row["yap_median"]),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel(f"{pheno_col} fraction")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax.set_title(f"YAP vs {pheno_col}", fontsize=10)

axes[0].set_ylabel("YAP N/C ratio (median per condition)")

# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#999999', markersize=9, label='Baseline'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0', markersize=9, label='No HA'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#DD8452', markersize=9, label='Single HA'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#C44E52', markersize=9, label='Repeated HA'),
]
fig.legend(handles=legend_elements, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("YAP N/C ratio vs nuclear morphology cluster composition (condition-level)", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "yap_by_phenotype.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"Done. Figures saved to {OUT_DIR}", file=sys.stderr)
