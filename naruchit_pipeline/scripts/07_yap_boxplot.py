"""07 YAP 箱线图：日期颜色和森林图对应。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rp.paths import legacy_path  # noqa: E402

df = pd.read_csv(
    legacy_path(
        "analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap",
        "analysis_summary", "nucleus_3d_yap_cell_metrics.csv",
    ),
    encoding="utf-8-sig",
)
df = df[df["qc_pass_uniform"] == True].copy()
df["log2_ratio"] = np.log2(df["yap_nuclear_shell_ratio"].replace(0, np.nan))

# 日期颜色（和森林图统一）
DAY_COLORS = {
    -3: "#BBBBBB",   # baseline
    -2: "#2166AC",   # Day -2 深蓝
    -1: "#4DAF4A",   # Day -1 绿
     0: "#E8985E",   # Day 0 橙（和 Single 条件色同色系但稍浅）
}

# 条件顺序：(sample, day, regimen)
order = ["D20D", "D30D", "D31D", "D33D", "D40D", "D41D", "D43D", "D50D", "D51D", "D53D"]
days = [-3, -2, -2, -2, -1, -1, -1, 0, 0, 0]
regimens = ["Baseline", "No HA", "Single", "Repeated", "No HA", "Single", "Repeated",
            "No HA", "Single", "Repeated"]

# 箱线图填充色：按日期浅色 + 条件区分深浅
def box_color(day, regimen):
    base = DAY_COLORS[day]
    # 同色不同透明度：No HA 最浅，Single 中，Repeated 深
    if regimen == "Baseline":
        return "#CCCCCC"
    if regimen == "No HA":
        return base + "40"  # 25% opacity
    if regimen == "Single":
        return base + "99"  # 60% opacity
    return base  # 100% opacity

box_colors = [box_color(d, r) for d, r in zip(days, regimens)]

labels = [
    "Baseline\nDay −3",
    "No HA\nDay −2", "Single\nDay −2", "Repeated\nDay −2",
    "No HA\nDay −1", "Single\nDay −1", "Repeated\nDay −1",
    "No HA\nDay 0", "Single\nDay 0", "Repeated\nDay 0",
]

fig, ax = plt.subplots(figsize=(14, 5))
fig.subplots_adjust(bottom=0.18)

box_data = [df[df["sample"] == s]["log2_ratio"].dropna().values for s in order]

bp = ax.boxplot(box_data, patch_artist=True, widths=0.55,
                showfliers=True,
                flierprops=dict(marker="o", markerfacecolor="none",
                                markeredgecolor="#BBBBBB", markersize=4, alpha=0.6),
                medianprops=dict(color="#333333", linewidth=1.5),
                whiskerprops=dict(color="#555555", linewidth=1.2),
                capprops=dict(color="#555555", linewidth=1.2))

for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_edgecolor("#555555")
    patch.set_linewidth(1.2)

# 日期分组背景色带
day_groups = {
    -3: (1, 1),
    -2: (2, 4),
    -1: (5, 7),
     0: (8, 10),
}
for day, (start, end) in day_groups.items():
    ax.axvspan(start - 0.5, end + 0.5, color=DAY_COLORS[day], alpha=0.06, zorder=0)

ax.set_xticks(range(1, len(order) + 1), labels, fontsize=10)
ax.set_ylabel("log2 YAP nuclear / shell ratio", fontsize=12)
ax.set_title("YAP nuclear enrichment by condition", fontsize=15, fontweight="bold", pad=12)
ax.set_ylim(-1, 4)
ax.grid(axis="y", color="#EEEEEE", lw=1)
ax.tick_params(axis="y", labelsize=10)

# n= 标在箱线上方
for i, s in enumerate(order):
    n = len(df[df["sample"] == s]["log2_ratio"].dropna())
    top = np.percentile(box_data[i], 75)
    ax.text(i + 1, top + 0.35, f"n={n}", ha="center", fontsize=8, color="#888888")

fig.tight_layout()
out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "yap_boxplot.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved: {out}", file=sys.stderr)
