"""11 YAP/DAPI 叠加网格图：行=条件，列=时间点，XY+XZ。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rp.paths import data_path  # noqa: E402

DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))
import oiffile

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

RAW_DIR = data_path("03_Raw data", "Immunostaining", "Confocal - YAP")

# 网格：行=条件，列=时间点
ROWS = ["Without HA\n(control)", "With HA\n(single)", "With HA\n(repeated)"]
COLS = ["-3D", "-2D", "-1D", "0D"]
# 样本对应
SAMPLES = {
    (0, 1): "D30D", (0, 2): "D40D", (0, 3): "D50D",   # No HA
    (1, 1): "D31D", (1, 2): "D41D", (1, 3): "D51D",   # Single
    (2, 1): "D33D", (2, 2): "D43D", (2, 3): "D53D",   # Repeated
}

def load_xy_xz(oib_path):
    """读 OIB，返回 DAPI/YAP 的 XY 和 XZ 投影。"""
    with oiffile.OifFile(oib_path) as handle:
        stack = handle.asarray()
    if stack.ndim == 4 and stack.shape[0] <= 4 and stack.shape[1] > 4:
        stack = np.transpose(stack, (1, 0, 2, 3))
    # stack shape: (Z, C, Y, X)
    dapi = stack[:, 0]  # (Z, Y, X)
    yap = stack[:, 1]   # (Z, Y, X)

    # XY 最大投影
    dapi_xy = dapi.max(axis=0)
    yap_xy = yap.max(axis=0)

    # XZ 最大投影（取中间列附近）
    mid_y = dapi.shape[1] // 2
    strip = 5
    dapi_xz = dapi[:, mid_y-strip:mid_y+strip, :].max(axis=1)  # (Z, X)
    yap_xz = yap[:, mid_y-strip:mid_y+strip, :].max(axis=1)

    # 归一化
    def norm(img):
        lo, hi = np.percentile(img[img > 0], (1, 99))
        return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    return norm(dapi_xy), norm(yap_xy), norm(dapi_xz), norm(yap_xz)

# 画图
fig, axes = plt.subplots(3, 4, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [3, 3, 3], "hspace": 0.15, "wspace": 0.05})

# 列标题
for j, col in enumerate(COLS):
    axes[0, j].set_title(col, fontsize=14, fontweight="bold", pad=8)

# 行标题
for i, row in enumerate(ROWS):
    axes[i, 0].set_ylabel(row, fontsize=12, fontweight="bold", rotation=0, labelpad=60, va="center")

# 填充每个格
for i in range(3):
    for j in range(4):
        ax = axes[i, j]
        if (i, j) not in SAMPLES:
            ax.axis("off")
            continue

        sample = SAMPLES[(i, j)]
        oib_path = RAW_DIR / f"{sample}.oib"
        if not oib_path.exists():
            print(f"Missing: {oib_path}", file=sys.stderr)
            ax.axis("off")
            continue

        dapi_xy, yap_xy, dapi_xz, yap_xz = load_xy_xz(oib_path)

        # 上半：XY 视图（DAPI红 + YAP绿）
        # 下半：XZ 视图
        # 用两个子图模拟：上面 XY，下面 XZ
        # 简单做法：整个格画 XY，下面留一条画 XZ
        # 用 inset 不太好，直接在 ax 里分上下

        # 清掉 ax，用上下两个小图
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # XY 占上 70%
        extent = [0, 1, 0.3, 1.0]
        ax_xy = fig.add_axes([
            ax.get_position().x0,
            ax.get_position().y0 + ax.get_position().height * 0.3,
            ax.get_position().width,
            ax.get_position().height * 0.7,
        ])
        # RGB 叠加：R=DAPI, G=YAP
        rgb_xy = np.zeros((*dapi_xy.shape, 3))
        rgb_xy[:, :, 0] = dapi_xy  # R
        rgb_xy[:, :, 1] = yap_xy  # G
        ax_xy.imshow(rgb_xy)
        ax_xy.set_xticks([])
        ax_xy.set_yticks([])
        for s in ax_xy.spines.values():
            s.set_visible(False)

        # XZ 占下 30%
        ax_xz = fig.add_axes([
            ax.get_position().x0,
            ax.get_position().y0,
            ax.get_position().width,
            ax.get_position().height * 0.28,
        ])
        rgb_xz = np.zeros((*dapi_xz.shape, 3))
        rgb_xz[:, :, 0] = dapi_xz
        rgb_xz[:, :, 1] = yap_xz
        ax_xz.imshow(rgb_xz)
        ax_xz.set_xticks([])
        ax_xz.set_yticks([])
        for s in ax_xz.spines.values():
            s.set_visible(False)

        # 中间虚线
        ax_xy.axhline(y=dapi_xy.shape[0] // 2, color="white", ls="--", lw=0.8, alpha=0.7)
        ax_xz.axhline(y=dapi_xz.shape[0] // 2, color="yellow", ls="--", lw=0.8, alpha=0.7)

# 总标题
fig.suptitle("YAP / DAPI", fontsize=16, fontweight="bold", y=0.98, color="#009900")
fig.text(0.95, 0.02, "Scale bars, 50 µm", fontsize=10, ha="right")

out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "yap_dapi_grid.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}", file=sys.stderr)
