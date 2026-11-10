"""10 按 Ekin 三面板布局展示 DAPI 分割 QC；不展示表型簇。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from skimage import color
    from skimage.segmentation import find_boundaries

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

    # Panel 3: 实例标签，仅用于检查分割
    instance_canvas = color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    instance_canvas[find_boundaries(labels, mode="inner")] = 0.0
    axes[2].imshow(instance_canvas)
    axes[2].set_title("3\nInstance labels (QC only)", fontsize=16, fontweight="bold", pad=8)

    for ax in axes:
        ax.axis("off")

    fig.suptitle("D50D (No HA, Day 0): DAPI segmentation QC",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=.025, right=.985, top=.84, bottom=.15, wspace=.035)

    out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "ekin_style_overlay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
