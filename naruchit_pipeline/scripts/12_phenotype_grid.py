"""12 分割 QC 网格图：行=条件，列=时间点；不赋予表型簇。"""
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

    # 读取各样本 DAPI 与已有分割，仅做 QC 展示。
    focus = pd.read_csv(FOCUS_CSV)
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

        sample_data[sample] = {"dapi": dapi_best, "mask": mask}

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

            # DAPI 归一化
            lo, hi = np.percentile(dapi[dapi > 0], (0.5, 99.7))
            dapi_display = np.clip((dapi - lo) / (hi - lo + 1e-6), 0, 1)

            # DAPI 灰度底 + 半透明实例色；颜色仅帮助检查相邻标签。
            ax.imshow(dapi_display, cmap="gray", vmin=0, vmax=1)
            instance_rgb = color.label2rgb(mask, bg_label=0, bg_color=(0, 0, 0))
            background = np.repeat((mask == 0)[..., None], 3, axis=2)
            ax.imshow(np.ma.masked_where(background, instance_rgb), alpha=0.35)

            # 黑色分隔线
            boundaries = find_boundaries(mask, mode="inner")
            ax.contour(boundaries, colors="black", linewidths=0.3)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("DAPI segmentation QC across HA conditions (instance colors are arbitrary)",
                 fontsize=16, fontweight="bold", y=0.97)

    out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "segmentation_grid.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
