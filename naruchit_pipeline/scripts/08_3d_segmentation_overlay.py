"""08 用现有 Cellpose 结果绘制分割 QC：不赋予未经验证的表型簇。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import color, exposure, segmentation

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

    # 实例颜色只用于检查相邻标签是否分开；颜色没有生物学含义。
    instance_rgb = color.label2rgb(labels_2d, bg_label=0, bg_color=(0, 0, 0))

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

    # 3. 实例标签（仅 QC）
    axes[2].imshow(instance_rgb)
    axes[2].set_title("3  Instance labels (QC only)", fontsize=14, fontweight="bold", loc="left")
    axes[2].axis("off")

    fig.tight_layout()
    out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "segmentation_3d_overlay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
