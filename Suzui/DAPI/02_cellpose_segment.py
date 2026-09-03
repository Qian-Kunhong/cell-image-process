from pathlib import Path
import numpy as np
import tifffile as tiff
from cellpose import models
import time

# ===== 输入/输出 =====
# 你已经把筛选后的文件复制回原目录，所以继续用原目录

mode=1
SUZUI_ROOT = Path(r"F:\Suzui")
ANALYSIS_ROOT = SUZUI_ROOT / "analysis_out"
# DATASET_NAME = "A-1-1 timelapse"
DATASET_NAME = "A-1-3"
TRAINING_ROOT = SUZUI_ROOT / "training data"

# data
if mode == 1:
    INP = ANALYSIS_ROOT / DATASET_NAME
    OUT = INP / "masks"
    print('processing data')

# # training data SNL
if mode == 2:
    INP = TRAINING_ROOT / "SNL"
    OUT = ANALYSIS_ROOT / "masks_training" / "SNL"
    print('processing SNL training data')

# # training data MEF
if mode == 3:
    INP = TRAINING_ROOT / "MEF"
    OUT = ANALYSIS_ROOT / "masks_training" / "MEF"
    print('processing MEF training data')

OUT.mkdir(parents=True, exist_ok=True)

# 可选：保存简单预览图
SAVE_PREVIEW = True
PREVIEW_DIR = OUT / "preview_tif"
if SAVE_PREVIEW:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ===== Cellpose 4 参数 =====
USE_GPU = True
PRETRAINED_MODEL = "cyto3"   # CP4 默认模型
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
DIAMETER = None              # CP4 中通常不必强行指定；不用就设 None
FILE_GLOB = "*.tif"


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def to_three_channel(img: np.ndarray) -> np.ndarray:
    """
    CP4 / Cellpose-SAM更适合按3通道输入。
    对单通道DAPI图，直接复制成3通道。
    """
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)

    if img.ndim == 3:
        # 已经是 HWC 且 C=3
        if img.shape[-1] == 3:
            return img
        # 单通道 HWC
        if img.shape[-1] == 1:
            return np.repeat(img, 3, axis=-1)
        # CHW 且 C=3
        if img.shape[0] == 3 and img.shape[-1] != 3:
            return np.moveaxis(img, 0, -1)
        # CHW 且 C=1
        if img.shape[0] == 1 and img.shape[-1] != 1:
            x = np.moveaxis(img, 0, -1)
            return np.repeat(x, 3, axis=-1)

    raise ValueError(f"Unsupported image shape for CP4 input: {img.shape}")


def make_preview(img_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    不依赖 opencv，直接把 mask 边界叠到灰度图上。
    """
    base = normalize_to_u8(img_gray)
    rgb = np.stack([base, base, base], axis=-1)

    # 简单边界：相邻像素标签不同的位置
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]

    rgb[boundary] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def main():
    t0_all = time.perf_counter()
    if not INP.exists():
        raise FileNotFoundError(f"Input folder not found: {INP}")

    files = sorted(INP.glob(FILE_GLOB))
    if not files:
        raise RuntimeError(f"No tif files found in: {INP}")

    print(f"Input folder : {INP}")
    print(f"Output folder: {OUT}")
    print(f"Found {len(files)} tif files")

    # ===== CP4写法 =====
    t0_model = time.perf_counter()
    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model=PRETRAINED_MODEL
    )
    print(f"[time] model init: {time.perf_counter() - t0_model:.2f}s | model={PRETRAINED_MODEL} gpu={USE_GPU}")

    done = 0
    skipped = 0
    failed = 0

    for p in files:
        out_npy = OUT / f"{p.stem}_mask.npy"

        if out_npy.exists():
            print(f"[skip] {p.name} -> mask exists")
            skipped += 1
            continue

        try:
            t0_one = time.perf_counter()
            t0 = time.perf_counter()
            img = tiff.imread(p)
            t_read = time.perf_counter() - t0

            # 保留一份2D灰度图做预览
            if img.ndim == 2:
                img_gray = img
            elif img.ndim == 3 and img.shape[-1] == 1:
                img_gray = img[..., 0]
            elif img.ndim == 3 and img.shape[0] == 1:
                img_gray = img[0]
            else:
                # 如果不是明确单通道，预览就用第一通道
                if img.ndim == 3 and img.shape[-1] >= 1:
                    img_gray = img[..., 0]
                elif img.ndim == 3 and img.shape[0] >= 1:
                    img_gray = img[0]
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")

            img_cp4 = to_three_channel(img)

            t0 = time.perf_counter()
            masks, flows, styles = model.eval(
                img_cp4,
                flow_threshold=FLOW_THRESHOLD,
                cellprob_threshold=CELLPROB_THRESHOLD,
                diameter=DIAMETER
            )
            t_eval = time.perf_counter() - t0

            t0 = time.perf_counter()
            np.save(out_npy, masks.astype(np.int32))

            if SAVE_PREVIEW:
                preview = make_preview(img_gray, masks)
                tiff.imwrite(str(PREVIEW_DIR / f"{p.stem}_overlay.tif"), preview)
            t_save = time.perf_counter() - t0

            n_obj = int(masks.max())
            print(f"[ok] {p.name} -> {out_npy.name} | nuclei={n_obj} | read={t_read:.2f}s eval={t_eval:.2f}s save={t_save:.2f}s total={time.perf_counter()-t0_one:.2f}s")
            done += 1

        except Exception as e:
            print(f"[fail] {p.name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Done   : {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")
    print(f"Masks saved to: {OUT}")
    if SAVE_PREVIEW:
        print(f"Preview saved to: {PREVIEW_DIR}")
    print(f"[time] total runtime: {time.perf_counter() - t0_all:.2f}s")


if __name__ == "__main__":
    main()
