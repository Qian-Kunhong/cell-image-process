from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# =========================
# Config
# =========================
ROOT_DIR = Path(
    r"F:\Suzui\analysis_out\A-1-1 timelapse_supervised_prediction_xgb_relative_global_umap"
)
PREDICTION_CSV = ROOT_DIR / "nucleus_predictions.csv"

JOBS = [
    {
        "folder": "inference_umap_by_image",
        "pattern": "*_umap.png",
        "output": "timelapse_umap.gif",
        "annotate_time": True,
        "annotate_scale_bar": False,
    },
    {
        "folder": "overlays_deviation_score",
        "pattern": "*_deviation_score_overlay.png",
        "output": "timelapse_deviation_score.gif",
        "annotate_time": True,
        "annotate_scale_bar": True,
    },
]

FRAME_DURATION_MS = 500
LOOP_FOREVER = True
SCALE_BAR_UM = 100.0
TIME_LABEL_PREFIX = "time "
TEXT_MARGIN_PX = 70
SCALE_BAR_MARGIN_BOTTOM_PX = 46
SCALE_BAR_LINE_WIDTH_PX = 5
SCALE_BAR_TEXT_GAP_PX = 16
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"time0*(\d+)", path.name, flags=re.IGNORECASE)
    time_idx = int(m.group(1)) if m else 10**9
    return time_idx, path.name.lower()


def extract_time_index(name: str) -> int | None:
    m = re.search(r"time0*(\d+)", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def read_pixel_size_map(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}

    out: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row.get("image_name", "")
            pixel_size = row.get("pixel_size_mean_um", "")
            if not image_name or image_name in out:
                continue
            try:
                out[image_name] = float(pixel_size)
            except Exception:
                continue
    return out


def normalize_image_key(name: str) -> str:
    stem = Path(name).stem
    stem = stem.lower()
    stem = re.sub(r"\.[a-z0-9]+$", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def get_time_label(path: Path) -> str | None:
    time_idx = extract_time_index(path.name)
    if time_idx is None:
        return None
    return f"{TIME_LABEL_PREFIX}{time_idx}"


def get_pixel_size_um(path: Path, pixel_size_map: dict[str, float]) -> float | None:
    key = normalize_image_key(path.stem)
    for image_name, pixel_size in pixel_size_map.items():
        if normalize_image_key(image_name) in key or key in normalize_image_key(image_name):
            return pixel_size
    return None


def draw_text_with_shadow(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, fill=SHADOW_COLOR, font=font)
    draw.text((x, y), text, fill=TEXT_COLOR, font=font)


def annotate_frame(img: Image.Image, path: Path, annotate_time: bool, annotate_scale_bar: bool, pixel_size_map: dict[str, float]) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size

    time_font = load_font(max(24, width // 28))
    bar_font = load_font(max(18, width // 42))

    if annotate_time:
        time_text = get_time_label(path)
        if time_text:
            draw_text_with_shadow(draw, (TEXT_MARGIN_PX, TEXT_MARGIN_PX), time_text, time_font)

    if annotate_scale_bar:
        pixel_size_um = get_pixel_size_um(path, pixel_size_map)
        if pixel_size_um and pixel_size_um > 0:
            bar_len_px = int(round(SCALE_BAR_UM / pixel_size_um))
            bar_len_px = max(40, min(bar_len_px, width - 2 * TEXT_MARGIN_PX))
            x0 = (width - bar_len_px) // 2
            x1 = x0 + bar_len_px
            y = height - SCALE_BAR_MARGIN_BOTTOM_PX
            draw.line((x0, y, x1, y), fill=TEXT_COLOR, width=SCALE_BAR_LINE_WIDTH_PX)

            bar_text = f"{int(SCALE_BAR_UM)} μm"
            text_bbox = draw.textbbox((0, 0), bar_text, font=bar_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_x = x0 + (bar_len_px - text_w) / 2
            text_y = y - text_h - SCALE_BAR_TEXT_GAP_PX
            draw_text_with_shadow(draw, (text_x, text_y), bar_text, bar_font)

    return out


def build_gif(
    folder: Path,
    pattern: str,
    output_path: Path,
    annotate_time: bool,
    annotate_scale_bar: bool,
    pixel_size_map: dict[str, float],
) -> None:
    files = sorted(folder.glob(pattern), key=sort_key)
    if not files:
        raise RuntimeError(f"No files matched {pattern} in {folder}")

    frames: list[Image.Image] = []
    target_size = None

    for path in files:
        img = Image.open(path).convert("RGB")
        img = annotate_frame(
            img=img,
            path=path,
            annotate_time=annotate_time,
            annotate_scale_bar=annotate_scale_bar,
            pixel_size_map=pixel_size_map,
        )
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        frames.append(img)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0 if LOOP_FOREVER else 1,
        optimize=False,
        disposal=2,
    )

    print(f"[saved] {output_path}")


def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")

    pixel_size_map = read_pixel_size_map(PREDICTION_CSV)

    for job in JOBS:
        folder = ROOT_DIR / job["folder"]
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        output_path = folder / job["output"]
        build_gif(
            folder=folder,
            pattern=job["pattern"],
            output_path=output_path,
            annotate_time=job["annotate_time"],
            annotate_scale_bar=job["annotate_scale_bar"],
            pixel_size_map=pixel_size_map,
        )


if __name__ == "__main__":
    main()
