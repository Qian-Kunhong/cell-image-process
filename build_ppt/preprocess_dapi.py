from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation\03_Raw data\Immunostaining\IN Cell\D0_3_17_21_E-cad N-cad\10x")
OUTPUT = Path(__file__).resolve().parent / "assets"


def dapi_path(day: str, condition: str) -> Path:
    separator = "-" if day == "D21" else "_"
    return SOURCE / day / f"{condition}{separator}DAPI.tif"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [dapi_path(day, condition) for day in ("D0", "D3", "D17", "D21") for condition in ("HA00", "HA01", "HA05", "HA10")]
    arrays = [np.asarray(Image.open(path), dtype=np.float32) for path in paths]
    sampled = np.concatenate([array[::8, ::8].ravel() for array in arrays])
    low, high = np.percentile(sampled, (0.5, 99.7))
    for path, array in zip(paths, arrays):
        display = np.clip((array - low) / max(high - low, 1), 0, 1)
        display = np.power(display, 0.72)
        rgb = np.zeros((*display.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (display * 55).astype(np.uint8)
        rgb[..., 1] = (display * 120).astype(np.uint8)
        rgb[..., 2] = (display * 255).astype(np.uint8)
        name = f"{path.parent.name}_{path.stem.replace('-', '_')}.png"
        target = OUTPUT / name
        if not target.exists():
            Image.fromarray(rgb, mode="RGB").save(target, optimize=True)

    figure_dir = Path(r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation\02_Submission documents")
    for figure_name in ("Figure 1.tif", "Figure 2.tif", "Figure 3.tif", "Figure 3-graphic.tif", "Figure 4.tif", "Figure 5.tif"):
        image = Image.open(figure_dir / figure_name).convert("RGB")
        image.thumbnail((1800, 1400), Image.Resampling.LANCZOS)
        image.save(OUTPUT / (Path(figure_name).stem.replace(" ", "_") + ".png"))


if __name__ == "__main__":
    main()
