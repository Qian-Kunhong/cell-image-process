from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\YAP quantification"
)
OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_outputs"
    / "hepatocyte_confocal_audit"
    / "exported_yap_slices"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for sample_dir in sorted(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("D")):
        c1_files = sorted(sample_dir.glob("*-C1.tif"))
        rows = []
        for c1_path in c1_files:
            c2_path = c1_path.with_name(c1_path.name.replace("-C1.tif", "-C2.tif"))
            c1 = np.array(Image.open(c1_path).convert("L"))
            c2 = np.array(Image.open(c2_path).convert("L"))
            merged = np.stack([c1, c2, np.zeros_like(c1)], axis=-1)
            tiles = [
                Image.fromarray(np.stack([c1] * 3, axis=-1)),
                Image.fromarray(np.stack([np.zeros_like(c2), c2, np.zeros_like(c2)], axis=-1)),
                Image.fromarray(merged),
            ]
            tile_size = 260
            label_width = 70
            row = Image.new("RGB", (label_width + tile_size * 3, tile_size), "black")
            ImageDraw.Draw(row).text((8, 10), c1_path.stem.split("-")[0], fill="white")
            for index, tile in enumerate(tiles):
                row.paste(tile.resize((tile_size, tile_size), Image.Resampling.BILINEAR), (label_width + index * tile_size, 0))
            rows.append(row)

        if not rows:
            continue
        header = Image.new("RGB", (rows[0].width, 36), "black")
        draw = ImageDraw.Draw(header)
        for index, label in enumerate(("C1", "C2", "C1 red + C2 green")):
            draw.text((70 + index * 260 + 8, 10), label, fill="white")
        sheet = Image.new("RGB", (header.width, header.height + sum(row.height for row in rows)), "black")
        sheet.paste(header, (0, 0))
        y = header.height
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(OUTPUT / f"{sample_dir.name}_exported_slices.png")


if __name__ == "__main__":
    main()
