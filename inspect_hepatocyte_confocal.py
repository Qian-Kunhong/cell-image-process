from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


RAW_ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining"
)
DATASETS = {
    "ecad_f_actin": RAW_ROOT / "Confocal - E-cad F-actin",
    "yap": RAW_ROOT / "Confocal - YAP",
}
OUTPUT_DIR = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit"


def section_value(settings, section: str, key: str, default=None):
    return settings.get(section, {}).get(key, default)


def robust_uint8(image: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    nonzero = image[image > 0]
    if nonzero.size < 100:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(nonzero, (0.5, 99.7))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    scaled = np.power(scaled, gamma)
    return np.rint(scaled * 255.0).astype(np.uint8)


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError(f"Expected ZCYX metadata, received {axes} with shape {array.shape}")
    # oiffile 2025.5.10 reports ZCYX metadata for these OIBs but returns
    # the materialized array as CZYX. Detect that case from the small channel
    # axis and transpose explicitly.
    if array.ndim == 4 and array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def channel_tiles(channel_images: np.ndarray) -> list[Image.Image]:
    channels = [robust_uint8(channel_images[index]) for index in range(channel_images.shape[0])]
    while len(channels) < 3:
        channels.append(np.zeros_like(channels[0]))

    dapi = channels[0]
    green = channels[1]
    red = channels[2]
    merged = np.stack([red, green, dapi], axis=-1)
    gray_tiles = [
        np.stack([dapi, dapi, dapi], axis=-1),
        np.stack([np.zeros_like(green), green, np.zeros_like(green)], axis=-1),
        np.stack([red, np.zeros_like(red), np.zeros_like(red)], axis=-1),
    ]
    return [Image.fromarray(tile) for tile in gray_tiles] + [Image.fromarray(merged)]


def make_projection_tiles(array: np.ndarray) -> list[Image.Image]:
    return channel_tiles(array.max(axis=0))


def focus_slice_index(array: np.ndarray) -> int:
    dapi = array[:, 0].astype(np.float32)
    score = np.mean(np.abs(np.diff(dapi, axis=1)), axis=(1, 2))
    score += np.mean(np.abs(np.diff(dapi, axis=2)), axis=(1, 2))
    return int(np.argmax(score))


def make_focus_tiles(array: np.ndarray) -> tuple[int, list[Image.Image]]:
    z_index = focus_slice_index(array)
    return z_index, channel_tiles(array[z_index])


def add_labeled_row(label: str, tiles: list[Image.Image], tile_size: int = 220) -> Image.Image:
    label_width = 150
    row = Image.new("RGB", (label_width + tile_size * len(tiles), tile_size), "black")
    draw = ImageDraw.Draw(row)
    draw.text((8, 10), label, fill="white")
    for index, tile in enumerate(tiles):
        resized = tile.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        row.paste(resized, (label_width + index * tile_size, 0))
    return row


def build_header(tile_size: int = 220) -> Image.Image:
    labels = ["DAPI MIP", "CH2 MIP", "CH3 MIP", "Merged MIP"]
    label_width = 150
    header = Image.new("RGB", (label_width + tile_size * len(labels), 36), "black")
    draw = ImageDraw.Draw(header)
    for index, label in enumerate(labels):
        draw.text((label_width + index * tile_size + 8, 10), label, fill="white")
    return header


def inspect_file(dataset: str, path: Path) -> tuple[dict, list[Image.Image], list[Image.Image]]:
    with oiffile.OifFile(path) as image_file:
        axes = image_file.axes
        shape = tuple(int(value) for value in image_file.shape)
        dtype = str(image_file.dtype)
        settings = image_file.mainfile
        array = ensure_zcyx(image_file.asarray(), axes)

        x_axis = settings.get("Axis 0 Parameters Common", {})
        y_axis = settings.get("Axis 1 Parameters Common", {})
        z_axis = settings.get("Axis 3 Parameters Common", {})
        common = settings.get("Acquisition Parameters Common", {})

        row = {
            "dataset": dataset,
            "sample": path.stem,
            "path": str(path),
            "axes": axes,
            "shape": "x".join(str(value) for value in shape),
            "dtype": dtype,
            "capture_date": common.get("ImageCaputreDate"),
            "device": common.get("Acquisition Device"),
            "scan_mode": common.get("ScanMode"),
            "sequential_mode": common.get("SequentialMode"),
            "zoom": common.get("ZoomValue"),
            "xy_width_um": x_axis.get("EndPosition"),
            "xy_height_um": y_axis.get("EndPosition"),
            "pixel_size_x_um": (
                float(x_axis.get("EndPosition", 0)) / float(x_axis.get("MaxSize", 1))
            ),
            "pixel_size_y_um": (
                float(y_axis.get("EndPosition", 0)) / float(y_axis.get("MaxSize", 1))
            ),
            "z_step_um": float(z_axis.get("Interval", 0)) / 1000.0,
        }

        dtype_max = float(np.iinfo(array.dtype).max)
        for channel_index in range(array.shape[1]):
            channel_number = channel_index + 1
            channel_settings = settings.get(f"Channel {channel_number} Parameters", {})
            values = array[:, channel_index]
            prefix = f"ch{channel_number}"
            row[f"{prefix}_dye"] = channel_settings.get("DyeName")
            row[f"{prefix}_excitation_nm"] = channel_settings.get("ExcitationWavelength")
            row[f"{prefix}_emission_nm"] = channel_settings.get("EmissionWavelength")
            row[f"{prefix}_pmt_voltage"] = channel_settings.get("AnalogPMTVoltage")
            row[f"{prefix}_p50"] = float(np.percentile(values, 50))
            row[f"{prefix}_p99"] = float(np.percentile(values, 99))
            row[f"{prefix}_p999"] = float(np.percentile(values, 99.9))
            row[f"{prefix}_max"] = int(values.max())
            row[f"{prefix}_zero_fraction"] = float(np.mean(values == 0))
            row[f"{prefix}_saturated_fraction"] = float(np.mean(values >= dtype_max))

        focus_index, focus_tiles = make_focus_tiles(array)
        row["focus_z_index_zero_based"] = focus_index
        row["focus_z_um_from_stack_start"] = focus_index * row["z_step_um"]
        return row, make_projection_tiles(array), focus_tiles


def save_contact_sheet(dataset: str, rows: list[Image.Image]) -> Path:
    header = build_header()
    width = max([header.width] + [row.width for row in rows])
    height = header.height + sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), "black")
    sheet.paste(header, (0, 0))
    y = header.height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output_path = OUTPUT_DIR / f"{dataset}_contact_sheet.png"
    sheet.save(output_path)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    preview_paths: list[str] = []

    for dataset, directory in DATASETS.items():
        projection_rows: list[Image.Image] = []
        focus_rows: list[Image.Image] = []
        for path in sorted(directory.glob("*.oib")):
            print(f"Reading {dataset}/{path.name}", flush=True)
            record, projection_tiles, focus_tiles = inspect_file(dataset, path)
            records.append(record)
            projection_rows.append(add_labeled_row(path.stem, projection_tiles))
            focus_rows.append(
                add_labeled_row(f"{path.stem} z={record['focus_z_index_zero_based']}", focus_tiles)
            )
        preview_paths.append(str(save_contact_sheet(f"{dataset}_mip", projection_rows)))
        preview_paths.append(str(save_contact_sheet(f"{dataset}_focus", focus_rows)))

    fieldnames = sorted({key for record in records for key in record})
    csv_path = OUTPUT_DIR / "confocal_metadata_and_intensity_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "raw_root": str(RAW_ROOT),
        "file_count": len(records),
        "datasets": {name: sum(row["dataset"] == name for row in records) for name in DATASETS},
        "samples": sorted({row["sample"] for row in records}),
        "metadata_csv": str(csv_path),
        "contact_sheets": preview_paths,
    }
    summary_path = OUTPUT_DIR / "confocal_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
