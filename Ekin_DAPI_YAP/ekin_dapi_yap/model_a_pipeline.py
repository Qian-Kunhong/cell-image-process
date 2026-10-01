from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops
from skimage.segmentation import expand_labels, find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent))
from .summarize_group_differences import build_descriptive_tables
from .phenotype_display import save_clear_phenotypes
from .yap_ratio_qc import ALGORITHM_VERSION, build_ring_labels, load_config, measure_array, qc_summary
from .yap_ratio_display import save_ratio_figure, save_sampling_figure, save_qc_summary_figure


DEFAULT_DATA_ROOT = Path(r"E:\Kino-oka Lab\Immunostaining Data_Ekin\2307YapLocalizationImmuno")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs" / "composite_no_dapi_intensity" / "40x" / "all_fields"
)
FOLDER_PATTERN = re.compile(
    r"^(?P<experimental_group>Ctrl|HA1|HA2)-(?P<seeding_density_code>2_5|5|7_5|10)-"
    r"(?P<magnification>40)-Image Export-(?P<export_index>\d+)$",
    flags=re.IGNORECASE,
)
EXPERIMENTAL_GROUP_ORDER = {"Ctrl": 0, "HA1": 1, "HA2": 2}
# Experimental conditions use a separate visual key from morphology phenotypes.
EXPERIMENTAL_GROUP_COLORS = {"Ctrl": "#171717", "HA1": "#D000B5", "HA2": "#00B8C4"}
EXPERIMENTAL_GROUP_METADATA = {
    "Ctrl": {
        "experimental_group_description": "Control (No HA)",
        "ha_exposure_h": None,
        "ha_concentration_nM": 0.0,
    },
    "HA1": {
        "experimental_group_description": "HA-1 (72 h, 2.5 nM)",
        "ha_exposure_h": 72.0,
        "ha_concentration_nM": 2.5,
    },
    "HA2": {
        "experimental_group_description": "HA-2 (48 h, 5 nM)",
        "ha_exposure_h": 48.0,
        "ha_concentration_nM": 5.0,
    },
}
SEEDING_DENSITY_CELLS_PER_CM2 = {
    "2.5": 2_500.0,
    "5": 5_000.0,
    "7.5": 7_500.0,
    "10": 10_000.0,
}


def load_model_a_core():
    repository_root = PROJECT_ROOT.parent
    candidates = [
        repository_root / "shared" / "dapi_model_a.py",
    ]
    core_path = next((path for path in candidates if path.exists()), None)
    if core_path is None:
        raise FileNotFoundError(
            "Could not locate the validated DAPI-only Model A core. Checked: "
            + ", ".join(str(path) for path in candidates)
        )
    spec = importlib.util.spec_from_file_location("validated_dapi_model_a_core", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Model A core: {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, core_path


CORE, CORE_PATH = load_model_a_core()


@dataclass(frozen=True)
class YAPImageSet:
    experimental_group_label: str
    seeding_density_folder_label: str
    seeding_density_cells_per_cm2: float
    magnification: str
    export_index: int
    image_id: str
    folder: Path
    dapi_org_path: Path
    yap_org_path: Path
    merge_path: Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DAPI-only unsupervised morphology discovery with post-hoc continuous "
            "YAP nuclear/perinuclear enrichment."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cpu", action="store_true", help="Run Cellpose on CPU.")
    parser.add_argument("--reuse-masks", action="store_true")
    parser.add_argument(
        "--skip-umap", action="store_true",
        help="Validation/debug mode: skip visualization-only UMAP and use PCA1/2 as plotting coordinates.",
    )
    parser.add_argument("--edge-buffer-px", type=int, default=2,
                        help="Exclude nuclei whose mask is within this many pixels of the image edge (default: 2).")
    parser.add_argument("--yap-qc-config", type=Path, help="Optional JSON overrides for post-hoc YAP QC only.")
    parser.add_argument("--yap-background-roi-dir", type=Path,
                        help="Optional binary PNG background ROIs named <image_id>_background.png; nonzero = cell-free.")
    args = parser.parse_args(argv)
    args.fit_magnification = "40x"
    args.feature_set = "composite"
    return args


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)


def experimental_group_display(label: str) -> str:
    metadata = EXPERIMENTAL_GROUP_METADATA[label]
    return f"{label}: {metadata['experimental_group_description']}"


def fitted_magnification(result_df: pd.DataFrame) -> str:
    values = result_df["magnification"].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected one independently fitted magnification, found: {values}")
    return values[0]


def compute_umap_for_dataset(scaled: np.ndarray) -> np.ndarray:
    """Use the validated exact path for small trials and ordinary NNDescent for larger runs."""
    if len(scaled) < 4096:
        return CORE.compute_umap(scaled)

    started = time.perf_counter()
    import umap

    print(f"[umap-large +{time.perf_counter() - started:7.1f}s] full import complete", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, max(2, len(scaled) - 1)),
        min_dist=0.1,
        random_state=CORE.RANDOM_STATE,
        init="random",
        n_epochs=CORE.UMAP_N_EPOCHS,
        n_jobs=1,
    )
    print(f"[umap-large +{time.perf_counter() - started:7.1f}s] fitting embedding", flush=True)
    embedding = reducer.fit_transform(scaled)
    print(f"[umap-large +{time.perf_counter() - started:7.1f}s] fit complete", flush=True)
    return embedding


def locate_dataset_root(root: Path) -> Path:
    root = root.resolve()
    direct_matches = [path for path in root.iterdir() if path.is_dir() and FOLDER_PATTERN.match(path.name)]
    if direct_matches:
        return root
    children = [path for path in root.iterdir() if path.is_dir()]
    nested = [
        child
        for child in children
        if any(path.is_dir() and FOLDER_PATTERN.match(path.name) for path in child.iterdir())
    ]
    if len(nested) == 1:
        return nested[0]
    raise RuntimeError(f"Could not identify the image-set directory below {root}")


def discover_image_sets(root: Path) -> tuple[list[YAPImageSet], pd.DataFrame]:
    dataset_root = locate_dataset_root(root)
    sets: list[YAPImageSet] = []
    rows: list[dict] = []
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        match = FOLDER_PATTERN.match(folder.name)
        if not match:
            continue
        raw_experimental_group = match.group("experimental_group")
        experimental_group_label = (
            "Ctrl" if raw_experimental_group.lower() == "ctrl" else raw_experimental_group.upper()
        )
        group_metadata = EXPERIMENTAL_GROUP_METADATA[experimental_group_label]
        seeding_density_folder_label = match.group("seeding_density_code").replace("_", ".")
        seeding_density = SEEDING_DENSITY_CELLS_PER_CM2[seeding_density_folder_label]
        magnification = f"{match.group('magnification')}x"
        stem = folder.name
        dapi_org = folder / f"{stem}_DAPI_ORG.png"
        yap_org = folder / f"{stem}_AF488_ORG.png"
        merge = folder / f"{stem}.png"
        status = "ok" if all(path.exists() for path in (dapi_org, yap_org, merge)) else "missing_required_file"
        rows.append(
            {
                "image_set_folder": str(folder),
                "experimental_group_label": experimental_group_label,
                **group_metadata,
                "seeding_density_folder_label": seeding_density_folder_label,
                "seeding_density_cells_per_cm2": seeding_density,
                "magnification": magnification,
                "export_index": int(match.group("export_index")),
                "dapi_org_path": str(dapi_org),
                "yap_af488_org_path": str(yap_org),
                "merge_path": str(merge),
                "pairing_status": status,
            }
        )
        if status != "ok":
            continue
        image_id = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
        sets.append(
            YAPImageSet(
                experimental_group_label=experimental_group_label,
                seeding_density_folder_label=seeding_density_folder_label,
                seeding_density_cells_per_cm2=seeding_density,
                magnification=magnification,
                export_index=int(match.group("export_index")),
                image_id=image_id,
                folder=folder,
                dapi_org_path=dapi_org,
                yap_org_path=yap_org,
                merge_path=merge,
            )
        )
    sets.sort(
        key=lambda item: (
            EXPERIMENTAL_GROUP_ORDER.get(item.experimental_group_label, 99),
            item.seeding_density_cells_per_cm2,
            int(item.magnification.rstrip("x")),
        )
    )
    if not sets:
        raise RuntimeError(f"No complete YAP/DAPI image sets found below {dataset_root}")
    return sets, pd.DataFrame(rows)


def load_gray(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale _ORG PNG, got shape={image.shape}: {path}")
    return image.astype(np.float32)


def validate_morphology_feature_names(columns) -> None:
    forbidden = (
        "yap",
        "af488",
        "oct4",
        "oct_4",
        "oct-4",
        "experimental_group",
        "ha_exposure",
        "ha_concentration",
        "seeding_density",
    )
    violations = [
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in forbidden)
    ]
    if violations:
        raise AssertionError(f"Post-hoc marker leakage into morphology model: {violations}")


def segment_image_sets(
    image_sets: list[YAPImageSet], output_root: Path, use_gpu: bool, reuse_masks: bool
) -> dict[str, Path]:
    mask_dir = output_root / "segmentation" / "masks"
    overlay_dir = output_root / "segmentation" / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = {item.image_id: mask_dir / f"{item.image_id}_mask.npy" for item in image_sets}
    pending = [item for item in image_sets if not (reuse_masks and mask_paths[item.image_id].exists())]
    model = CORE.load_cellpose_model(use_gpu=use_gpu) if pending else None
    for item in image_sets:
        dapi = load_gray(item.dapi_org_path)
        mask_path = mask_paths[item.image_id]
        if reuse_masks and mask_path.exists():
            mask = np.load(mask_path)
        else:
            cellpose_input = np.stack([dapi, dapi, dapi], axis=-1)
            masks, _, _ = model.eval(
                cellpose_input,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                diameter=None,
            )
            mask = np.asarray(masks, dtype=np.int32)
            np.save(mask_path, mask)
        if mask.shape != dapi.shape:
            raise ValueError(f"Mask/image mismatch for {item.image_id}")
        CORE.save_mask_overlay(dapi, mask, overlay_dir / f"{item.image_id}_overlay.png")
        print(f"[segmentation] {item.image_id}: nuclei={int(mask.max())}", flush=True)
    return mask_paths


def extract_dapi_features(item: YAPImageSet, mask_path: Path, edge_buffer_px: int = 2) -> pd.DataFrame:
    if edge_buffer_px < 0:
        raise ValueError("edge_buffer_px must be nonnegative")
    dapi = load_gray(item.dapi_org_path)
    mask = np.load(mask_path)
    height, width = dapi.shape
    rows: list[dict] = []
    for prop in regionprops(mask, intensity_image=dapi):
        coords = prop.coords
        values = dapi[coords[:, 0], coords[:, 1]].astype(float)
        min_row, min_col, max_row, max_col = prop.bbox
        area = float(prop.area)
        perimeter = float(prop.perimeter)
        major = float(prop.major_axis_length)
        minor = float(prop.minor_axis_length)
        touches_border = bool(min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width)
        edge_gap = int(min(min_row, min_col, height - max_row, width - max_col))
        rows.append(
            {
                "cell_id": f"{item.image_id}__cell_{int(prop.label):05d}",
                "experimental_group_label": item.experimental_group_label,
                **EXPERIMENTAL_GROUP_METADATA[item.experimental_group_label],
                "seeding_density_folder_label": item.seeding_density_folder_label,
                "seeding_density_cells_per_cm2": item.seeding_density_cells_per_cm2,
                "replicate": "not_provided",
                "magnification": item.magnification,
                "imaging_condition": f"magnification_{item.magnification}",
                "image_id": item.image_id,
                "export_index": item.export_index,
                "image_set_folder": str(item.folder),
                "label": int(prop.label),
                "centroid_row_px": float(prop.centroid[0]),
                "centroid_col_px": float(prop.centroid[1]),
                "dapi_org_path": str(item.dapi_org_path),
                "yap_af488_org_path": str(item.yap_org_path),
                "merge_image_path": str(item.merge_path),
                "touches_border": touches_border,
                "image_edge_gap_px": edge_gap,
                "edge_buffer_px": edge_buffer_px,
                "area_px": area,
                "perimeter_px": perimeter,
                "equivalent_diameter_px": float(prop.equivalent_diameter_area),
                "major_axis_length_px": major,
                "minor_axis_length_px": minor,
                "aspect_ratio": CORE.safe_div(major, minor),
                "eccentricity": float(prop.eccentricity),
                "circularity": CORE.circularity(area, perimeter),
                "solidity": float(prop.solidity),
                "extent": float(prop.extent),
                "dapi_mean_intensity": float(np.mean(values)),
                "dapi_std_intensity": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "dapi_min_intensity": float(np.min(values)),
                "dapi_max_intensity": float(np.max(values)),
                "dapi_intensity_range": float(np.max(values) - np.min(values)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["qc_keep"] = frame["image_edge_gap_px"] > edge_buffer_px
    frame["qc_reason"] = np.where(
        frame["touches_border"], "touches_border",
        np.where(frame["qc_keep"], "keep", f"within_{edge_buffer_px}px_image_edge"),
    )
    # Incomplete nuclei must not supply area/shape values to neighbors either.
    # Raw masks remain in use as occupancy exclusions for YAP ring sampling.
    kept = frame.loc[frame["qc_keep"]].reset_index(drop=True)
    if not kept.empty:
        kept = CORE.add_composite_features(CORE.add_spatial_features(kept))
        for column in kept.columns.difference(frame.columns):
            frame[column] = frame["label"].map(kept.set_index("label")[column])
    return frame


def extract_all_dapi_features(
    image_sets: list[YAPImageSet], mask_paths: dict[str, Path], edge_buffer_px: int = 2
) -> pd.DataFrame:
    tables = []
    for item in image_sets:
        frame = extract_dapi_features(item, mask_paths[item.image_id], edge_buffer_px=edge_buffer_px)
        tables.append(frame)
        print(
            f"[features] {item.image_id}: total={len(frame)}, "
            f"qc_keep={int(frame['qc_keep'].sum()) if len(frame) else 0}",
            flush=True,
        )
    return pd.concat(tables, ignore_index=True)


def measure_yap_posthoc(item: YAPImageSet, mask_path: Path, config=None,
                        background_roi_dir: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    mask = np.load(mask_path)
    yap = load_gray(item.yap_org_path)
    background_roi = None
    if background_roi_dir is not None:
        roi_path = background_roi_dir / f"{item.image_id}_background.png"
        if roi_path.exists():
            background_roi = np.asarray(Image.open(roi_path).convert("L")) > 0
    return measure_array(yap, mask, item.image_id, config, background_roi)


def phenotype_colors(labels) -> dict[str, tuple]:
    return CORE.phenotype_colors(labels)


def save_ring_qc_overlay(
    item: YAPImageSet, mask_path: Path, ring_labels: np.ndarray, info: dict, path: Path,
    measurements: pd.DataFrame | None = None,
) -> None:
    if measurements is not None:
        save_sampling_figure(load_gray(item.yap_org_path), np.load(mask_path), ring_labels,
                             measurements, info, path)
        return
    yap = CORE.normalize_to_uint8(load_gray(item.yap_org_path))
    base = np.stack([yap, yap, yap], axis=-1).astype(float)
    ring = ring_labels > 0
    base[ring] = 0.68 * base[ring] + 0.32 * np.array([255.0, 190.0, 30.0])
    mask = np.load(mask_path)
    base[find_boundaries(mask, mode="inner")] = np.array([30.0, 144.0, 255.0])
    base[find_boundaries(ring_labels, mode="inner")] = np.array([255.0, 190.0, 30.0])
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ax.imshow(np.clip(base, 0, 255).astype(np.uint8))
    handles = [
        Line2D([], [], color="#1e90ff", label="DAPI-defined nuclear boundary"),
        Patch(facecolor="#ffbe1e", alpha=0.5, label="Non-overlapping perinuclear sampling ring"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.set_title(
        f"{experimental_group_display(item.experimental_group_label)}, "
        f"{item.seeding_density_cells_per_cm2:g} cells/cm², {item.magnification}: YAP ring QC\n"
        f"inner gap={info['inner_gap_px']} px; outer distance={info['outer_distance_px']} px"
    )
    ax.axis("off")
    fig.subplots_adjust(right=0.77, top=0.90)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_phenotype_overlay(
    item: YAPImageSet, mask_path: Path, result_df: pd.DataFrame, path: Path
) -> None:
    mask = np.load(mask_path)
    sub = result_df[result_df["image_id"].eq(item.image_id)]
    # Keep the whole fitted model's palette, including absent components (n=0).
    colors = phenotype_colors(result_df["dominant_phenotype"])
    save_clear_phenotypes(
        load_gray(item.dapi_org_path), mask, sub, colors,
        f"{experimental_group_display(item.experimental_group_label)}, "
        f"{item.seeding_density_cells_per_cm2:g} cells/cm², {item.magnification}: DAPI-only phenotype",
        path,
        mask_path.parent.parent / "masks_qc_keep" / f"{item.image_id}_mask_qc_keep.npy",
        edge_buffer_px=int(sub["edge_buffer_px"].iloc[0]) if "edge_buffer_px" in sub and len(sub) else 0,
    )


def save_yap_ratio_overlay(
    item: YAPImageSet, mask_path: Path, result_df: pd.DataFrame, path: Path, raw: bool = False
) -> None:
    save_ratio_figure(np.asarray(Image.open(item.merge_path).convert("RGB")), np.load(mask_path),
                      result_df[result_df["image_id"].eq(item.image_id)],
                      f"{experimental_group_display(item.experimental_group_label)}, "
                      f"{item.seeding_density_cells_per_cm2:g} cells/cm², {item.magnification}", path, raw=raw)


def save_square_umap_scatter_plots(result_df: pd.DataFrame, figures_dir: Path) -> dict:
    """Redraw the two categorical UMAP plots; no embedding/model recomputation."""
    magnification = fitted_magnification(result_df)
    colors = phenotype_colors(result_df["dominant_phenotype"])
    xy = result_df[["umap_1", "umap_2"]].to_numpy(dtype=float)
    if not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Square UMAP plots require finite, nonempty existing coordinates")
    midpoint = (xy.min(axis=0) + xy.max(axis=0)) / 2
    span = max(float(np.ptp(xy, axis=0).max()), 1e-6) * 1.10
    limits = [(float(center-span/2), float(center+span/2)) for center in midpoint]
    group_colors = EXPERIMENTAL_GROUP_COLORS
    layouts = {}
    for mode, filename in (
        ("phenotype", "umap_dominant_phenotype.png"),
        ("ha_group", "umap_posthoc_ha_experimental_group.png"),
    ):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0.14, 0.20, 0.72, 0.72])
        if mode == "phenotype":
            for name, sub in result_df.groupby("dominant_phenotype"):
                ax.scatter(sub["umap_1"], sub["umap_2"], s=10, alpha=.70, color=colors[name], label=name)
            title = f"{magnification} pooled dominant morphology phenotypes\nDAPI-only UMAP/GMM; YAP excluded"
            legend_columns = min(3, len(colors))
        else:
            for group in ["Ctrl", "HA1", "HA2"]:
                sub = result_df[result_df["experimental_group_label"].eq(group)]
                ax.scatter(sub["umap_1"], sub["umap_2"], s=10, alpha=.90,
                           color=group_colors[group], label=experimental_group_display(group))
            title = f"{magnification} post-hoc HA experimental-group distribution\non DAPI-only UMAP"
            legend_columns = 1
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xlim=limits[0], ylim=limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=14, pad=14)
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        legend = fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .025),
                            ncol=legend_columns, markerscale=1.5, fontsize=11, frameon=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box, legend_box = ax.get_window_extent(renderer), legend.get_window_extent(renderer)
        if not np.isclose(box.width, box.height) or legend_box.y1 >= box.y0:
            raise AssertionError("UMAP axes must be square and legend must stay outside")
        layouts[filename] = dict(canvas_pixels=[2200, 2200], axes_box_aspect=float(box.height/box.width),
                                 xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), legend_outside_axes=True)
        # Do not use bbox_inches='tight': it would crop the square canvas to a rectangle.
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(figures_dir / filename, dpi=220)
        plt.close(fig)
    return layouts


def save_umap_plots(result_df: pd.DataFrame, probability_columns: list[str], figures_dir: Path) -> None:
    magnification = fitted_magnification(result_df)
    save_square_umap_scatter_plots(result_df, figures_dir)

    n = len(probability_columns)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.4 * nrows), squeeze=False)
    blue = (31 / 255, 119 / 255, 180 / 255)
    cmap = LinearSegmentedColormap.from_list("membership_blue", [(*blue, 0.06), (*blue, 1.0)])
    norm = Normalize(0.0, 1.0)
    for index, column in enumerate(probability_columns):
        ax = axes[index // ncols][index % ncols]
        sc = ax.scatter(result_df["umap_1"], result_df["umap_2"], c=result_df[column], s=10, cmap=cmap, norm=norm, edgecolors="none")
        ax.set_title(column.replace("P_phenotype_", "P(Phenotype ") + ")")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(sc, ax=ax, label="GMM posterior probability")
    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle(
        f"{magnification} morphology phenotype memberships\n"
        "Single hue; higher posterior is more opaque; YAP excluded",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "umap_membership_probabilities.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_summary_plots(result_df: pd.DataFrame, feature_columns: list[str], figures_dir: Path) -> None:
    feature_means = result_df.groupby("dominant_phenotype")[feature_columns].mean()
    overall_mean = result_df[feature_columns].mean()
    overall_std = result_df[feature_columns].std(ddof=0).replace(0, np.nan)
    heatmap = ((feature_means - overall_mean) / overall_std).replace([np.inf, -np.inf], np.nan).fillna(0)
    fig, ax = plt.subplots(figsize=(max(12, len(feature_columns) * 0.42), max(3.5, len(heatmap) * 0.7)))
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(heatmap)), labels=heatmap.index)
    ax.set_xticks(range(len(feature_columns)), labels=feature_columns, rotation=75, ha="right")
    ax.set_title("Morphology phenotype characterization (DAPI-only standardized means)")
    fig.colorbar(image, ax=ax, label="Mean difference / overall SD")
    fig.tight_layout()
    fig.savefig(figures_dir / "phenotype_feature_heatmap.png", dpi=220)
    plt.close(fig)

    save_yap_condition_boxplot(result_df, figures_dir)
    save_yap_condition_boxplot(result_df, figures_dir, raw=False)


def save_yap_condition_boxplot(result_df: pd.DataFrame, figures_dir: Path, raw: bool = True) -> pd.DataFrame:
    """One ratio definition for ALL conditions; never fill missing corrected ratios with raw values."""
    column = ("posthoc_yap_raw_log2_nuclear_perinuclear_ratio" if raw
              else "posthoc_yap_log2_nuclear_perinuclear_ratio")
    mode = "UNCORRECTED" if raw else "Background-corrected"
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    positions, data, labels, box_groups, rows = [], [], [], [], []
    tick_positions = []
    pos = 1
    palette = EXPERIMENTAL_GROUP_COLORS
    density_order = [2_500.0, 5_000.0, 7_500.0, 10_000.0]
    for experimental_group in ["Ctrl", "HA1", "HA2"]:
        for density in density_order:
            subset = result_df.loc[
                result_df["experimental_group_label"].eq(experimental_group)
                & result_df["seeding_density_cells_per_cm2"].eq(density)]
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            tick_positions.append(pos)
            labels.append(f"{experimental_group}\n{density / 1000:g}k\nn={len(values)}/{len(subset)}")
            rows.append(dict(experimental_group_label=experimental_group,
                             seeding_density_cells_per_cm2=density,
                             ratio_definition=mode, source_column=column,
                             n_model_cells=len(subset), n_available=len(values),
                             n_image_fields=subset["image_id"].nunique(),
                             median_log2_ratio=float(np.median(values)) if len(values) else np.nan))
            if len(values):
                positions.append(pos)
                data.append(values)
                box_groups.append(experimental_group)
            else:
                ax.text(pos, .04, "NA", transform=ax.get_xaxis_transform(), ha="center", color="gray")
            pos += 1
        pos += 1
    box = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False) if data else {"boxes": []}
    for patch, group in zip(box["boxes"], box_groups):
        patch.set_facecolor(palette[group])
        patch.set_alpha(0.75)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(tick_positions, labels=labels, fontsize=9)
    ax.set_xlim(.4, tick_positions[-1] + .6)
    ax.set_xlabel("HA experimental group / seeding density (thousands of cells/cm²)")
    ax.set_ylabel(f"{mode} log2(YAP nuclear/perinuclear)\nMedian intensity within each sampling region")
    ax.set_title(
        "YAP localization by HA group and seeding density\n"
        f"{mode} ratios for all conditions; post-hoc only"
    )
    note = "No background subtraction; additive background can affect comparisons." if raw else "Missing background gives NA; no substitution with uncorrected values."
    fig.text(.5, .025, note + "\n"
             "n = available / DAPI-model cells. Boxes: IQR; line: median; whiskers: 1.5×IQR; outliers hidden.\n"
             "Cell-level descriptive distributions, not replicate-level inference; perinuclear ring is a cytoplasmic proxy.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if raw else "_background_corrected"
    fig.savefig(figures_dir / f"yap_localization_by_ha_group_and_seeding_density{suffix}.png", dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def save_group_and_seeding_density_plots(result_df: pd.DataFrame, figures_dir: Path) -> None:
    magnification = fitted_magnification(result_df)
    density_order = [2_500.0, 5_000.0, 7_500.0, 10_000.0]
    density_colors = {
        2_500.0: "#440154",
        5_000.0: "#31688e",
        7_500.0: "#35b779",
        10_000.0: "#fde725",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), sharex=True, sharey=True)
    for ax, experimental_group in zip(axes, ["Ctrl", "HA1", "HA2"]):
        group_df = result_df[result_df["experimental_group_label"].eq(experimental_group)]
        sample_sizes = []
        for density in density_order:
            sub = group_df[group_df["seeding_density_cells_per_cm2"].eq(density)]
            sample_sizes.append(len(sub))
            ax.scatter(
                sub["umap_1"],
                sub["umap_2"],
                s=10,
                alpha=0.65,
                color=density_colors[density],
            )
        sizes = "/".join(str(value) for value in sample_sizes)
        ax.set_title(f"{experimental_group_display(experimental_group)}\nn(2.5k/5k/7.5k/10k)={sizes}")
        ax.set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=density_colors[value],
            label=f"{value / 1000:g} × 10³",
            markersize=7,
        )
        for value in density_order
    ]
    fig.legend(
        handles=legend_handles,
        title="Seeding density\n(cells/cm²)",
        loc="center left",
        bbox_to_anchor=(0.885, 0.5),
        borderaxespad=0.0,
    )
    fig.suptitle(
        "Seeding-density comparison within each HA experimental group on DAPI-only UMAP\n"
        "HA group and seeding density are post-hoc metadata; neither entered preprocessing, UMAP or GMM"
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.92))
    fig.savefig(figures_dir / "umap_posthoc_seeding_density_within_ha_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_phenotype_composition_plot(result_df, figures_dir)


def save_phenotype_composition_plot(result_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot composition with cell counts above bars and concise group labels below."""
    magnification = fitted_magnification(result_df)
    density_order = [2_500.0, 5_000.0, 7_500.0, 10_000.0]
    phenotype_order = sorted(
        result_df["dominant_phenotype"].unique(), key=lambda value: int(str(value).split()[-1])
    )
    colors = phenotype_colors(phenotype_order)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True)
    for ax, experimental_group in zip(axes, ["Ctrl", "HA1", "HA2"]):
        sub = result_df[result_df["experimental_group_label"].eq(experimental_group)]
        counts = (
            sub.groupby(["seeding_density_cells_per_cm2", "dominant_phenotype"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=density_order, columns=phenotype_order, fill_value=0)
        )
        fractions = counts.div(counts.sum(axis=1), axis=0)
        bottom = np.zeros(len(density_order), dtype=float)
        x = np.arange(len(density_order))
        for phenotype in phenotype_order:
            values = fractions[phenotype].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, color=colors[phenotype], label=phenotype)
            bottom += values
        totals = counts.sum(axis=1).to_numpy(dtype=int)
        ax.set_xticks(x, labels=[f"{value / 1000:g}k" for value in density_order])
        metadata = EXPERIMENTAL_GROUP_METADATA[experimental_group]
        group_label = ("Control (No HA)" if experimental_group == "Ctrl" else
                       f"HA-{experimental_group[-1]} ({metadata['ha_exposure_h']:g}h, "
                       f"{metadata['ha_concentration_nM']:g}nM)")
        ax.set_xlabel(group_label, fontsize=11, labelpad=10)
        ax.set_ylim(0, 1)
        for xpos, total in zip(x, totals):
            ax.text(xpos, 1.015, f"n={total}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, clip_on=False)
    axes[0].set_ylabel("Morphology phenotype fraction")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="DAPI-only phenotype")
    fig.suptitle(
        "Morphology phenotype composition across 3 HA groups × 4 seeding densities\n"
        f"Descriptive only: one {magnification} image field per combination"
    )
    fig.supxlabel("Seeding density (cells/cm²)", y=.015, fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .98))
    fig.savefig(figures_dir / "phenotype_composition_by_ha_group_and_seeding_density.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_tables(result_df: pd.DataFrame, feature_columns: list[str], tables_dir: Path) -> None:
    phenotype_rows = []
    for phenotype, sub in result_df.groupby("dominant_phenotype"):
        row = {
            "dominant_phenotype": phenotype,
            "n_cells": len(sub),
            "mean_max_gmm_posterior": sub["gmm_max_posterior"].mean(),
        }
        for column in feature_columns:
            row[f"{column}__mean"] = sub[column].mean()
            row[f"{column}__median"] = sub[column].median()
        phenotype_rows.append(row)
    pd.DataFrame(phenotype_rows).to_csv(tables_dir / "phenotype_dapi_feature_summary.csv", index=False, encoding="utf-8-sig")

    yap_summary = (
        result_df.groupby("dominant_phenotype", as_index=False)
        .agg(
            n_cells=("cell_id", "size"),
            n_valid_yap_ratio=("posthoc_yap_ratio_valid", "sum"),
            n_high_coverage_yap_ratio=("posthoc_yap_ring_coverage_pass", "sum"),
            median_yap_nuclear_perinuclear_ratio=("posthoc_yap_nuclear_perinuclear_ratio", "median"),
            median_yap_log2_nuclear_perinuclear_ratio=("posthoc_yap_log2_nuclear_perinuclear_ratio", "median"),
        )
    )
    yap_summary.to_csv(tables_dir / "phenotype_posthoc_yap_summary.csv", index=False, encoding="utf-8-sig")

    group_summary = (
        result_df.groupby(
            [
                "experimental_group_label",
                "experimental_group_description",
                "ha_exposure_h",
                "ha_concentration_nM",
                "seeding_density_folder_label",
                "seeding_density_cells_per_cm2",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_cells=("cell_id", "size"),
            n_valid_yap_ratio=("posthoc_yap_ratio_valid", "sum"),
            n_high_coverage_yap_ratio=("posthoc_yap_ring_coverage_pass", "sum"),
            median_yap_nuclear_perinuclear_ratio=("posthoc_yap_nuclear_perinuclear_ratio", "median"),
            median_yap_log2_nuclear_perinuclear_ratio=("posthoc_yap_log2_nuclear_perinuclear_ratio", "median"),
            median_max_gmm_posterior=("gmm_max_posterior", "median"),
        )
        .sort_values(["experimental_group_label", "seeding_density_cells_per_cm2"])
    )
    group_summary.to_csv(tables_dir / "ha_group_seeding_density_posthoc_yap_summary.csv", index=False, encoding="utf-8-sig")

    composition = (
        result_df.groupby(
            [
                "experimental_group_label",
                "ha_exposure_h",
                "ha_concentration_nM",
                "seeding_density_folder_label",
                "seeding_density_cells_per_cm2",
                "dominant_phenotype",
            ],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "n_cells"})
    )
    totals = composition.groupby(
        ["experimental_group_label", "seeding_density_cells_per_cm2"], dropna=False
    )["n_cells"].transform("sum")
    composition["phenotype_fraction"] = composition["n_cells"] / totals
    composition.to_csv(tables_dir / "ha_group_seeding_density_phenotype_composition.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()

    def stage(message: str) -> None:
        print(f"[stage +{time.perf_counter() - started:7.1f}s] {message}", flush=True)

    stage(f"YAP post-hoc algorithm: {ALGORITHM_VERSION}; pipeline: {Path(__file__).resolve()}")
    output_root = args.output_root.resolve()
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    ring_qc_dir = figures_dir / "ring_qc"
    phenotype_overlay_dir = figures_dir / "phenotype_overlays"
    yap_overlay_dir = figures_dir / "yap_ratio_overlays"
    for directory in (tables_dir, figures_dir, ring_qc_dir, phenotype_overlay_dir, yap_overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_sets, manifest = discover_image_sets(args.data_root)
    manifest.to_csv(tables_dir / "image_manifest.csv", index=False, encoding="utf-8-sig")
    design_rows = []
    for experimental_group in ["Ctrl", "HA1", "HA2"]:
        group_metadata = EXPERIMENTAL_GROUP_METADATA[experimental_group]
        for folder_label, density in SEEDING_DENSITY_CELLS_PER_CM2.items():
            design_rows.append(
                {
                    "experimental_group_label": experimental_group,
                    **group_metadata,
                    "seeding_density_folder_label": folder_label,
                    "seeding_density_cells_per_cm2": density,
                }
            )
    pd.DataFrame(design_rows).to_csv(
        tables_dir / "experimental_design.csv", index=False, encoding="utf-8-sig"
    )
    selected_sets = [item for item in all_sets if item.magnification.lower() == args.fit_magnification.lower()]
    if not selected_sets:
        raise RuntimeError(f"No complete {args.fit_magnification} image sets found")
    stage(f"discovered {len(all_sets)} complete image sets; selected {len(selected_sets)} at {args.fit_magnification}")

    mask_paths = segment_image_sets(selected_sets, output_root, use_gpu=not args.cpu, reuse_masks=args.reuse_masks)
    stage("segmentation complete")
    all_features = extract_all_dapi_features(selected_sets, mask_paths, edge_buffer_px=args.edge_buffer_px)
    all_features.to_csv(tables_dir / "dapi_features_all_cells.csv", index=False, encoding="utf-8-sig")
    dapi_qc_summary = (
        all_features.groupby(
            ["experimental_group_label", "seeding_density_cells_per_cm2", "magnification", "image_id"],
            as_index=False,
        )
        .agg(n_segmented=("cell_id", "size"), n_qc_keep=("qc_keep", "sum"))
    )
    dapi_qc_summary["qc_keep_fraction"] = dapi_qc_summary["n_qc_keep"] / dapi_qc_summary["n_segmented"]
    dapi_qc_summary.to_csv(tables_dir / "qc_summary.csv", index=False, encoding="utf-8-sig")
    fit_df = all_features[all_features["qc_keep"]].reset_index(drop=True)
    if len(fit_df) < 20:
        raise RuntimeError("Fewer than 20 QC-kept nuclei are available")

    feature_switches = getattr(args, "feature_switches", None)
    enabled_features, disabled_features = CORE.resolve_model_features(
        "raw" if args.feature_set == "baseline" else "augmented", feature_switches
    )
    stage(f"feature switches: enabled={len(enabled_features)}, disabled={len(disabled_features)}")
    print("[features enabled] " + ", ".join(enabled_features), flush=True)
    print("[features disabled] " + (", ".join(disabled_features) if disabled_features else "none"), flush=True)
    stage(f"preprocessing {len(fit_df)} cells using selected DAPI-derived features")
    comparison, comparison_artifacts = CORE.compare_feature_models(
        fit_df, feature_switches=feature_switches
    )
    selected_feature_set = "raw" if args.feature_set == "baseline" else "augmented"
    scaled, feature_columns, _, raw_labels, raw_probabilities, pca, selection, selected_k = comparison_artifacts[selected_feature_set]
    preprocess_info = comparison[selected_feature_set]["preprocessing"]
    pd.DataFrame([
        {"feature_set": name, **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}}
        for name, report in comparison.items() if name in {"raw", "augmented"}
    ]).to_csv(tables_dir / "raw_vs_composite_model_comparison.csv", index=False, encoding="utf-8-sig")
    save_json(comparison, tables_dir / "raw_vs_composite_model_comparison.json")
    validate_morphology_feature_names(feature_columns)
    stage("dynamic GMM search")
    ranks, phenotype_labels, ordered_raw_ids, probabilities = CORE.reorder_phenotypes_by_size(raw_labels, raw_probabilities)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("GMM posterior probabilities do not sum to one")
    stage(f"GMM complete; selected K={selected_k}")
    stage("UMAP visualization")
    if args.skip_umap:
        umap_values = pca.transform(scaled)[:, :2]
        stage("UMAP skipped by explicit flag; PCA1/2 used only as plotting coordinates")
    else:
        umap_values = compute_umap_for_dataset(scaled)
        stage("UMAP complete")

    result_df = fit_df.copy()
    pca_values = pca.transform(scaled)
    result_df["pca_1"] = pca_values[:, 0]
    result_df["pca_2"] = pca_values[:, 1]
    result_df["umap_1"] = umap_values[:, 0]
    result_df["umap_2"] = umap_values[:, 1]
    result_df["gmm_component_raw"] = raw_labels
    result_df["dominant_phenotype_rank"] = ranks + 1
    result_df["dominant_phenotype"] = phenotype_labels
    result_df["gmm_max_posterior"] = probabilities.max(axis=1)
    probability_columns = []
    for index in range(selected_k):
        column = f"P_phenotype_{index + 1}"
        result_df[column] = probabilities[:, index]
        probability_columns.append(column)

    # The complete morphology model is finalized before YAP pixels are opened.
    stage("post-hoc YAP nuclear/perinuclear measurement")
    yap_config = load_config(getattr(args, "yap_qc_config", None))
    yap_tables = []
    ring_info = []
    ring_labels_by_image: dict[str, np.ndarray] = {}
    for item in selected_sets:
        table, ring_labels, info = measure_yap_posthoc(item, mask_paths[item.image_id], yap_config,
                                                    getattr(args, "yap_background_roi_dir", None))
        yap_tables.append(table)
        ring_info.append(info)
        ring_labels_by_image[item.image_id] = ring_labels
        save_ring_qc_overlay(item, mask_paths[item.image_id], ring_labels, info, ring_qc_dir / f"{item.image_id}_ring_qc.png", table)
    yap_df = pd.concat(yap_tables, ignore_index=True)
    yap_df.to_csv(tables_dir / "yap_nuclear_perinuclear_measurements.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ring_info).to_csv(tables_dir / "yap_ring_and_background_qc.csv", index=False, encoding="utf-8-sig")
    result_df = result_df.merge(yap_df, on=["cell_id", "image_id", "label"], how="left", validate="one_to_one")
    qc_summary(result_df).to_csv(tables_dir / "yap_technical_qc_summary.csv", index=False, encoding="utf-8-sig")
    save_qc_summary_figure(result_df, figures_dir / "yap_qc_summary.png")

    selection.to_csv(tables_dir / "gmm_model_selection.csv", index=False, encoding="utf-8-sig")
    CORE.save_selection_plot(selection, selected_k, "YAP localization dataset", args.fit_magnification, figures_dir / "gmm_model_selection.png")
    save_umap_plots(result_df, probability_columns, figures_dir)
    save_summary_plots(result_df, feature_columns, figures_dir)
    save_group_and_seeding_density_plots(result_df, figures_dir)
    summarize_tables(result_df, feature_columns, tables_dir)
    for item in selected_sets:
        save_phenotype_overlay(item, mask_paths[item.image_id], result_df, phenotype_overlay_dir / f"{item.image_id}_phenotypes.png")
        save_yap_ratio_overlay(item, mask_paths[item.image_id], result_df, yap_overlay_dir / f"{item.image_id}_yap_ratio.png")
        save_yap_ratio_overlay(item, mask_paths[item.image_id], result_df,
                              figures_dir / "yap_ratio_raw_uncorrected" / f"{item.image_id}_yap_ratio_raw.png", raw=True)

    result_df.to_csv(tables_dir / "model_a_single_cell_results.csv", index=False, encoding="utf-8-sig")
    for table_name, table in build_descriptive_tables(result_df).items():
        table.to_csv(tables_dir / f"{table_name}.csv", index=False, encoding="utf-8-sig")
    valid_yap = int(result_df["posthoc_yap_ratio_valid"].fillna(False).sum())
    verification = {
        "yap_excluded_from_model_features": not any("yap" in column.lower() or "af488" in column.lower() for column in feature_columns),
        "yap_measured_only_after_gmm_and_umap": True,
        "posterior_rows_sum_to_one": bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)),
        "posterior_column_count_equals_selected_k": len(probability_columns) == selected_k,
        "experimental_metadata_excluded_from_model_features": not any(
            token in column.lower()
            for column in feature_columns
            for token in ("experimental_group", "ha_", "seeding_density")
        ),
        "replicate": "not_provided",
    }
    if not all(
        verification[key]
        for key in (
            "yap_excluded_from_model_features",
            "yap_measured_only_after_gmm_and_umap",
            "posterior_rows_sum_to_one",
            "posterior_column_count_equals_selected_k",
            "experimental_metadata_excluded_from_model_features",
        )
    ):
        raise AssertionError(f"Verification failed: {verification}")
    run_info = {
        "scope": (
            f"All complete {args.fit_magnification} image sets from Ctrl/HA1/HA2 across "
            "seeding densities 2.5e3/5e3/7.5e3/1e4 cells/cm²"
        ),
        "data_root": args.data_root.resolve(),
        "output_root": output_root,
        "feature_set": args.feature_set,
        "edge_qc": {
            "buffer_px": args.edge_buffer_px,
            "keep_rule": "minimum mask-bbox distance to image edge > buffer_px",
            "spatial_context": "QC-kept nuclei only; excluded nuclear morphology is not used by neighbors",
            "raw_masks_preserved": True,
            "limitation": "Conservative image-boundary tolerance, not a validated completeness classifier; kNN remains field-of-view censored near edges",
        },
        "dapi_input": "8-bit grayscale *_DAPI_ORG.png exports",
        "yap_input": "8-bit grayscale *_AF488_ORG.png exports; interpreted as YAP because of dataset identity, pending antibody record",
        "n_complete_image_sets_discovered": len(all_sets),
        "n_image_sets_fitted": len(selected_sets),
        "n_segmented_cells": len(all_features),
        "n_qc_cells": len(result_df),
        "n_valid_posthoc_yap_ratios": valid_yap,
        "n_yap_sampling_available": int(result_df["posthoc_yap_sampling_valid"].sum()),
        "n_raw_uncorrected_yap_ratios": int(result_df["posthoc_yap_raw_ratio_valid"].sum()),
        "model_feature_columns": feature_columns,
        "feature_switches": feature_switches or {name: True for name in CORE.ALL_SWITCHABLE_MODEL_FEATURES},
        "configured_disabled_model_features": disabled_features,
        "preprocessing": preprocess_info,
        "gmm": {
            "candidate_k": selection["n_clusters"].astype(int).tolist(),
            "selected_k": selected_k,
            "selection_metric": "bic_penalized (validated existing repository method)",
            "ordered_raw_component_ids_by_size": ordered_raw_ids,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            "selected_k_at_search_upper_bound": selected_k == int(selection["n_clusters"].max()),
            "fraction_max_posterior_below_0_8": float((probabilities.max(axis=1) < 0.8).mean()),
        },
        "raw_vs_composite_model_comparison": comparison,
        "umap": {
            "role": "visualization only", "input": "DAPI-only preprocessed features",
            "skipped": bool(args.skip_umap),
            "coordinate_source": "PCA1/2 validation placeholder" if args.skip_umap else "UMAP",
        },
        "posthoc_yap": {
            "algorithm_version": ALGORITHM_VERSION,
            "measurement": "background-corrected nuclear median / non-overlapping perinuclear-ring median",
            "classification_threshold": "none; continuous ratio retained",
            "ring_width_rule": "default inner gap 0; outer distance 0.20 median nuclear diameter; all segmented DAPI nuclei excluded; 0.35 wide-ring sensitivity saved",
            "background_rule": "user cell-free ROI or inferred far-from-nuclei median + robust noise; insufficient background gives NaN, no percentile fallback",
            "qc_config": yap_config.__dict__,
            "qc_interpretation": "any nonempty nuclear-excluded ring measured; coverage, clipping and noise are warnings only; ratio_valid means corrected ratio is mathematically available, not validated",
            "raw_ratios": "stored and plotted separately; no background fallback or mixing with corrected ratios",
            "limitations": "perinuclear enrichment proxy, not a true whole-cell cytoplasmic measurement; no membrane/cytoplasm marker",
            "used_for_preprocessing_umap_or_gmm": False,
        },
        "metadata_interpretation": {
            "source": "user-supplied experimental-design diagram",
            "Ctrl": "Control (No HA); HA concentration 0 nM; exposure duration not applicable",
            "HA1": "HA-1: 72 h exposure at 2.5 nM",
            "HA2": "HA-2: 48 h exposure at 5 nM",
            "folder_numeric_labels": "seeding density in thousands of cells/cm²",
            "seeding_density_cells_per_cm2": [2500, 5000, 7500, 10000],
            "replicate": "not_provided",
            "design_limitation": "HA1 versus HA2 changes both HA concentration and exposure time, so concentration and time effects cannot be isolated",
            "inference_limit": (
                f"one {args.fit_magnification} image field per HA-group/seeding-density combination; "
                "cell-level p-values would be pseudoreplication"
            ),
        },
        "core_model_a_source": CORE_PATH,
        "verification": verification,
    }
    save_json(run_info, output_root / "run_info.json")
    stage("all tables and figures written")
    # Keep stdout portable on Windows consoles that still use a GBK code page.
    print(json.dumps(json_ready(run_info), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}")
        raise
