"""QC figures: separate raw image, fixed ratio scale, explicit missing values."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


def save_ratio_figure(merge, mask, table, title, path: Path, raw=False):
    norm = Normalize(-2, 2, clip=True)  # Shared across all fields; not biological cutoffs.
    cmap = plt.get_cmap("coolwarm")
    canvas = np.full((*mask.shape, 3), 0.04)
    canvas[mask > 0] = 0.20  # Not in morphology result, not ratio=0.
    keep = set(table.label.astype(int))
    valid_labels = set()
    for row in table.itertuples(index=False):
        value = (row.posthoc_yap_raw_log2_nuclear_perinuclear_ratio if raw
                 else row.posthoc_yap_log2_nuclear_perinuclear_ratio)
        passed = np.isfinite(value)
        canvas[mask == row.label] = cmap(norm(value))[:3] if passed else (0.53, 0.53, 0.53)
        if passed:
            valid_labels.add(int(row.label))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    axes[0].imshow(merge)
    axes[0].set_title("Original merge (display only)")
    axes[1].imshow(canvas)
    for prop in regionprops(mask):
        if prop.label in keep and prop.label not in valid_labels:
            axes[1].plot(prop.centroid[1], prop.centroid[0], "x", color="white", ms=3, mew=.6)
    mode = "UNCORRECTED" if raw else "Background-corrected"
    axes[1].set_title(f"{mode} ratio available: {len(valid_labels)}/{len(table)} cells")
    for ax in axes:
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.88, top=.86, bottom=.14, wspace=.025)
    cax = fig.add_axes([.90, .20, .015, .60])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    bar.set_label(f"log2(YAP nuclear/perinuclear); median, {mode}")
    fig.legend(handles=[Patch(facecolor="0.53", label="Gray + X: this ratio unavailable (not low YAP)"),
                        Patch(facecolor="0.20", label="Dark gray: excluded from DAPI model")],
               loc="lower center", ncol=2, bbox_to_anchor=(.47, .07), frameon=False)
    note = "UNCORRECTED: affected by additive background; do not mix with corrected values" if raw else "No hard support/SNR gates; availability is not measurement validation"
    fig.suptitle(title + "\nPost-hoc only; ALL DAPI nuclei excluded. " + note, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_comparison_figure(mask, table, title, path):
    cmap, norm = plt.get_cmap("coolwarm"), Normalize(-2, 2, clip=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    columns = ["posthoc_yap_legacy_v1_ratio", "posthoc_yap_nuclear_perinuclear_ratio", "posthoc_yap_raw_nuclear_perinuclear_ratio"]
    titles = ["V1 wide ring + old background (audit only)", "V3 DAPI-excluded: background-corrected", "V3 DAPI-excluded: UNCORRECTED"]
    for ax, column, subtitle in zip(axes, columns, titles):
        canvas = np.full((*mask.shape, 3), .04)
        canvas[mask > 0] = .2
        for row in table.itertuples(index=False):
            ratio = getattr(row, column)
            canvas[mask == row.label] = cmap(norm(np.log2(ratio)))[:3] if np.isfinite(ratio) and ratio > 0 else (.53, .53, .53)
        ax.imshow(canvas)
        ax.set_title(subtitle, fontsize=11)
        ax.axis("off")
    fig.subplots_adjust(left=.01, right=.91, top=.84, bottom=.13, wspace=.025)
    cax = fig.add_axes([.935, .20, .012, .55])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both", label="log2(nuclear/perinuclear)")
    fig.suptitle(title + "\nSame cells, unchanged DAPI model. Fixed color scale. Gray = unavailable, not log2(ratio)=0.", fontsize=12)
    fig.text(.45, .045, "Middle needs background; right does NOT subtract background. Do not pool these different ratio definitions.", ha="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_qc_summary_figure(table, path):
    fields = table[["image_id", "experimental_group_label", "seeding_density_cells_per_cm2"]].drop_duplicates()
    fields = fields.sort_values(["experimental_group_label", "seeding_density_cells_per_cm2"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ticklabels = []
    for x, row in enumerate(fields.itertuples(index=False)):
        cells = table[table.image_id.eq(row.image_id)]
        valid = cells[cells.posthoc_yap_ratio_valid]
        vals = valid.posthoc_yap_log2_nuclear_perinuclear_ratio.dropna()
        ticklabels.append(f"{row.experimental_group_label}\n{row.seeding_density_cells_per_cm2/1000:g}k")
        if len(vals):
            axes[0].boxplot([vals], positions=[x], widths=.55, showfliers=False)
        else:
            axes[0].text(x, .05, "NA", ha="center", color="gray", transform=axes[0].get_xaxis_transform())
        fraction = len(valid)/len(cells)
        raw_n = int(cells.posthoc_yap_raw_ratio_valid.sum())
        axes[1].bar(x-.17, raw_n/len(cells), width=.32, color="#81b7bd", label="Raw ratio available" if x == 0 else None)
        axes[1].bar(x+.17, fraction, width=.32, color="#3b8897", label="Corrected ratio available" if x == 0 else None)
        axes[1].text(x, max(fraction, raw_n/len(cells))+.04, f"{raw_n} / {len(valid)}\nof {len(cells)}", ha="center", fontsize=7)
    axes[0].axhline(0, color="gray", ls=":")
    axes[0].set_ylabel("log2 nuclear/perinuclear ratio\nBackground-corrected only")
    axes[0].set_title("Descriptive per-field distributions; NA = not reportable, not zero")
    axes[1].set_ylim(0, 1.25)
    axes[1].set_ylabel("Ratio availability fraction")
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, -.44), ncol=2, frameon=False)
    axes[1].set_xticks(range(len(ticklabels)), ticklabels)
    axes[1].set_xlabel("HA group / seeding density (thousands of cells per cm²)")
    fig.suptitle("YAP v3: all DAPI nuclei excluded; any nonempty sampling region retained\nNo manual-validation requirement. Raw and background-corrected ratios are NOT interchangeable.")
    fig.tight_layout(rect=(0, .06, 1, .93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_sampling_figure(yap, mask, rings, table, info, path: Path):
    # Contrast adjustment affects visualization only, never measurement.
    low, high = np.percentile(yap, [1, 99])
    gray = np.clip((yap.astype(float)-low) / max(high-low, 1), 0, 1)
    canvas = np.repeat(gray[..., None], 3, axis=-1)
    good = rings > 0
    canvas[good] = .55*canvas[good] + .45*np.array([.0, 1., .25])
    canvas[find_boundaries(mask, mode="inner")] = [0., .6, 1.]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("AF488 export (display contrast only)")
    axes[1].imshow(canvas)
    axes[1].set_title(f"All segmented nuclei + full sampling rings ({info['inner_gap_px']} to {info['outer_distance_px']} px)")
    if info["background_method"].startswith("inferred"):
        import json
        fraction = json.loads(info["config_json"])["background_distance_fraction"]
        far = distance_transform_edt(mask == 0) > max(2, int(round(fraction*info["median_nuclear_diameter_px"])))
        if far.any() and not far.all():
            axes[0].contour(far.astype(float), levels=[.5], colors=["#ffd22b"], linewidths=.5)
    for ax in axes:
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.85, bottom=.15, wspace=.03)
    fig.legend(handles=[Patch(facecolor="#00ff40", label="Sampling pixels: outside EVERY segmented nucleus"),
                        Patch(facecolor="#0099ff", label="All nuclear boundaries"),
                        Patch(facecolor="#ffd22b", label="Inferred background boundary (where available)")],
               loc="lower center", ncol=2, bbox_to_anchor=(.5, .04), frameon=False)
    fig.suptitle(f"{info['image_id']}: DAPI-excluded YAP sampling\n"
                 f"Background adequate: {info['background_adequate']}; B={info['background']:.2f}, "
                 f"noise scale={info['background_noise_scale_dn']:.2f} DN. No dark pixels removed from rings.", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
