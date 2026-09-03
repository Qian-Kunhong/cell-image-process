# Ekin_DAPI_OCT4/legacy — 旧 DAPI/OCT4 双染流程

本流程使用既有 DAPI 核形态与邻域特征，不是新版 Model A 的 `baseline` 入口。新版入口在上一级目录。旧算法独立保留，目录名不再携带软件版本。

This folder contains a DAPI + Oct-4 workflow that is intentionally separated from the original single-stain scripts.

## Main rules

- DAPI is used for segmentation, morphology, and neighborhood features.
- Oct-4 is used for intensity features only.
- DAPI intensity is not used for clustering or supervised inference.
- The same DAPI-derived mask is reused on the paired Oct-4 image.

## Scripts

- `02_segment_dapi_double.py`
  - Segment nuclei from DAPI images only.
- `03_extract_dapi_oct4_features.py`
  - Extract DAPI morphology/neighborhood features.
  - Extract Oct-4 intensity features with the same masks.
  - Output:
    - `nucleus_features.csv` -> DAPI features only
    - `nucleus_intensity_features.csv` -> Oct-4 intensity only
  - Automatically calls `03-1_qc_filter_double.py` by default.
- `03-1_qc_filter_double.py`
  - Reuses the validated QC logic from the original workflow.
  - It is a worker script only: `03_extract_dapi_oct4_features.py` decides the dataset and passes explicit directories into it.
  - Original masks are preserved.
  - QC-kept and QC-removed masks are exported separately under `qc_training/`.
- `04_cluster_dapi_oct4.py`
  - Cluster training data from DAPI features.
  - Save DAPI deviation score vs Oct-4 mean intensity.
- `05_infer_dapi_oct4.py`
  - Train/infer from DAPI features only.
  - Save predicted DAPI deviation score vs Oct-4 mean intensity on new data.

## Pairing assumption

The scripts pair DAPI, Oct-4, and mask files by a normalized filename key that removes channel tokens such as `DAPI`, `Oct-4`, and `TexasRed`.

Examples that should pair:

- `A-1_fld001_time001_DAPI_bestwix1.tif`
- `A-1_fld001_time001_Oct-4_bestwix1.tif`
- `A-1_fld001_time001_DAPI_bestwix1_mask.npy`

For your current paper dataset, `TexasRed` is treated as the Oct-4 channel.

## First things to edit

At the top of the main scripts, the simplest mode-1 edit is:

- `DATASET_NAME`
- optional: `OUTPUT_DATASET_NAME`

The scripts will derive the input path as `SUZUI_ROOT / DATASET_NAME` and the output path as `ANALYSIS_ROOT / OUTPUT_DATASET_NAME`.

You can still override explicit paths if needed, but usually you do not need to.

Other common settings:
- `mode`
- `TRAINING_SET_NAME`
- `DAPI_IMAGE_DIR`
- `OCT4_IMAGE_DIR`
- `MASK_DIR`
- `OUT_DIR`

If DAPI and Oct-4 images are stored in the same folder, you can point both channel directories to the same path.

For `03-1_qc_filter_double.py`, there is intentionally no second mode switch anymore.
