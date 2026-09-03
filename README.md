# Cell Image Processing and Phenotype Analysis

A microscopy image-analysis pipeline developed for my graduate research on quantitative evaluation of cultured cell states.

The project combines **image processing, cell/nucleus segmentation, feature engineering, unsupervised learning, supervised machine learning, sensitivity analysis, and model interpretation** to characterize cell-state heterogeneity from microscopy images.

## 日本語概要

大学院での細胞培養研究に使用している画像解析・機械学習パイプラインです。顕微鏡画像から細胞核を分割し、核形態や局所的な細胞密度・近傍構造を数値化したうえで、教師なし学習による表現型分類、XGBoostによる状態予測、Sobol感度分析・SHAPによる特徴量解析を行います。

## Research Workflow

```text
Microscopy images
      ↓
Focus-quality selection
      ↓
Cellpose nucleus segmentation
      ↓
QC and feature extraction
      ↓
Unsupervised phenotype discovery
(PCA / GMM / UMAP)
      ↓
Supervised classification
(XGBoost)
      ↓
Feature importance / interpretation
(Sobol sensitivity analysis / SHAP)
```

## Main Components

### 1. Focus selection
`01_select_focus_and_copy_bf.py`

- Groups microscopy images by acquisition position and time point
- Evaluates image sharpness using a Tenengrad/Sobel-based focus score
- Selects high-quality focal planes for downstream analysis
- Matches corresponding bright-field images

### 2. Nucleus segmentation
`02_cellpose_segment.py`

- Loads fluorescence microscopy images
- Performs nucleus/cell segmentation with **Cellpose**
- Supports GPU inference
- Saves segmentation masks and preview images

### 3. Quality control and feature extraction
`03-1_qc_filter.py`  
`03_feature_extractior.py`

Extracts nucleus-level and image-level features, including:

- Nuclear area and shape
- Circularity / eccentricity / aspect-ratio related features
- Intensity-related features
- Nearest-neighbor distances
- k-nearest-neighbor statistics
- Local cell density
- Adaptive neighborhood features

The spatial features are designed to represent not only an individual nucleus but also its local cellular environment.

### 4. Unsupervised phenotype analysis
`04_unsupervised_learning.py`  
`04-b_adaptive_clustering.py`  
`04-c_deviation_intensity_analysis.py`

- Robust preprocessing and dimensionality reduction
- **PCA** and optional **UMAP** visualization
- **Gaussian Mixture Model (GMM)** / clustering-based phenotype discovery
- Generation of pseudo-labels for downstream supervised learning
- Comparison of discovered phenotypes with fluorescence intensity measurements

### 5. Supervised cell-state classification
`05_distinguish cell type.py`  
`05-b_relative_batch_normalized.py`

- Trains an **XGBoost** classifier using phenotype labels generated in the clustering stage
- Applies the trained model to new microscopy datasets
- Handles image-level normalization and batch-related variation
- Outputs nucleus-level prediction probabilities and visualization results

### 6. Sensitivity analysis
`06_SA.py`

- Performs **Sobol global sensitivity analysis** using SALib
- Evaluates how strongly individual morphological and spatial features influence model predictions
- Exports first-order and total-order sensitivity indices

### 7. Model interpretation
`07_SHAP.py`

- Performs **SHAP analysis** on the XGBoost model
- Ranks influential image-derived features
- Generates summary and feature-importance plots for interpretation of model decisions

## Technology Stack

- **Language:** Python
- **Image processing:** OpenCV, tifffile, scikit-image
- **Segmentation:** Cellpose
- **Data processing:** NumPy, pandas, SciPy
- **Machine learning:** scikit-learn, XGBoost
- **Dimensionality reduction / clustering:** PCA, GMM, UMAP
- **Sensitivity analysis:** SALib / Sobol method
- **Model interpretation:** SHAP
- **Visualization:** Matplotlib

## Example Feature Types

The pipeline converts microscopy images into quantitative descriptors such as:

```text
Nuclear morphology
  ├─ area
  ├─ perimeter
  ├─ circularity
  ├─ eccentricity
  └─ aspect ratio

Spatial environment
  ├─ nearest-neighbor distance
  ├─ kNN distance statistics
  ├─ local cell density
  ├─ neighbor count
  └─ neighborhood morphology statistics
```

These features are then used to investigate relationships between cell morphology, local cellular organization, and experimentally measured cell-state indicators.

## Repository Notes

This repository contains **research-oriented analysis code** rather than a packaged software library.

- Experimental microscopy datasets are not included in the repository.
- Several scripts currently contain local path settings that should be changed before running on another environment.
- Parameters are adjusted according to the microscopy dataset and experimental conditions.
- The pipeline is under active development as the research progresses.

## Purpose

The main goal of this project is to explore how image-derived morphological and spatial information can be converted into quantitative features and used with machine-learning methods to evaluate heterogeneous cell states.

## DAPI-only Model A workflows

- `Ekin_DAPI_OCT_1.0/`: 既有 DAPI 核形态/邻域特征流程，配对 OCT4 强度用于后续对照。
- `Ekin_DAPI_OCT_1.1/`: feeder-free DAPI Model A；OCT4 仅在聚类后按 BASC 二元表征。
- `Ekin_DAPI_YAP_2.0/`: DAPI 复合特征 Model A；YAP 仅在聚类后以连续核/核周比值表征。
- 复合特征公式、量纲、方向和代理解释见 `Ekin_DAPI_OCT_1.1/COMPOSITE_FEATURES.md`。

### 版本命名约定

- **1.x：基于原始特征的基线系列。** 原始特征包含已提取的核形态、强度或邻域统计，不是原始像素。
- **2.x：基于生物学相关自适应复合特征的系列。** 在原始特征基础上加入无量纲形态/邻域代理，仍保留原始模型作对照；这些代理不等同于细胞周期、多能性或分化标签。

当前实现的例外：1.1 已接入原始/复合特征对照，并默认输出增强模型结果，同时是 2.0 的共享核心。因此版本号表示系列来源，不能严格用来判定是否仅采用原始特征。本次重命名没有改变算法或重算结果。

原目录 `double_work_build`、`Hochest_OCT4_1.0`、`YAP_DAPI_1.0` 已分别更名为以上三个目录。既有输出和掩膜随目录保留；旧 `run_info.json` 中的绝对路径属于历史运行记录，未改写。

The 1.1 and 2.0 Model A workflows compare raw and augmented feature models, search K=2–12, and retain all GMM posteriors. 20x and 40x are fitted independently; equal phenotype numbers across magnifications do not imply the same phenotype. Experimental group, treatment time/concentration, seeding density, and other metadata never enter the morphology model.
