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

- `Suzui/DAPI/`: 原根目录中的 Suzui 选焦、分割、特征/QC、聚类、推理、SA/SHAP 和时序 GIF 脚本。上文列出的编号脚本均已移到此处。
- `Suzui/DAPI_OCT4/`: 原 `double_work_build`；Suzui 配对染色对照，复用相邻 DAPI 实现，详见 [Suzui 流程说明](Suzui/README.md)。
- `Ekin_DAPI_OCT4/`: Ekin feeder-free Model A，OCT4 仅在聚类后按 BASC 二元表征，与 Suzui 配对染色流程分开。
- `Ekin_DAPI_YAP/`: Model A，YAP 仅在聚类后以连续核/核周比值表征。
- `shared/`: 公共 DAPI 模型实现和入口配置，不重复复制算法。
- 公式见 `Ekin_DAPI_OCT4/COMPOSITE_FEATURES.md`。

### 命名与运行约定

- 按来源与通道/流程组织，不携带算法版本；单流程可用 `Ekin_DAPI_OCT4`，有多个共享流程时采用 `Suzui/DAPI`、`Suzui/DAPI_OCT4`。
- 两个分析目录都有 `run_baseline.py` 和 `run_composite.py`，可直接在 PyCharm 右键运行。
- `baseline` 使用基线形态、DAPI 强度和邻域特征；`composite` 在此基础上增加无量纲生物学相关代理。两者都计算对照指标，但采用指定方案生成最终结果。
- 入口顶部 `MAGNIFICATION` 可设为 `20x` 或 `40x`，也可通过运行参数覆盖。方案、倍率及 OCT4 样本分别存放结果，默认不互相覆盖；同配置重跑更新同一路径，不同数据集应指定独立 `--output-root`。
- 数据来源、通道、流程、特征方案与软件版本分开表示；代码版本通过 Git 提交追踪，不再用 1.x/2.x 暗示特征类型。

PyCharm 解释器请选已有的 `cellpose` 环境。命令行示例：`python Ekin_DAPI_YAP/run_baseline.py --fit-magnification 20x`。旧 `.ps1` 和试验入口保留兼容，默认 composite，继续采用它们原有的输出路径；推荐日常使用两个新 Python 入口。

本次没有重算历史结果或改写其绝对路径。Git 不包含生成结果、掩膜和模型缓存；其他工作树/原项目中已有的旧结果目录保留，不自动搬迁或删除。

Both Model A workflows compare raw and augmented feature models, search K=2–12, and retain all GMM posteriors. 20x and 40x are fitted independently; equal phenotype numbers across magnifications do not imply the same phenotype. Experimental group, treatment time/concentration, seeding density, and other metadata never enter the morphology model.
