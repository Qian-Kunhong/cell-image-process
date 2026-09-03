细胞图像分析项目：文件结构与使用总说明
更新日期：2026-09-03
======================================================================

一、先选数据来源，再选流程

Suzui 数据：Suzui/DAPI/ 是常规 DAPI、训练集及时间序列分析；
Suzui/DAPI_OCT4/ 是配对 DAPI/OCT4 对照（原 double_work_build）。
两者默认根目录都是 F:\Suzui，但不是同一批图像或完全相同的配置。

Ekin 数据：Ekin_DAPI_OCT4/ 和 Ekin_DAPI_YAP/ 是两条 DAPI-only Model A
流程，分别以 OCT4 或 YAP 作聚类后表征，不与 Suzui 双染流程混用。

二、正式文件结构与作用（不列生成结果及缓存）

cell image/
|-- README.txt                         本总说明，优先阅读
|-- README.md                          仓库首页索引
|-- .gitignore                         排除结果、掩膜、缓存
|-- Suzui/
|   |-- README.md                      Suzui 专项说明
|   |-- DAPI/
|   |   |-- 01_select_focus_and_copy_bf.py  选焦、匹配明场图
|   |   |-- 02_cellpose_segment.py      Cellpose 核分割
|   |   |-- 03_feature_extractior.py    核形态/强度/邻域特征提取
|   |   |-- 03-1_qc_filter.py           分割与特征 QC
|   |   |-- 04_unsupervised_learning.py 无监督聚类主流程
|   |   |-- 04-b_adaptive_clustering.py 自适应聚类变体
|   |   |-- 04-c_deviation_intensity_analysis.py  偏离分数与强度对照
|   |   |-- 05_distinguish cell type.py XGBoost 训练与新图像推理
|   |   |-- 05-b_relative_batch_normalized.py    相对批次归一化入口
|   |   |-- 06_SA.py                    Sobol 敏感性分析
|   |   |-- 07_SHAP.py                  SHAP 特征贡献分析
|   |   |-- cluster_xgb_analysis_common.py      SA/SHAP 共用实现
|   |   `-- make_timelapse_gifs.py      时间序列结果 GIF
|   `-- DAPI_OCT4/
|       |-- 02_segment_dapi_double.py  只用 DAPI 分割
|       |-- 03_extract_dapi_oct4_features.py    同一掩膜提取两通道信息
|       |-- 03-1_qc_filter_double.py   双染 QC 工作脚本
|       |-- 04_cluster_dapi_oct4.py    DAPI 特征聚类与 OCT4 强度对照
|       |-- 05_infer_dapi_oct4.py      DAPI 特征推理与 OCT4 强度对照
|       |-- double_work_utils.py      通道配对、定位 DAPI 核心等工具
|       `-- README_double_work.md     双染参数与配对规则
|-- Ekin_DAPI_OCT4/
|   |-- run_baseline.py                PyCharm：基线特征模型入口
|   |-- run_composite.py               PyCharm：增强特征模型入口
|   |-- day2_trial.py                  OCT4 Model A 兼容入口
|   |-- run_day2_40x.ps1               Day 2、40x 启动脚本
|   |-- run_day4_sample1_20x.ps1        Day 4 Sample 1、20x
|   |-- run_day4_sample1_40x.ps1        Day 4 Sample 1、40x
|   |-- run_day4_sample2_20x.ps1        Day 4 Sample 2、20x
|   |-- run_day4_sample2_40x.ps1        Day 4 Sample 2、40x
|   |-- test_day2_trial.py             BASC、特征与颜色测试
|   |-- COMPOSITE_FEATURES.md          复合特征公式、量纲与科学解释
|   `-- README.md                      OCT4 专项说明
|-- Ekin_DAPI_YAP/
|   |-- run_baseline.py                PyCharm：基线特征模型入口
|   |-- run_composite.py               PyCharm：增强特征模型入口
|   |-- yap_40x_trial.py               YAP 主流程（也支持 20x）
|   |-- run_all_20x.ps1                20x 启动脚本
|   |-- run_all_40x.ps1                40x 启动脚本
|   |-- summarize_group_differences.py 从单细胞结果生成组间描述统计
|   |-- test_yap_40x_trial.py          元数据、泄漏和核周环测试
|   `-- README.md                      YAP 专项说明
|-- shared/
|   |-- __init__.py                    公共模块标识
|   |-- dapi_model_a.py                Ekin 共用特征/PCA/GMM/绘图核心；
|   |                                 当前也包含 OCT4/BASC 工作流实现
|   `-- entrypoint.py                  两类 Python 入口的参数与输出配置
`-- tests/
    |-- test_entrypoints.py            特征选择、倍率和输出隔离测试
    `-- test_suzui_layout.py           Suzui 目录与依赖迁移测试

根目录临时文档提取文件不属于分析流程：docx_extracted_text.txt、
extract_docx_text.py 未纳入 Git；tmp_relation_rebuilt.csv 也不是启动入口。

三、PyCharm 怎么运行

解释器选择已有 cellpose 环境：
C:\Users\dodos\miniforge3\envs\cellpose\python.exe

Ekin：在对应通道目录右键运行 run_baseline.py 或 run_composite.py。
入口顶部 MAGNIFICATION 默认 "40x"，可改为 "20x"。
运行配置 Parameters 可传入 --data-root "数据目录" --fit-magnification 20x。
OCT4 还可指定 --culture-day、--sample、--replicate。
--reuse-masks 只复用当前输出目录里已有的掩膜，不自动寻找其他目录。
旧 .ps1 和试验入口保留兼容，默认 composite，使用它们原有的输出路径。

baseline：既有核形态、DAPI 灰度及邻域统计，不是直接输入原始像素。
composite：在 baseline 上增加相对 DAPI 异质性、归一化间距、局部拥挤、
邻域形态差异和邻居方向不对称等无量纲代理。
两种入口都计算原始/增强对照指标，但最终 phenotype、坐标与后表征采用
入口指定的模型。它们调用同一份公共代码，不是复制的两套算法。

Suzui：在 DAPI/ 或 DAPI_OCT4/ 按原编号运行脚本。先检查顶部 mode、
DATASET_NAME、TRAINING_SET_NAME 和路径。通常顺序为选焦（如需要）→
分割→提取/QC→聚类→推理→SA/SHAP。双染提取脚本默认调用其 QC 工作脚本。
原先指向根目录脚本的 PyCharm 运行配置，需更新到 Suzui/DAPI/ 下。

四、依赖与数据路径

Suzui/DAPI_OCT4 复用相邻 Suzui/DAPI 的特征提取、QC、聚类及推理。
Ekin 两个 Model A 流程复用 shared/dapi_model_a.py。
来源不同的流程不合并数据表，也不互换标记解释。

Suzui 默认根目录：F:\Suzui；常规流程包括 training data、A-1 系列等。
双染默认数据为 paper_Oct-4\Tic-SNL,Rac1,Oct-4x。
输出通常位于 analysis_out；masks_double、features_double 等命名不变。

Ekin OCT4 默认数据：
E:\Kino-oka Lab\Immunostaining Data_Ekin\Immunostaining Data_Ekin\Day 2 Data
Ekin YAP 默认数据：
E:\Kino-oka Lab\Immunostaining Data_Ekin\2307YapLocalizationImmuno
以上是本机默认路径，换数据或机器时必须检查。

五、结果在哪里，先看什么

新 Python 入口默认输出：
OCT4：Ekin_DAPI_OCT4/outputs/<baseline或composite>/<倍率>/<日期样本重复标识>/
YAP ：Ekin_DAPI_YAP/outputs/<baseline或composite>/<倍率>/all_fields/
同配置重跑更新同一路径；不同数据集请指定不同 --output-root。
tables/ 放表格，figures/ 放图片，segmentation/ 放掩膜与预览；
run_info.json 记录参数、最终 feature_set 和验证信息。

Ekin 重点结果：
- model_a_single_cell_results.csv：单细胞特征、主导成分和全部后验。
- gmm_model_selection.csv：各候选 K 的模型选择指标。
- raw_vs_composite_model_comparison.csv/json：原始/增强模型对照。
- phenotype 叠图、特征热图、后验概率图和标记后表征图。

YAP 看组别差距，优先打开 tables/ 下的四张统计表：
- field_descriptive_statistics.csv：视野中位数、四分位数、均值、标准差。
- same_density_group_contrasts.csv：同密度 HA1-Ctrl、HA2-Ctrl、HA2-HA1 差值。
  普通特征给中位数差/百分比变化；YAP log2 指标给差值和反变换倍数。
- phenotype_composition.csv：各视野 phenotype 占比。
- same_density_phenotype_composition_contrasts.csv：组间占比百分点差。

YAP 的 --skip-umap 是验证/调试选项：使用 PCA1/2 绘图占位，旧 umap_
文件名仍沿用。必须查看 run_info.json，不能将占位图解读为 UMAP。

六、科学边界

Ekin Model A：
- 20x、40x 分别拟合；跨倍率或跨 baseline/composite 的同号 phenotype
  不代表同一表型。
- YAP/AF488、OCT4 强度及 HA 组、时间、浓度、密度等实验元数据不进入
  形态模型预处理/PCA/UMAP/GMM。
- OCT4 按 BASC 二元化后表征；YAP 保持连续核/核周比值，不发明阳性阈值。
- 核周环只是采样代理，不是真实全细胞胞质；PNG、曝光和分割均有限制。
- 复合特征不等同于细胞周期、多能性或分化标签。
- 当前 K 搜索 2-12，上限命中或低后验必须保留警告，不人为固定 K=2/3。
- 当前 YAP 每个 HA组×密度×倍率只有一个视野；组差表仅作描述，不把
  细胞当生物学重复做显著性检验。HA1/HA2 同时改变时间和浓度，不能拆解。

Suzui 双染：DAPI 用于分割、形态/邻域特征；同一掩膜测量 OCT4。
双染流程不把 DAPI 灰度强度用于聚类/推理，OCT4 强度单独用于对照；
不套用 Ekin 的 BASC 规则。常规 DAPI 流程仍遵守各脚本现有配置。

七、版本与维护

不再用 1.x/2.x 目录名表示基线/复合。目录描述来源与通道，入口描述方案，
Git 提交记录代码版本。README.txt 是结构与日常使用总入口；各子目录
文档保留详细规则和公式。合并代码不等于迁移历史结果。
生成图片、结果表、掩膜及缓存不随 Git 提交；整理目录不自动重算数据。

项目根目录运行测试（使用 cellpose 环境）：
python -m unittest discover -s tests -v
python -m unittest discover -s Ekin_DAPI_OCT4 -p "test_*.py" -v
python -m unittest discover -s Ekin_DAPI_YAP -p "test_*.py" -v

主要依赖：Cellpose、NumPy、pandas、SciPy、scikit-image、scikit-learn、
Matplotlib；具体流程还使用 UMAP、OpenCV、tifffile、XGBoost、SALib、SHAP。
本项目是研究代码，运行前应核对数据配置和科学适用范围。
