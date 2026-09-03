# DAPI 基线特征与复合特征对照说明

这些特征全部由单细胞 DAPI 分割、DAPI 灰度和同一视野内的核质心构造。它们是形态或空间代理，**不**是细胞周期、多能性、分化状态或力学状态标签。基线特征包含像素长度、像素面积、灰度及无量纲量；新增复合特征均无量纲。20× 与 40× 始终独立拟合，像素单位不等于微米。

## 1. 先理解“基线”和“复合”的关系

- **基线特征（baseline）**：原先已提取的核形态、DAPI 强度和邻域统计。“原始特征”不是原始像素，也不是没有空间信息。
- **复合特征模型（composite）**：保留基线特征，再增加 7 个相对强度/尺度归一化/邻域关系特征，并不是用 7 个量替代全部基线特征。
- 代码中有 31 个候选基线特征。全缺失或零方差列会被剔除；本次 40× 的 `adaptive_neighbor_count` 恒定而被剔除，因此热图是 **30 个基线 + 7 个复合 = 37 列**。其他数据集的有效列数可能不同。
- 复合模型是否改善置信度或稳定性需要对照评估，不能仅凭名称判断。

## 2. 原始／基线特征名称对照

下表顺序与 `shared/dapi_model_a.py` 的 `MODEL_FEATURES` 一致。`px` 为像素长度，`px²` 为像素面积；灰度单位是 PNG 导出强度（任意单位），不是绝对分子含量。

### 2.1 单核大小与形状（10 项）

| 代码名称 | 中文名称 | 单位 | 定义与读法 |
|---|---|---|---|
| `area_px` | 核面积 | px² | 分割核像素数，记为 A；名称虽为 `_px`，面积单位实际是 px² |
| `perimeter_px` | 核周长 | px | 分割轮廓估计长度，记为 P；受像素化和分割边界影响 |
| `equivalent_diameter_px` | 等面积圆直径 | px | `sqrt(4A/pi)`；把核面积换算成等面积圆的直径 |
| `major_axis_length_px` | 核长轴长度 | px | 与核二阶矩匹配的椭圆长轴长度 |
| `minor_axis_length_px` | 核短轴长度 | px | 与核二阶矩匹配的椭圆短轴长度 |
| `aspect_ratio` | 长短轴比 | 无量纲 | `major/minor`；越大越细长 |
| `eccentricity` | 偏心率 | 无量纲 | 匹配椭圆的偏心率；0 近似圆，越接近 1 越伸长 |
| `circularity` | 圆度 | 无量纲 | `4piA/P²`；连续几何中圆为 1，像素轮廓估计可能偏离理想范围 |
| `solidity` | 凸实度 | 无量纲 | 核面积／凸包面积；越低表示相对于凸包的凹陷越多，不是物理硬度 |
| `extent` | 外接矩形占据率 | 无量纲 | 核面积／轴对齐外接矩形面积；同时受形状和方向影响 |

### 2.2 核内 DAPI 强度（5 项）

这些量只使用核 mask 内的 DAPI 像素，不使用 YAP/OCT4 通道。曝光、背景、饱和、染色和导出方式都会影响数值。

| 代码名称 | 中文名称 | 单位 | 定义与读法 |
|---|---|---|---|
| `dapi_mean_intensity` | 核内 DAPI 平均强度 | 灰度单位 | 核内像素灰度均值，记为 mu；不等于总 DNA 含量 |
| `dapi_std_intensity` | 核内 DAPI 强度标准差 | 灰度单位 | 核内灰度离散程度，记为 sigma；当前代码多像素时用样本标准差 |
| `dapi_min_intensity` | 核内 DAPI 最低强度 | 灰度单位 | 核内最暗像素值 |
| `dapi_max_intensity` | 核内 DAPI 最高强度 | 灰度单位 | 核内最亮像素值；可能受饱和影响 |
| `dapi_intensity_range` | 核内 DAPI 强度极差 | 灰度单位 | `max - min`；对极端值敏感 |

### 2.3 邻域与空间统计（16 项）

距离均指核质心之间的欧氏距离。`nb` 表示邻核，不包括自身；`mean` 是均值，`std` 是标准差。通常取最近 6 个邻核，第 6 近邻距离记为 `r_k`；不足 6 个时取所有可用邻核。邻域形态统计使用半径 `r_k` 内的邻核，并包括等距离并列对象，因此数量可能超过 6。

| 代码名称 | 中文名称 | 单位 | 定义与读法 |
|---|---|---|---|
| `nn1_distance_px` | 最近邻距离 | px | 到最近一个邻核的质心距离；越大通常越疏 |
| `knn6_distance_mean_px` | 最近 6 邻核距离均值 | px | 各近邻距离的均值 |
| `knn6_distance_std_px` | 最近 6 邻核距离标准差 | px | 各近邻距离的离散程度 |
| `local_density_per_px2` | 局部核数量密度 | 核数/px² | `邻域核数/(pi r_k²)`；不是细胞接种密度 |
| `adaptive_neighbor_count` | 自适应半径内邻核数 | 个数 | 半径 `r_k` 内的邻核数量；常恒为 6，因此可能因零方差被移除 |
| `nb_area_mean_px2` | 邻核面积均值 | px² | 周围核的平均面积 |
| `nb_area_std_px2` | 邻核面积标准差 | px² | 周围核之间的大小差异；不是当前核与邻核的差异 |
| `nb_circularity_mean` | 邻核圆度均值 | 无量纲 | 周围核的平均圆度 |
| `nb_circularity_std` | 邻核圆度标准差 | 无量纲 | 周围核之间的圆度差异 |
| `nb_eccentricity_mean` | 邻核偏心率均值 | 无量纲 | 周围核的平均偏心率 |
| `nb_eccentricity_std` | 邻核偏心率标准差 | 无量纲 | 周围核之间的偏心率差异 |
| `nb_aspect_ratio_mean` | 邻核长短轴比均值 | 无量纲 | 周围核的平均伸长程度 |
| `nb_aspect_ratio_std` | 邻核长短轴比标准差 | 无量纲 | 周围核之间的伸长程度差异 |
| `nb_dapi_mean_intensity_mean` | 邻核平均 DAPI 强度的均值 | 灰度单位 | 先计算每个邻核的平均灰度，再对邻核取均值；不是把所有像素直接混合求均值 |
| `nb_dapi_mean_intensity_std` | 邻核平均 DAPI 强度的标准差 | 灰度单位 | 各邻核平均灰度之间的离散程度；不是单个核内部的灰度波动 |
| `fixed_neighbor_count` | 图内固定半径邻核数 | 个数 | 半径为该视野核等面积圆直径中位数的 2.5 倍；图内固定，不是所有图片共用固定像素半径 |

当前 YAP 流程先做边缘 QC，再在保留核之间计算上述邻域统计。图像外的邻居不可见，所以靠近视野边缘时仍存在空间截断偏差。

## 3. 复合特征由哪些基线量构造

先用三个例子理解“复合”：

- `chromatin_cv_proxy = dapi_std_intensity / abs(dapi_mean_intensity)`：不只问核内波动有多大，还问它相对于平均亮度有多大。
- `nearest_spacing_nuclear_units = nn1_distance_px / equivalent_diameter_px`：把像素距离换成自身核直径的倍数。例如间距 20 px、核直径 10 px，则为 2。
- `neighbor_size_log_disagreement = median_j abs(ln(A_j/A_i))`：比较当前核与各邻核的面积。它不同于 `nb_area_std_px2`；即使邻核彼此大小完全一致，当前核也可能与它们差异很大。

表中 `i` 为当前核，`j` 为邻核；`A`、`P`、`major`、`minor`、`C`、`E` 分别对应核面积、周长、长轴、短轴、圆度和偏心率，`x` 为二维核质心。邻域复合量由这些单核基线量及邻居关系直接构造，不一定能仅凭邻域均值/标准差重建。

| 特征 | 公式 | 增大方向与解释 |
|---|---|---|
| `nuclear_size_log_ratio` | `ln(A / median_image(A))` | 相对同视野中位数更大的核；不是细胞周期标签 |
| `nuclear_elongation_log` | `ln(major/minor)` | 核更伸长 |
| `boundary_irregularity` | `P/(2 sqrt(pi A)) - 1` | 边界偏离等面积圆更多；受分割质量影响 |
| `convexity_deficit` | `1 - solidity` | 凹陷或边界不规则更多 |
| `chromatin_cv_proxy` | `SD(DAPI)/abs(mean(DAPI))` | 核内 DAPI 相对异质性更高；只是染色质异质性代理 |
| `chromatin_range_ratio` | `(max(DAPI)-min(DAPI))/abs(mean(DAPI))` | 核内动态范围更大；对饱和和噪声敏感 |
| `nearest_spacing_nuclear_units` | `d_nearest/equivalent_diameter` | 以自身核直径计的最近邻间距更大 |
| `local_crowding_area_fraction_proxy` | `sum(A_neighbor)/(pi r_k^2)` | k 近邻圆盘内核面积占比代理更高；不是实际细胞覆盖率 |
| `neighbor_size_log_disagreement` | `median_j abs(ln(A_j/A_i))` | 与邻核大小更不一致 |
| `neighbor_shape_disagreement` | `median_j sqrt((C_j-C_i)^2+(E_j-E_i)^2)` | 与邻核圆度/偏心率更不一致；两项本身无量纲且等权 |
| `neighborhood_angular_asymmetry` | `norm(mean_j((x_j-x_i)/norm(x_j-x_i)))` | 邻居更集中在单侧，可作群落边缘/空隙附近代理；0 近似各向均匀，1 近似单侧 |

其中 `A` 为核面积、`P` 为周长、`C` 为圆度、`E` 为偏心率，`r_k` 为第 k 个近邻距离。零分母由显式下限保护，缺失值仍由既有中位数插补处理。

## 4. 热图怎么读

`phenotype_feature_heatmap.png` 每一行是该模型中的一个 phenotype，每一列是实际入模的特征。格子数值为：

`(该 phenotype 的特征均值 - 全部拟合细胞的特征均值) / 全部拟合细胞的特征标准差`

红色表示高于总体均值，蓝色表示低于总体均值，接近白色表示接近总体均值。例如面积列为 +1，表示该表型平均核面积比总体均值高 1 个总体标准差；它不表示面积增加了 100%。颜色显示范围为 -2.5 到 +2.5，范围外的值饱和显示。此热图是原始特征值均值的标准化对照，不是 PCA 载荷，也不是预处理后的 RobustScaler 数值。

颜色不代表好坏或统计显著性。此图比较 phenotype 的形态特征，不直接比较 Ctrl/HA1/HA2。Phenotype 编号没有预定义的生物学名称，也不能跨倍率或跨重拟合直接对应。

## 5. 模型比较和边界诊断

每次运行同时拟合“原始特征”和“原始 + 复合特征”两套模型，输出选择 K、BIC、平均最大后验概率、`Pmax < 0.8` 比例、三个随机初始化种子的 ARI 稳定性，以及每个复合特征与原始特征的最大绝对 Spearman 相关。结论文件不会预设复合特征一定改善。

为避免没有依据的重复加权，`nuclear_size_log_ratio`、`nuclear_elongation_log`、`boundary_irregularity` 和 `convexity_deficit` 会导出并用于解释，但因为它们分别是面积、长短轴比、圆度和 solidity 的代数重表达，不加入增强模型。增强模型只增加两个相对 DAPI 异质性代理、核尺度归一化间距、局部拥挤和三个邻域关系特征；不使用人工权重求和。

K 搜索为 2–12（样本量不足时自动缩小）。报告会显式标记最优 K 是否仍落在搜索上限；这是一项边界警告，不会为了得到较少类别而强制改 K。最终表保留所选模型的全部后验概率和主导成分。

YAP/AF488、连续 OCT4、HA 组、时间、浓度、接种密度、倍率、样本和重复信息均不得进入特征矩阵。OCT4 仅在 GMM 完成后按既有 BASC 规则二元化；YAP 仅在 GMM 完成后作为连续核/核周比值表征。
