# DAPI 复合形态与空间特征

这些特征全部由单细胞 DAPI 分割、DAPI 灰度和同一视野内的核质心构造。它们是形态或空间代理，**不**是细胞周期、多能性、分化状态或力学状态标签。所有公式均无量纲；20× 与 40× 始终独立拟合。

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
| `neighbor_size_log_disagreement` | `median_j |ln(A_j/A_i)|` | 与邻核大小更不一致 |
| `neighbor_shape_disagreement` | `median_j sqrt((C_j-C_i)^2+(E_j-E_i)^2)` | 与邻核圆度/偏心率更不一致；两项本身无量纲且等权 |
| `neighborhood_angular_asymmetry` | `norm(mean_j((x_j-x_i)/norm(x_j-x_i)))` | 邻居更集中在单侧，可作群落边缘/空隙附近代理；0 近似各向均匀，1 近似单侧 |

其中 `A` 为核面积、`P` 为周长、`C` 为圆度、`E` 为偏心率，`r_k` 为第 k 个近邻距离。零分母由显式下限保护，缺失值仍由既有中位数插补处理。

## 模型比较和边界诊断

每次运行同时拟合“原始特征”和“原始 + 复合特征”两套模型，输出选择 K、BIC、平均最大后验概率、`Pmax < 0.8` 比例、三个随机初始化种子的 ARI 稳定性，以及每个复合特征与原始特征的最大绝对 Spearman 相关。结论文件不会预设复合特征一定改善。

为避免没有依据的重复加权，`nuclear_size_log_ratio`、`nuclear_elongation_log`、`boundary_irregularity` 和 `convexity_deficit` 会导出并用于解释，但因为它们分别是面积、长短轴比、圆度和 solidity 的代数重表达，不加入增强模型。增强模型只增加两个相对 DAPI 异质性代理、核尺度归一化间距、局部拥挤和三个邻域关系特征；不使用人工权重求和。

K 搜索为 2–12（样本量不足时自动缩小）。报告会显式标记最优 K 是否仍落在搜索上限；这是一项边界警告，不会为了得到较少类别而强制改 K。最终表保留所选模型的全部后验概率和主导成分。

YAP/AF488、连续 OCT4、HA 组、时间、浓度、接种密度、倍率、样本和重复信息均不得进入特征矩阵。OCT4 仅在 GMM 完成后按既有 BASC 规则二元化；YAP 仅在 GMM 完成后作为连续核/核周比值表征。
