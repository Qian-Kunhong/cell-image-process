# 写作项目调用入口

本目录是分析仓库与论文/报告写作项目之间的稳定接口。原始脚本和结果图不移动、不改名，避免破坏已有分析；清单可随结果更新而重建。

## 直接使用

- `manifest.json`：完整机器可读索引，包含项目根目录、代码路径、图片路径和分类。
- `code_manifest.csv`：脚本路径与用途；`role=entrypoint` 等条目适合追溯分析入口。
- `image_manifest.csv`：全部图片路径；优先筛选 `recommended_for_writing=True`，可得到主分析目录中的正文/PPT 候选图。
- `absolute_path`：同一台电脑上的另一个项目可直接读取。
- `relative_path`：仓库迁移后，用 `project_root / relative_path` 重新定位。

Python 示例：

```python
import json
from pathlib import Path

index = json.loads(Path("writing_assets/manifest.json").read_text(encoding="utf-8"))
main_figures = [Path(x["absolute_path"]) for x in index["images"] if x["recommended_for_writing"]]
```

## 更新清单

在项目根目录运行：

```powershell
python .\writing_assets\build_manifest.py
```

若写作项目需要一个可独立移动的图片包：

```powershell
python .\writing_assets\build_manifest.py --copy recommended --destination "D:\paper\figures"
```

`recommended` 仅复制主分析目录 `Ekin_DAPI_YAP` 中 `figures_ppt` 和 `figures_ha_main` 的正文候选图，并保留原目录层级以防重名。`publication` 还会包含 Doubao 快照中的候选图；`--copy all` 会复制所有结果图（包括 QC、逐视野 overlay 和分割图，体积较大）。

## 图片分类

| `kind` | 用途 |
|---|---|
| `publication_candidate` | 正文、汇报或补充材料的优先候选 |
| `analysis_figure` | 常规分析图 |
| `field_overlay` | 单视野表型定位图 |
| `qc` | 分割、核周环等质量控制图 |
| `other_image` | 临时图或不属于标准输出目录的图片 |

注意：清单同时收录 `Ekin_DAPI_YAP` 与 `Ekin_DAPI_YAP - Doubao` 的结果，两者是不同代码/结果来源，写作时不要仅凭同名图片互相替换。优先根据研究问题和对应 `run_info.json` 选择结果版本。

`source_status=main` 表示当前主目录；`alternate_snapshot` 表示 Doubao 快照；`legacy_output` 表示旧 OCT4 输出；`temporary` 表示根目录临时图片。
