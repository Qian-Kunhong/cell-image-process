"""路径解析与运行上下文。

所有脚本都不许硬写盘符：数据路径从 config/project.json 读，输出目录由 make_run_dir 生成。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORE_SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(CORE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_SOURCE_ROOT))

from cell_image_core.provenance import (  # noqa: E402
    make_run_dir as core_make_run_dir,
    save_run_info as core_save_run_info,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.json"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """读项目配置。路径字段若为相对路径，按 PROJECT_ROOT 解析。"""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("data_root", "output_root", "legacy_code_root"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((PROJECT_ROOT / cfg[key]).resolve())
    return cfg


def data_path(*parts: str) -> Path:
    """拼出数据根下的路径，并校验存在性（不存在直接报错，不静默返回空）。"""
    root = Path(load_config()["data_root"])
    p = root.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"数据路径不存在：{p}")
    return p


def legacy_path(*parts: str) -> Path:
    """拼出旧仓库（只读引用）下的路径。新项目不改旧仓，只读它的产物。"""
    root = Path(load_config()["legacy_code_root"])
    p = root.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"旧仓库路径不存在：{p}")
    return p


def make_run_dir(step: str, tag: str = "", base: Path | str | None = None) -> Path:
    """为一次运行建独立输出目录：<base>/<step>/<时间戳>_[tag]。

    同一秒内重复调用不会覆盖前一次，而是加 _2、_3 后缀；返回的目录一定新建。
    """
    root = Path(base) if base else Path(load_config()["output_root"])
    return core_make_run_dir(root, step, tag)


def save_run_info(
    out_dir: Path,
    script: str | None = None,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """把这次运行的脚本、参数、环境写进 run_info.json，供日后复现。"""
    return core_save_run_info(
        out_dir,
        script=script,
        params=params,
        extra=extra,
        project_root=PROJECT_ROOT,
    )
