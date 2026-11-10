"""Run directories and machine-readable provenance."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_dir(base: Path | str, step: str, tag: str = "") -> Path:
    root = Path(base)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    output = root / step / name
    index = 2
    while output.exists():
        output = root / step / f"{name}_{index}"
        index += 1
    output.mkdir(parents=True)
    return output


def save_run_info(
    output_dir: Path | str,
    *,
    script: str | None = None,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    project_root: Path | str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": script or (Path(sys.argv[0]).name if sys.argv else None),
        "argv": sys.argv[1:],
        "params": params or {},
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    if project_root is not None:
        payload["project_root"] = str(Path(project_root))
    if extra:
        payload["extra"] = extra
    output = Path(output_dir) / "run_info.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output

