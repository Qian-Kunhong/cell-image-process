"""测试路径解析与运行目录：不依赖真实数据盘。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rp.paths import make_run_dir, save_run_info  # noqa: E402


class TestRunDir(unittest.TestCase):
    def test_run_dir_is_new_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d1 = make_run_dir("step", tag="demo", base=tmp)
            d2 = make_run_dir("step", tag="demo", base=tmp)
            self.assertTrue(d1.is_dir())
            self.assertTrue(d2.is_dir())
            self.assertNotEqual(d1, d2)  # 同一秒第二次不会覆盖第一次

    def test_run_info_is_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = make_run_dir("step", base=tmp)
            path = save_run_info(d, script="demo.py", params={"k": 3})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["params"]["k"], 3)
            self.assertIn("python", payload)


if __name__ == "__main__":
    unittest.main()
