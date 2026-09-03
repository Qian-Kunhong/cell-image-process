"""Relocation checks without executing analysis scripts or writing to F:\\Suzui."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DAPI = ROOT / "Suzui" / "DAPI"
PAIRED = ROOT / "Suzui" / "DAPI_OCT4"
spec = importlib.util.spec_from_file_location("suzui_layout_utils", PAIRED / "double_work_utils.py")
utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utils
spec.loader.exec_module(utils)


class SuzuiLayoutTests(unittest.TestCase):
    def test_paired_wrappers_resolve_dapi_core(self):
        for filename in ("02_segment_dapi_double.py", "03_extract_dapi_oct4_features.py",
                         "03-1_qc_filter_double.py", "04_cluster_dapi_oct4.py", "05_infer_dapi_oct4.py"):
            self.assertEqual(utils.sibling_single_work_dir(PAIRED / filename), DAPI)

    def test_all_sources_parse_and_sibling_script_references_resolve(self):
        files = list((ROOT / "Suzui").rglob("*.py"))
        self.assertEqual(len(files), 19)
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "with_name" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and node.args[0].value.endswith(".py")):
                    self.assertTrue(path.with_name(node.args[0].value).is_file(), str(path))

    def test_sa_shap_dependencies_remain_colocated(self):
        for filename in ("06_SA.py", "07_SHAP.py", "cluster_xgb_analysis_common.py",
                         "05_distinguish cell type.py"):
            self.assertTrue((DAPI / filename).is_file())

    def test_paired_data_defaults_remain_suzui(self):
        for filename in ("02_segment_dapi_double.py", "03_extract_dapi_oct4_features.py",
                         "04_cluster_dapi_oct4.py", "05_infer_dapi_oct4.py"):
            self.assertIn('F:\\Suzui', (PAIRED / filename).read_text(encoding="utf-8-sig"))

    def test_ekin_no_longer_owns_suzui_legacy(self):
        self.assertFalse((ROOT / "Ekin_DAPI_OCT4" / "legacy").exists())


if __name__ == "__main__":
    unittest.main()
