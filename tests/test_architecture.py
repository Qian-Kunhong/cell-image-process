"""Static boundaries for the project-neutral package."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "cell_image_core"


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_core_has_no_project_paths_or_import_side_effects(self):
        forbidden = ("Ekin_DAPI", "Suzui", "naruchit_pipeline", "F:\\", "E:\\")
        for path in CORE.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} leaked into {path}")
            tree = ast.parse(text, filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.For, ast.While, ast.With, ast.Try)):
                    self.fail(f"module-level executable control flow in {path}:{node.lineno}")

    def test_core_does_not_import_presentation_or_model_packages(self):
        forbidden_roots = {"matplotlib", "sklearn", "umap", "seaborn"}
        for path in CORE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(imported & forbidden_roots, f"presentation/model dependency in {path}")


if __name__ == "__main__":
    unittest.main()

