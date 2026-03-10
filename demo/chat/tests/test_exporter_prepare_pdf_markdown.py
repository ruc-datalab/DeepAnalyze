import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def load_exporter_module():
    root = Path(__file__).resolve().parents[1]
    exporter_path = root / "backend_app" / "services" / "exporter.py"

    backend_app_pkg = types.ModuleType("backend_app")
    backend_app_pkg.__path__ = [str(root / "backend_app")]
    sys.modules.setdefault("backend_app", backend_app_pkg)

    services_pkg = types.ModuleType("backend_app.services")
    services_pkg.__path__ = [str(root / "backend_app" / "services")]
    sys.modules.setdefault("backend_app.services", services_pkg)

    workspace_stub = types.ModuleType("backend_app.services.workspace")
    workspace_stub.build_download_url = lambda rel_path: f"/workspace/download?path={rel_path}"
    workspace_stub.get_session_workspace = lambda session_id: str(root / "workspace" / session_id)
    workspace_stub.register_generated_paths = lambda session_id, rel_paths: set(rel_paths)
    workspace_stub.uniquify_path = lambda target: target
    sys.modules["backend_app.services.workspace"] = workspace_stub

    module_name = "backend_app.services.exporter"
    spec = importlib.util.spec_from_file_location(module_name, exporter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PreparePdfMarkdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_exporter_module()

    def test_localizes_workspace_download_image_pair(self):
        with TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            image_path = workspace_root / "generated" / "plot.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")

            markdown = (
                "- [plot.png](/workspace/download?session_id=session_test&path=generated%2Fplot.png&download=1)\n"
                "![plot.png](/workspace/download?session_id=session_test&path=generated%2Fplot.png&download=1)\n"
            )

            rendered = self.exporter.prepare_pdf_markdown(markdown, workspace_root)

            self.assertNotIn("/workspace/download", rendered)
            self.assertIn(f"![plot.png](<{image_path.resolve().as_posix()}>)", rendered)
            self.assertIn("`plot.png`", rendered)

    def test_localizes_single_image_markdown(self):
        with TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            image_path = workspace_root / "visualizations" / "chart.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")

            markdown = "![chart](visualizations/chart.png)\n"

            rendered = self.exporter.prepare_pdf_markdown(markdown, workspace_root)

            self.assertIn(f"![chart](<{image_path.resolve().as_posix()}>)", rendered)
            self.assertIn("`chart`", rendered)

    def test_localizes_workspace_download_image_pair_with_relative_workspace_root(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workspace_root = temp_path / "workspace" / "session_rel"
            image_path = workspace_root / "generated" / "plot.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")

            markdown = (
                "- [plot.png](/workspace/download?session_id=session_rel&path=generated%2Fplot.png&download=1)\n"
                "![plot.png](/workspace/download?session_id=session_rel&path=generated%2Fplot.png&download=1)\n"
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                rendered = self.exporter.prepare_pdf_markdown(
                    markdown,
                    Path("workspace") / "session_rel",
                )
            finally:
                os.chdir(previous_cwd)

            self.assertNotIn("/workspace/download", rendered)
            self.assertIn(f"![plot.png](<{image_path.resolve().as_posix()}>)", rendered)
            self.assertIn("`plot.png`", rendered)


if __name__ == "__main__":
    unittest.main()
