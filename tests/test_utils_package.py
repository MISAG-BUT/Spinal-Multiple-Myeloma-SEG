"""Contracts for the shared utilities package and its legacy public API."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class UtilitiesPackageTests(unittest.TestCase):
    def test_legacy_reexports_preserve_public_names_and_state(self):
        import utils
        from utils import legacy, run_nnunet_inference

        self.assertIs(run_nnunet_inference, legacy.run_nnunet_inference)
        namespace = {}
        exec("from utils import *", namespace)
        self.assertEqual(
            set(namespace) - {"__builtins__"},
            {name for name in vars(legacy) if not name.startswith("_")},
        )
        original = legacy.nib
        with patch.object(utils, "nib", object()) as replacement:
            self.assertIs(legacy.nib, replacement)
            self.assertIs(legacy.reorient_to_ras.__globals__["nib"], replacement)
        self.assertIs(utils.nib, original)

    def test_benchmark_imports_leave_legacy_and_gui_unloaded_from_other_cwd(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run(
                [sys.executable, "-I", "-B", "-c", """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import utils
from utils import benchmark
from utils.benchmark import lesion_inference, model_installation, model_registry, nifti, spine_inference
assert 'utils.legacy' not in sys.modules
assert 'napari' not in sys.modules
assert 'torch' not in sys.modules
assert model_installation.cache_location() == (root / 'Model_cache', 'project')
""", str(root)],
                cwd=cwd, check=True, capture_output=True, text=True,
            )
