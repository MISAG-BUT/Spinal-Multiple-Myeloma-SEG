"""Offline Phase 1 tests; dummy files intentionally contain no NIfTI data."""

import ast
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark_model_registry import LESION_MODELS, SPINE_MODEL, get_model, get_lesion_model
from run_prediction_of_nnUNet_benchmark_models import cli, parse_arguments


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = "ConvCT VMI_40 VMI_80 VMI_120 CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100".split()
EXPECTED = (
    ("MM_Lesion_seg_Leave_One_Out_without_ConvCT", "VMI_40 VMI_80 VMI_120 CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_VMI_40", "ConvCT VMI_80 VMI_120 CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_VMI_80", "ConvCT VMI_40 VMI_120 CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_VMI_120", "ConvCT VMI_40 VMI_80 CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_CaSupp_25", "ConvCT VMI_40 VMI_80 VMI_120 CaSupp_50 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_CaSupp_50", "ConvCT VMI_40 VMI_80 VMI_120 CaSupp_25 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_CaSupp_75", "ConvCT VMI_40 VMI_80 VMI_120 CaSupp_25 CaSupp_50 CaSupp_100"),
    ("MM_Lesion_seg_Leave_One_Out_without_CaSupp_100", "ConvCT VMI_40 VMI_80 VMI_120 CaSupp_25 CaSupp_50 CaSupp_75"),
    ("MM_Lesion_seg_all_together", "ConvCT VMI_40 VMI_80 VMI_120 CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100"),
    ("MM_Lesion_seg_just_ConvCT", "ConvCT"),
    ("MM_Lesion_seg_just_VMI_40", "VMI_40"),
    ("MM_Lesion_seg_just_VMI_80", "VMI_80"),
    ("MM_Lesion_seg_just_VMI_120", "VMI_120"),
    ("MM_Lesion_seg_just_CaSupp_25", "CaSupp_25"),
    ("MM_Lesion_seg_just_CaSupp_50", "CaSupp_50"),
    ("MM_Lesion_seg_just_CaSupp_75", "CaSupp_75"),
    ("MM_Lesion_seg_just_CaSupp_100", "CaSupp_100"),
    ("MM_Lesion_seg_all_VMI", "VMI_40 VMI_80 VMI_120"),
    ("MM_Lesion_seg_all_CaSupp", "CaSupp_25 CaSupp_50 CaSupp_75 CaSupp_100"),
)


class RegistryTests(unittest.TestCase):
    def test_all_exact_names_and_channels(self):
        self.assertEqual(set(LESION_MODELS), set(range(700, 719)))
        for model_id, (suffix, modalities) in enumerate(EXPECTED, 700):
            with self.subTest(model_id=model_id):
                model = get_lesion_model(model_id)
                self.assertEqual(model.dataset_name, "Dataset{}_{}".format(model_id, suffix))
                self.assertEqual(model.zip_filename, model.dataset_name + ".zip")
                self.assertEqual(model.required_modalities, tuple(modalities.split()))
                self.assertEqual(model.channel_names, tuple(modalities.split()) + ("mask_spine",))
                self.assertEqual(model.binary_mask_channel, "mask_spine")
                self.assertEqual(model.folds, (0, 1, 2, 3, 4))
                self.assertEqual(model.zenodo_record_id, 19708429)
                self.assertEqual(model.expected_nnunet_version, "2.6.4")
                self.assertEqual(model.trainer_directory, "nnUNetTrainer__nnUNetPlans__3d_fullres")
                self.assertEqual(model.checkpoint_filename, "checkpoint_final.pth")
                self.assertRegex(model.md5, r"^[0-9a-f]{32}$")
                self.assertGreater(model.zip_size, 1600000000)

    def test_spine_and_unsupported_ids(self):
        self.assertIs(get_model(802), SPINE_MODEL)
        self.assertEqual(SPINE_MODEL.folds, ("all",))
        self.assertEqual(SPINE_MODEL.zenodo_record_id, 18598645)
        self.assertEqual(SPINE_MODEL.zip_size, 3681185990)
        self.assertEqual(SPINE_MODEL.md5, "02f72a4c4bf8c34790b36400db7cc6d1")
        self.assertEqual(SPINE_MODEL.required_modalities, ("ConvCT",))
        self.assertTrue(SPINE_MODEL.archive_model_subtree.startswith("Spinal-Multiple-Myeloma-SEG_nnUNet_models/Dataset802_"))
        self.assertIsNone(SPINE_MODEL.expected_nnunet_version)
        with self.assertRaises(ValueError):
            get_lesion_model(802)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.mask = self.root / "mask.nii.gz"
        self.mask.write_bytes(b"dummy mask")
        self.output = self.root / "outputs"
        self.base = ["--input_dir", str(self.inputs), "--output_dir", str(self.output)]

    def files(self, *names):
        for name in names:
            (self.inputs / (name + ".nii.gz")).write_bytes(b"dummy image")

    def parse(self, *args):
        return parse_arguments(self.base + list(args))

    def invalid(self, expected, *args):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
            self.parse(*args)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(expected, error.getvalue())

    def supplied(self, model):
        return ["--lesion_model", str(model), "--spine_segmentation", str(self.mask)]

    def test_automatic_requires_convct(self):
        self.files("VMI_40")
        self.invalid("ConvCT.nii.gz", "--lesion_model", "710")
        self.files("ConvCT")
        self.assertTrue(self.parse("--lesion_model", "710").split)

    def test_supplied_mask_without_convct(self):
        self.files("VMI_40")
        plan = self.parse(*self.supplied(710))
        self.assertEqual([name for name, _ in plan.modality_paths], ["VMI_40"])
        self.assertIsNone(plan.split)

    def test_709_requires_convct_but_not_vmi(self):
        self.invalid("ConvCT.nii.gz", *self.supplied(709))
        self.files("ConvCT")
        self.parse(*self.supplied(709))

    def test_708_requires_all_modalities(self):
        self.invalid("CaSupp_100.nii.gz", *self.supplied(708))
        self.files(*MODALITIES)
        self.assertEqual(len(self.parse(*self.supplied(708)).modality_paths), 8)

    def test_700_without_convct(self):
        self.files(*MODALITIES[1:])
        self.assertEqual(len(self.parse(*self.supplied(700)).modality_paths), 7)

    def test_each_model_requires_only_its_modalities(self):
        for model_id, (_, modalities) in enumerate(EXPECTED, 700):
            with self.subTest(model_id=model_id):
                for item in self.inputs.iterdir():
                    item.unlink()
                required = modalities.split()
                self.files(*required)
                plan = self.parse(*self.supplied(model_id))
                self.assertEqual([name for name, _ in plan.modality_paths], required)
                (self.inputs / (required[-1] + ".nii.gz")).unlink()
                self.invalid(required[-1] + ".nii.gz", *self.supplied(model_id))

    def test_spine_only_modes(self):
        self.invalid("ConvCT.nii.gz", "--skip_lesion_segmentation")
        plan = self.parse("--skip_lesion_segmentation", "--spine_segmentation", str(self.mask))
        self.assertEqual(plan.modality_paths, ())
        self.files("ConvCT")
        self.assertIsNone(self.parse("--skip_lesion_segmentation").lesion_model)

    def test_capitalization_and_no_recursive_guessing(self):
        (self.inputs / "vmi_40.nii.gz").write_bytes(b"dummy")
        self.invalid("Incorrect capitalization", *self.supplied(710))
        (self.inputs / "vmi_40.nii.gz").unlink()
        nested = self.inputs / "nested"
        nested.mkdir()
        (nested / "VMI_40.nii.gz").write_bytes(b"dummy")
        self.invalid("Missing required reconstruction files: VMI_40.nii.gz", *self.supplied(710))

    def test_conflicts_and_missing_files(self):
        self.invalid("not allowed", "--lesion_model", "710", "--skip_lesion_segmentation")
        self.invalid("invalid choice", "--lesion_model", "719")
        self.invalid("required", )
        self.invalid("cannot be combined", *self.supplied(710), "--spine_model_dir", str(self.root))
        self.invalid("does not exist", "--lesion_model", "710", "--spine_segmentation", str(self.root / "missing"))
        self.invalid("Spine model directory does not exist", "--skip_lesion_segmentation", "--spine_model_dir", str(self.root / "missing"))

    def test_split_matches_v2_parser(self):
        self.files("ConvCT")
        source = ast.parse((ROOT / "run_prediction_of_nnUNet_networks_on_new_data.py").read_text(encoding="utf-8"))
        split_call = next(node for node in ast.walk(source) if isinstance(node, ast.Call)
                          and node.args and isinstance(node.args[0], ast.Constant)
                          and node.args[0].value == "--split")
        conversion = next(item.value for item in split_call.keywords if item.arg == "type")
        v2_parse = eval(compile(ast.Expression(conversion), "<v2 split>", "eval"))
        self.assertTrue(self.parse("--skip_lesion_segmentation").split)
        for value in ("True", "False", "TRUE", "true", "false", "1", "yes", "garbage"):
            with self.subTest(value=value):
                self.assertEqual(self.parse("--skip_lesion_segmentation", "--split", value).split, v2_parse(value))
                self.invalid("Explicit --split", *self.supplied(710), "--split", value)

    def test_required_directories(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(["--skip_lesion_segmentation"])
        self.files("ConvCT")
        self.invalid("Input directory does not exist", "--skip_lesion_segmentation", "--input_dir", str(self.root / "missing"))
        self.output.write_bytes(b"existing file")
        self.invalid("Output path is not a directory", "--skip_lesion_segmentation")

    def test_execution_is_offline_and_does_not_write_outputs(self):
        self.files("VMI_40")
        unrelated = self.inputs / "notes.txt"
        unrelated.write_text("preserve me")
        self.output.mkdir()
        sentinel = self.output / "prior_prediction.nii.gz"
        sentinel.write_bytes(b"prior result")
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("benchmark_nifti.validate_plan_inputs", return_value=({}, None, None)), patch("socket.socket", side_effect=AssertionError("network forbidden")), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(cli(self.base + self.supplied(710) + ["--validate_only"]), 3)
        self.assertIn("model 802 bypassed", stdout.getvalue())
        self.assertIn("mask_spine", stdout.getvalue())
        self.assertIn("Validation only", stdout.getvalue())
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(any(name in sys.modules for name in ("torch", "nnunetv2", "SimpleITK")))

    def test_direct_script_exit_and_no_output_creation(self):
        self.files("ConvCT")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "run_prediction_of_nnUNet_benchmark_models.py")] + self.base + ["--skip_lesion_segmentation"], capture_output=True, text=True)
        # Phase 1 dummy data passes path checks but must now fail NIfTI validation.
        self.assertEqual(result.returncode, 2)
        self.assertIn("validation failed", result.stderr)
        self.assertFalse(self.output.exists())


class PackagingTests(unittest.TestCase):
    def test_entry_points_and_modules(self):
        text = (ROOT / "pyproject.toml").read_text()
        for command, target in {
            "spinal-db-viewer": "spinal_mm_cli:database_viewer",
            "spinal-run-nnunet-TCIA": "spinal_mm_cli:run_prediction_tcia",
            "spinal-run-nnunet-new-prediction": "spinal_mm_cli:run_prediction_new",
            "spinal-run-nnunet-benchmark-prediction": "run_prediction_of_nnUNet_benchmark_models:cli",
        }.items():
            self.assertIn('{} = "{}"'.format(command, target), text)
        for module in ("benchmark_model_registry", "run_prediction_of_nnUNet_benchmark_models"):
            self.assertIn('"{}",'.format(module), text)


if __name__ == "__main__":
    unittest.main()
