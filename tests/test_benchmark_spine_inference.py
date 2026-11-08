"""Offline Phase 4A tests: synthetic CTs, fake predictor, no real weights/GPU."""

import builtins
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from benchmark_model_installation import InstallationError, ModelInstaller, ResolvedModel
from benchmark_model_registry import SPINE_MODEL
from benchmark_nifti import _new_image, compare_geometry, load_nifti, validate_plan_inputs
from benchmark_spine_inference import (
    SpineInferenceError, create_predictor, infer_spine, run_spine_only, split_crops, validate_labels,
)
from run_prediction_of_nnUNet_benchmark_models import cli, parse_arguments


def ct(z=7):
    # Mild obliquity, nontrivial spacing and translation. RAS, no shear.
    angle = 0.12
    affine = np.array([[np.cos(angle)*0.7, -np.sin(angle)*0.9, 0, 12],
                       [np.sin(angle)*0.7, np.cos(angle)*0.9, 0, -23],
                       [0, 0, 1.6, 44], [0, 0, 0, 1]])
    data = np.broadcast_to(np.arange(1, z+1, dtype=np.float32), (4, 5, z)).copy()
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    return image


class FakePredictor:
    def __init__(self, transform=None, fail_second=False):
        self.inputs = []
        self.initialized = None
        self.transform = transform
        self.fail_second = fail_second

    # Explicit signatures match verified nnU-Net v2.6.4; reject stray arguments.
    def initialize_from_trained_model_folder(self, model_training_output_dir, use_folds,
                                             checkpoint_name="checkpoint_final.pth"):
        self.initialized = (model_training_output_dir, {"use_folds": use_folds, "checkpoint_name": checkpoint_name})

    def predict_from_files_sequential(self, list_of_lists_or_source_folder,
                                      output_folder_or_list_of_truncated_output_files,
                                      save_probabilities=False, overwrite=True,
                                      folder_with_segs_from_prev_stage=None):
        assert save_probabilities is False and overwrite is False
        assert folder_with_segs_from_prev_stage is None
        inputs, outputs = list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files
        assert isinstance(inputs, list) and len(inputs) == 1 and len(inputs[0]) == 1
        assert isinstance(outputs, list) and len(outputs) == 1
        assert Path(inputs[0][0]).name == "case_0001_0000.nii.gz"
        assert Path(outputs[0]).name == "case_0001"
        source = load_nifti(inputs[0][0])
        self.inputs.append(source)
        if self.fail_second and len(self.inputs) == 2:
            raise RuntimeError("second split failed")
        labels = np.asanyarray(source.dataobj).astype(np.uint8)
        prediction = _new_image(labels, source, source.affine)
        if self.transform:
            prediction = self.transform(prediction, len(self.inputs))
        nib.save(prediction, outputs[0] + ".nii.gz")


class SpineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="V3 test ž ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "inputs"
        self.input.mkdir()
        self.output = self.root / "results"
        self.model_dir = self.root / SPINE_MODEL.dataset_name / SPINE_MODEL.trainer_directory
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "konvCT"}, "labels": {str(i): i for i in range(29)},
        }))
        self.resolved = ResolvedModel(self.model_dir, 802, "structural-only", {})

    def plan(self, *, supplied=False, split=None, validate_only=False, lesion=None):
        arguments = ["--input_dir", str(self.input), "--output_dir", str(self.output)]
        arguments += ["--lesion_model", str(lesion)] if lesion else ["--skip_lesion_segmentation"]
        if supplied:
            arguments += ["--spine_segmentation", str(self.root / "supplied.nii.gz")]
        if split is not None:
            arguments += ["--split", str(split)]
        if validate_only:
            arguments += ["--validate_only"]
        return parse_arguments(arguments)

    def save_ct(self, z=7):
        image = ct(z)
        nib.save(image, str(self.input / "ConvCT.nii.gz"))
        return image

    def execute(self, plan, predictor=None):
        installer = Mock()
        installer.resolve.return_value = self.resolved
        result = run_spine_only(plan, validate_plan_inputs(plan), installer=installer,
                                predictor_factory=lambda: predictor or FakePredictor())
        return result, installer

    def test_split_even_odd_unsplit_small_and_single_slice(self):
        for z, split, count in ((6, True, 2), (7, True, 2), (7, False, 1), (2, True, 2), (1, True, 1)):
            with self.subTest(z=z, split=split):
                reference = ct(z)
                fake = FakePredictor()
                with tempfile.TemporaryDirectory(dir=self.root) as work:
                    result, effective = infer_spine(reference, self.resolved, work, split=split, predictor_factory=lambda: fake)
                    self.assertEqual(len(list(Path(work).rglob("*_0000.nii.gz"))), count)
                self.assertEqual(len(fake.inputs), count)
                self.assertEqual(effective, count == 2)
                self.assertEqual(result.shape, reference.shape)
                np.testing.assert_array_equal(result.dataobj, reference.dataobj)
                np.testing.assert_array_equal(result.affine, reference.affine)
                self.assertEqual(result.header.get_xyzt_units(), reference.header.get_xyzt_units())
                if count == 2:
                    mid = z // 2
                    self.assertEqual(fake.inputs[0].shape[-1], mid)
                    self.assertEqual(fake.inputs[1].shape[-1], z-mid)
                    np.testing.assert_allclose(nib.affines.apply_affine(fake.inputs[1].affine, [2, 3, 0]),
                                               nib.affines.apply_affine(reference.affine, [2, 3, mid]), atol=1e-5)
                directory, kwargs = fake.initialized
                self.assertEqual(directory, str(self.model_dir))
                self.assertEqual(kwargs, {"use_folds": ("all",), "checkpoint_name": "checkpoint_final.pth"})

    def test_source_derived_split_rule_regression(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root / "utils.py").read_text(encoding="utf-8"))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "split_convCT_data")
        midpoint = next(node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "mid_z" for target in node.targets))
        self.assertIsInstance(midpoint.op, ast.FloorDiv)
        self.assertEqual(midpoint.right.value, 2)
        self.assertFalse(any(isinstance(node, ast.If) for node in ast.walk(function)))
        self.assertEqual([crop.shape[-1] for crop in split_crops(ct(7))], [3, 4])

    def test_prediction_geometry_and_label_failures(self):
        def shape(prediction, _):
            return _new_image(np.ones((2, 2, 2), np.uint8), prediction, prediction.affine)
        def affine(prediction, _):
            changed = prediction.affine.copy()
            changed[0, 3] += 0.1
            return _new_image(np.asanyarray(prediction.dataobj), prediction, changed)
        for transform, message in ((shape, "shape"), (affine, "alignment")):
            with self.subTest(message=message), tempfile.TemporaryDirectory(dir=self.root) as work:
                with self.assertRaisesRegex(SpineInferenceError, message):
                    infer_spine(ct(), self.resolved, work, predictor_factory=lambda: FakePredictor(transform))
        for value in (-1, 0.5, np.nan, np.inf, 29):
            def transform(prediction, _, value=value):
                return _new_image(np.full(prediction.shape, value, np.float64), prediction, prediction.affine)
            with self.subTest(value=value), tempfile.TemporaryDirectory(dir=self.root) as work:
                with self.assertRaises(SpineInferenceError):
                    infer_spine(ct(), self.resolved, work, predictor_factory=lambda: FakePredictor(transform))

    def test_background_only_part_is_allowed_but_empty_full_mask_is_not(self):
        def transform(prediction, index):
            if index == 1:
                return _new_image(np.zeros(prediction.shape, np.uint8), prediction, prediction.affine)
            return prediction
        with tempfile.TemporaryDirectory(dir=self.root) as work:
            result, _ = infer_spine(ct(), self.resolved, work, predictor_factory=lambda: FakePredictor(transform))
        self.assertTrue(np.all(result.get_fdata()[:, :, :3] == 0))
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_labels(_new_image(np.zeros(ct().shape, np.uint8), ct(), ct().affine))

    def test_complete_automatic_run_provenance_and_existing_outputs(self):
        reference = self.save_ct()
        source_bytes = (self.input / "ConvCT.nii.gz").read_bytes()
        self.output.mkdir()
        sentinel = self.output / "old_prediction.nii.gz"
        sentinel.write_bytes(b"preserve")
        result, installer = self.execute(self.plan())
        installer.resolve.assert_called_once_with(802, spine_model_dir=None, allow_download=False)
        labelled = load_nifti(result / "spine/spine_labelled.nii.gz")
        binary = load_nifti(result / "spine/spine_binary.nii.gz")
        np.testing.assert_array_equal(labelled.dataobj, reference.dataobj)
        self.assertTrue(np.all(binary.get_fdata() == 1))
        manifest = json.loads((result / "run.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["model"]["folds"], ["all"])
        self.assertEqual(manifest["model"]["verification"], "structural-only")
        self.assertNotIn(str(self.root), json.dumps(manifest))
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertEqual((self.input / "ConvCT.nii.gz").read_bytes(), source_bytes)
        self.assertFalse(list(self.output.glob(".v3-*")))

    def test_second_split_failure_never_publishes_a_run(self):
        self.save_ct()
        with self.assertRaisesRegex(SpineInferenceError, "second split failed"):
            self.execute(self.plan(), FakePredictor(fail_second=True))
        self.assertEqual(list(self.output.iterdir()), [])

    def test_supplied_mask_no_convct_bypasses_model_and_predictor(self):
        source = self.root / "supplied.nii.gz"
        reference = ct()
        data = np.zeros(reference.shape, np.uint32)
        data[1, 2, 3] = 70000
        nib.save(_new_image(data, reference, reference.affine), source)
        original = source.read_bytes()
        plan = self.plan(supplied=True)
        with patch("benchmark_spine_inference.ModelInstaller", side_effect=AssertionError("model bypass")), patch("benchmark_spine_inference.create_predictor", side_effect=AssertionError("predictor bypass")):
            result = run_spine_only(plan, validate_plan_inputs(plan))
        np.testing.assert_array_equal(load_nifti(result / "spine/spine_labelled.nii.gz").dataobj, data)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual((result / "spine/spine_supplied.nii.gz").read_bytes(), original)
        manifest = json.loads((result / "run.json").read_text())
        self.assertIsNone(manifest["model"])
        self.assertIsNone(manifest["split_effective"])
        self.assertEqual(manifest["spine_source"], "supplied")

    def test_optional_convct_grid_mismatch_in_supplied_mode(self):
        self.save_ct()
        image = ct()
        affine = image.affine.copy()
        affine[0, 3] += 1
        nib.save(_new_image(np.ones(image.shape, np.uint8), image, affine), self.root / "supplied.nii.gz")
        plan = self.plan(supplied=True)
        with self.assertRaisesRegex(ValueError, "alignment"):
            run_spine_only(plan, validate_plan_inputs(plan))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli(["--input_dir", str(self.input), "--output_dir", str(self.output),
                                  "--skip_lesion_segmentation", "--spine_segmentation", str(self.root / "supplied.nii.gz"),
                                  "--validate_only"]), 2)
        self.assertFalse(self.output.exists())

    def test_missing_model_never_downloads(self):
        self.save_ct()
        plan = self.plan()
        installer = ModelInstaller(self.root / "missing cache", opener=Mock(side_effect=AssertionError("network forbidden")))
        with self.assertRaisesRegex(InstallationError, "--install_models"):
            run_spine_only(plan, validate_plan_inputs(plan), installer=installer)
        installer.opener.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(installer.root.rglob("*.part")))
        self.assertFalse(list(installer.root.rglob(".v3-installation.json")))

    def test_missing_inference_dependencies_are_actionable(self):
        original_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("missing torch")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=importing), self.assertRaisesRegex(SpineInferenceError, "nnunetv2==2.6.4"):
            create_predictor()

    def test_loaded_trainer_and_configuration_are_checked(self):
        from types import SimpleNamespace
        fake = FakePredictor()
        fake.trainer_name = "WrongTrainer"
        with tempfile.TemporaryDirectory(dir=self.root) as work, self.assertRaisesRegex(SpineInferenceError, "trainer"):
            infer_spine(ct(), self.resolved, work, predictor_factory=lambda: fake)
        fake.trainer_name = "nnUNetTrainer"
        fake.configuration_manager = SimpleNamespace(configuration={"wrong": True})
        fake.plans_manager = SimpleNamespace(get_configuration=lambda name: SimpleNamespace(configuration={"fullres": True}))
        with tempfile.TemporaryDirectory(dir=self.root) as work, self.assertRaisesRegex(SpineInferenceError, "3d_fullres"):
            infer_spine(ct(), self.resolved, work, predictor_factory=lambda: fake)

    def test_validation_only_attempts_no_torch_or_nnunet_import(self):
        import subprocess
        import sys
        self.save_ct()
        source_root = Path(__file__).resolve().parents[1]
        script = '''
import builtins,sys
original=builtins.__import__
def importing(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','nnunetv2'):
        raise AssertionError('Inference import during validation: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=importing
from run_prediction_of_nnUNet_benchmark_models import cli
sys.exit(cli(sys.argv[1:]))
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, "--input_dir", str(self.input),
                                 "--output_dir", str(self.output), "--skip_lesion_segmentation", "--validate_only"],
                                cwd=str(source_root), capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("Validation only", result.stdout)
        self.assertFalse(self.output.exists())

    def test_supplied_convct_aligned_and_negative_labels_rejected(self):
        reference = self.save_ct()
        source = self.root / "supplied.nii.gz"
        values = np.asanyarray(reference.dataobj).astype(np.int16)
        values[0, 0, 0] = -1
        nib.save(_new_image(values, reference, reference.affine), source)
        plan = self.plan(supplied=True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            run_spine_only(plan, validate_plan_inputs(plan))
        values[0, 0, 0] = 9
        nib.save(_new_image(values, reference, reference.affine), source)
        result = run_spine_only(plan, validate_plan_inputs(plan))
        output = load_nifti(result / "spine/spine_labelled.nii.gz")
        compare_geometry(reference, output)
        np.testing.assert_array_equal(output.affine, load_nifti(self.input / "ConvCT.nii.gz").affine)

    def test_install_authorization_is_forwarded_only_for_spine(self):
        from dataclasses import replace
        self.save_ct()
        plan = replace(self.plan(), install_models=True)
        _, installer = self.execute(plan)
        installer.resolve.assert_called_once_with(802, spine_model_dir=None, allow_download=True)

    def test_qform_only_reference_geometry_is_preserved(self):
        reference = ct()
        reference.set_sform(None, 0)
        nib.save(reference, self.input / "ConvCT.nii.gz")
        original = load_nifti(self.input / "ConvCT.nii.gz")
        result, _ = self.execute(self.plan(split=False))
        for name in ("spine_labelled.nii.gz", "spine_binary.nii.gz"):
            output = load_nifti(result / "spine" / name)
            self.assertEqual(int(output.header["qform_code"]), 1)
            self.assertEqual(int(output.header["sform_code"]), 0)
            np.testing.assert_array_equal(output.affine, original.affine)

    def test_failed_publication_preserves_existing_destination(self):
        self.save_ct()
        from benchmark_model_installation import _publish_directory
        def publish(source, destination):
            destination.mkdir()
            (destination / "sentinel").write_bytes(b"existing output")
            return _publish_directory(source, destination)
        with patch("benchmark_spine_inference._publish_directory", side_effect=publish), self.assertRaises(FileExistsError):
            self.execute(self.plan())
        runs = list(self.output.iterdir())
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0] / "sentinel").read_bytes(), b"existing output")
        self.assertFalse((runs[0] / "run.json").exists())
        self.assertFalse(list(self.output.glob(".v3-*")))

    def test_cli_success_validation_only_lesion_preview_and_failures(self):
        self.save_ct()
        base = ["--input_dir", str(self.input), "--output_dir", str(self.output), "--skip_lesion_segmentation"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch("benchmark_spine_inference.create_predictor", side_effect=AssertionError("no inference")), patch("benchmark_spine_inference.ModelInstaller", side_effect=AssertionError("no resolution")):
                self.assertEqual(cli(base + ["--validate_only"]), 3)
            self.assertFalse(self.output.exists())
            installer = Mock()
            installer.resolve.return_value = self.resolved
            with patch("benchmark_spine_inference.ModelInstaller", return_value=installer), patch("benchmark_spine_inference.create_predictor", return_value=FakePredictor()):
                self.assertEqual(cli(base), 0)
            with patch("benchmark_spine_inference.ModelInstaller", side_effect=InstallationError("missing model")):
                self.assertEqual(cli(base), 4)
            with patch("benchmark_spine_inference.ModelInstaller", return_value=installer), patch("benchmark_spine_inference.create_predictor", side_effect=SpineInferenceError("missing dependencies")):
                self.assertEqual(cli(base), 5)
            with patch("benchmark_spine_inference.run_spine_only", side_effect=AssertionError("no partial lesion run")):
                self.assertEqual(cli(base[:-1] + ["--lesion_model", "709", "--validate_only"]), 3)

    def test_concurrent_runs_have_unique_outputs(self):
        self.save_ct()
        plan = self.plan()
        validated = validate_plan_inputs(plan)
        results, errors = [], []
        def run():
            try:
                installer = Mock()
                installer.resolve.return_value = self.resolved
                results.append(run_spine_only(plan, validated, installer=installer, predictor_factory=FakePredictor))
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(errors)
        self.assertEqual(len(set(results)), 2)
        self.assertFalse(list(self.output.glob(".v3-*")))

    def test_cli_publication_collision_and_native_error_return_five(self):
        self.save_ct()
        self.output.mkdir()
        unrelated = self.output / "unrelated.txt"
        unrelated.write_bytes(b"preserve unrelated")
        arguments = ["--input_dir", str(self.input), "--output_dir", str(self.output), "--skip_lesion_segmentation"]
        from benchmark_model_installation import _publish_directory
        for failure in ("collision", "native"):
            with self.subTest(failure=failure):
                destinations = []
                def publish(stage, destination):
                    destination.mkdir()
                    (destination / "existing.txt").write_bytes(b"existing output")
                    destinations.append(destination)
                    if failure == "native":
                        raise InstallationError("filesystem lacks exclusive rename")
                    return _publish_directory(stage, destination)
                installer = Mock()
                installer.resolve.return_value = self.resolved
                error = io.StringIO()
                with patch("benchmark_spine_inference.ModelInstaller", return_value=installer), patch("benchmark_spine_inference.create_predictor", return_value=FakePredictor()), patch("benchmark_spine_inference._publish_directory", side_effect=publish), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                    self.assertEqual(cli(arguments), 5)
                self.assertEqual(len(destinations), 1)
                self.assertEqual((destinations[0] / "existing.txt").read_bytes(), b"existing output")
                self.assertFalse((destinations[0] / "run.json").exists())
                self.assertFalse(list(self.output.glob(".v3-*")))
                self.assertEqual(unrelated.read_bytes(), b"preserve unrelated")
                self.assertNotIn("Traceback", error.getvalue())
                self.assertIn("V3 spine execution failed", error.getvalue())
                self.assertRegex(error.getvalue(), "Retry|supporting atomic")

    def test_automatic_empty_prediction_is_a_cli_failure(self):
        self.save_ct()
        def empty(prediction, index):
            return _new_image(np.zeros(prediction.shape, np.uint8), prediction, prediction.affine)
        installer = Mock()
        installer.resolve.return_value = self.resolved
        error = io.StringIO()
        with patch("benchmark_spine_inference.ModelInstaller", return_value=installer), patch("benchmark_spine_inference.create_predictor", return_value=FakePredictor(empty)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            self.assertEqual(cli(["--input_dir", str(self.input), "--output_dir", str(self.output), "--skip_lesion_segmentation"]), 5)
        self.assertIn("empty", error.getvalue())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_supplied_empty_mask_and_binary_conversion_fail_without_side_effects(self):
        from benchmark_nifti import binary_spine_mask
        reference = ct()
        empty = _new_image(np.zeros(reference.shape, np.uint8), reference, reference.affine)
        source = self.root / "supplied.nii.gz"
        nib.save(empty, source)
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "empty"):
            binary_spine_mask(empty)
        with patch("benchmark_spine_inference.ModelInstaller", side_effect=AssertionError("resolution bypass")), patch("benchmark_spine_inference.create_predictor", side_effect=AssertionError("predictor bypass")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli(["--input_dir", str(self.input), "--output_dir", str(self.output), "--skip_lesion_segmentation", "--spine_segmentation", str(source)]), 2)
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_supplied_mode_install_opt_in_still_imports_no_inference_dependencies(self):
        reference = ct()
        source = self.root / "supplied.nii.gz"
        nib.save(_new_image(np.ones(reference.shape, np.uint16)*300, reference, reference.affine), source)
        importer = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "nnunetv2"):
                raise AssertionError("unexpected inference import")
            return importer(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded), patch("benchmark_spine_inference.ModelInstaller", side_effect=AssertionError("model bypass")), patch("socket.socket", side_effect=AssertionError("network forbidden")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli(["--input_dir", str(self.input), "--output_dir", str(self.output), "--skip_lesion_segmentation", "--spine_segmentation", str(source), "--install_models"]), 0)
        self.assertEqual(len(list(self.output.glob("run-*"))), 1)

    def test_large_uint64_supplied_labels_are_not_rounded_or_overflowed(self):
        reference = ct()
        values = np.zeros(reference.shape, np.uint64)
        values[0, 0, 0] = 2**53 + 1
        values[-1, -1, -1] = np.iinfo(np.uint64).max
        source = self.root / "supplied.nii.gz"
        nib.save(_new_image(values, reference, reference.affine), source)
        result = run_spine_only(self.plan(supplied=True), validate_plan_inputs(self.plan(supplied=True)))
        np.testing.assert_array_equal(load_nifti(result / "spine/spine_labelled.nii.gz").dataobj, values)

    def test_malformed_model_dataset_is_rejected_before_predictor_construction(self):
        invalid = (
            [], {"channel_names": {"0": "wrong"}},
            {"channel_names": {"0": "konvCT"}, "labels": []},
            {"channel_names": {"0": "konvCT"}, "labels": {}},
            {"channel_names": {"0": "konvCT"}, "labels": {"background": "0", "1": 1}},
            {"channel_names": {"0": "konvCT"}, "labels": {"background": 0}},
        )
        for dataset in invalid:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory(dir=self.root) as work:
                (self.model_dir / "dataset.json").write_text(json.dumps(dataset))
                factory = Mock(side_effect=AssertionError("must reject before construction"))
                with self.assertRaisesRegex(SpineInferenceError, "Model 802"):
                    infer_spine(ct(), self.resolved, work, predictor_factory=factory)
                factory.assert_not_called()

    def test_real_installer_explicit_install_then_offline_reuse_and_external_path(self):
        # Reuse Phase 3's tiny archive fixture; the real installer enforces hashes,
        # extraction, structure, receipts, and locks. Checkpoints remain dummy bytes.
        from test_benchmark_installation import fixture, Response
        from dataclasses import replace
        import zipfile
        spec, payload = fixture(802)
        source_zip = io.BytesIO(payload)
        replacement = io.BytesIO()
        with zipfile.ZipFile(source_zip) as original, zipfile.ZipFile(replacement, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in original.infolist():
                data = original.read(member)
                if member.filename.endswith("dataset.json"):
                    dataset = json.loads(data)
                    dataset["labels"] = {str(i): i for i in range(29)}
                    data = json.dumps(dataset).encode()
                archive.writestr(member, data)
        import hashlib
        payload = replacement.getvalue()
        spec = replace(spec, zip_size=len(payload), md5=hashlib.md5(payload).hexdigest())
        opener = Mock(side_effect=lambda *args, **kwargs: Response(payload))
        installer = ModelInstaller(self.root / "cache", opener=opener, progress=lambda *args: None)
        self.save_ct()
        with patch("benchmark_model_installation.get_model", return_value=spec):
            plan = replace(self.plan(), install_models=True)
            first = run_spine_only(plan, validate_plan_inputs(plan), installer=installer, predictor_factory=FakePredictor)
            opener.assert_called_once()
            opener.side_effect = AssertionError("network forbidden on reuse")
            for authorization in (False, True):
                current = replace(plan, install_models=authorization)
                run_spine_only(current, validate_plan_inputs(current), installer=installer, predictor_factory=FakePredictor)
            self.assertEqual(opener.call_count, 1)
            resolved = installer.resolve(802, allow_download=False)
            external = replace(self.plan(), spine_model_dir=resolved.model_directory)
            fake = FakePredictor()
            run_spine_only(external, validate_plan_inputs(external), installer=installer, predictor_factory=lambda: fake)
            self.assertEqual(fake.initialized[0], str(resolved.model_directory))
            self.assertEqual(opener.call_count, 1)
        manifest = json.loads((first / "run.json").read_text())
        self.assertEqual(manifest["model"]["verification"], "pinned-archive-and-local-sha256")


if __name__ == "__main__":
    unittest.main()
