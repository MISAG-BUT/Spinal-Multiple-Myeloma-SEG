"""Synthetic Phase 4B tests. No Torch, nnU-Net, real checkpoints, or network."""

from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from benchmark_model_registry import get_model, get_lesion_model, LESION_MODELS
from benchmark_model_installation import InstallationError, ModelInstaller, ResolvedModel
from benchmark_nifti import (
    _new_image, compare_geometry, load_nifti, prepare_lesion_channels, validate_plan_inputs,
    validate_label_values, binary_spine_mask, reconstruct_lesions,
)
from benchmark_lesion_inference import (
    LesionInferenceError, infer_lesions, lesion_model_metadata, run_lesion_pipeline,
    validate_lesion_prediction,
)
from run_prediction_of_nnUNet_benchmark_models import cli, parse_arguments
from test_benchmark_spine_inference import ct, FakePredictor


class LesionPredictor:
    def __init__(self, spec, *, transform=None, empty=False, fail=False):
        self.spec, self.transform, self.empty, self.fail = spec, transform, empty, fail
        self.initialized = None
        self.channels = []

    def initialize_from_trained_model_folder(self, model_training_output_dir, use_folds,
                                             checkpoint_name="checkpoint_final.pth"):
        self.initialized = (model_training_output_dir, use_folds, checkpoint_name)

    def predict_from_files_sequential(self, list_of_lists_or_source_folder,
                                      output_folder_or_list_of_truncated_output_files,
                                      save_probabilities=False, overwrite=True,
                                      folder_with_segs_from_prev_stage=None):
        assert save_probabilities is False and overwrite is False
        assert folder_with_segs_from_prev_stage is None
        assert len(list_of_lists_or_source_folder) == 1
        paths = list_of_lists_or_source_folder[0]
        assert len(paths) == len(self.spec.channel_names)
        assert [Path(path).name for path in paths] == ["case_0001_{:04d}.nii.gz".format(i) for i in range(len(paths))]
        # Keep snapshots: the real workflow intentionally deletes working files.
        loaded = [load_nifti(path) for path in paths]
        self.channels = [_new_image(np.asanyarray(channel.dataobj).copy(), channel, channel.affine) for channel in loaded]
        for channel in self.channels:
            compare_geometry(self.channels[0], channel)
            np.testing.assert_array_equal(channel.affine, self.channels[0].affine)
        assert np.isin(self.channels[-1].dataobj, (0, 1)).all()
        if self.fail:
            raise RuntimeError("mock lesion predictor failure")
        data = np.zeros(self.channels[0].shape, np.uint8)
        if not self.empty:
            data[0, 0, 0] = data[-1, -1, -1] = 1
        prediction = _new_image(data, self.channels[0], self.channels[0].affine)
        if self.transform:
            prediction = self.transform(prediction)
        assert len(output_folder_or_list_of_truncated_output_files) == 1
        output = output_folder_or_list_of_truncated_output_files[0]
        assert Path(output).name == "case_0001"
        nib.save(prediction, output + ".nii.gz")


class LesionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="V3 lesion ž ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "results"
        source = ct(27)
        self.reference = _new_image(np.zeros((23, 25, 27), np.float32), source, source.affine)
        self.mask_data = np.zeros(self.reference.shape, np.uint16)
        self.mask_data[11, 12, 13] = 300
        self.mask_path = self.root / "spine_supplied.nii.gz"
        self.models = {}

    def model(self, dataset_id):
        if dataset_id in self.models:
            return self.models[dataset_id]
        spec = get_model(dataset_id)
        directory = self.root / "models" / spec.dataset_name / spec.trainer_directory
        directory.mkdir(parents=True)
        dataset = {"channel_names": {str(i): name for i, name in enumerate(spec.channel_names)},
                   "file_ending": ".nii.gz", "labels": {"background": 0, "Lesions": 1}}
        if dataset_id == 802:
            dataset["labels"] = {str(i): i for i in range(29)}
        (directory / "dataset.json").write_text(json.dumps(dataset))
        (directory / "plans.json").write_text(json.dumps({"dataset_name": spec.dataset_name,
            "plans_name": "nnUNetPlans", "configurations": {"3d_fullres": {}}}))
        for fold in spec.folds:
            path = directory / ("fold_{}".format(fold))
            path.mkdir()
            (path / spec.checkpoint_filename).write_bytes(b"dummy weights; never loaded")
        result = ResolvedModel(directory, dataset_id, "test-double", {"artifact": {"record_id": spec.zenodo_record_id}})
        self.models[dataset_id] = result
        return result

    def plan(self, model_id=710, *, supplied=True, install=False, validate_only=False):
        spec = get_lesion_model(model_id)
        for index, modality in enumerate(spec.required_modalities):
            data = np.full(self.reference.shape, index + 10, np.float32)
            nib.save(_new_image(data, self.reference, self.reference.affine), self.inputs / (modality + ".nii.gz"))
        arguments = ["--input_dir", str(self.inputs), "--output_dir", str(self.output), "--lesion_model", str(model_id)]
        if supplied:
            nib.save(_new_image(self.mask_data, self.reference, self.reference.affine), self.mask_path)
            arguments += ["--spine_segmentation", str(self.mask_path)]
        else:
            data = np.zeros(self.reference.shape, np.float32)
            data[11, 12, 13] = 7
            nib.save(_new_image(data, self.reference, self.reference.affine), self.inputs / "ConvCT.nii.gz")
        if install:
            arguments += ["--install_models"]
        if validate_only:
            arguments += ["--validate_only"]
        return parse_arguments(arguments)

    def installer(self, events=None):
        installer = Mock()
        def resolve(dataset_id, **kwargs):
            if events is not None:
                events.append(("resolve", dataset_id))
            return self.model(dataset_id)
        installer.resolve.side_effect = resolve
        return installer

    def run_pipeline(self, plan, fake=None, installer=None):
        fake = fake or LesionPredictor(plan.lesion_model)
        result = run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=installer or self.installer(),
                                    spine_predictor_factory=FakePredictor, lesion_predictor_factory=lambda: fake)
        return result, fake

    def test_all_19_models_exact_channels_folds_and_reconstruction(self):
        for model_id, spec in LESION_MODELS.items():
            with self.subTest(model=model_id):
                plan = self.plan(model_id)
                validated = validate_plan_inputs(plan)
                before = {path: path.read_bytes() for path in self.inputs.iterdir()}
                source_bytes = self.mask_path.read_bytes()
                installer = self.installer()
                result, fake = self.run_pipeline(plan, installer=installer)
                installer.resolve.assert_called_once_with(model_id, allow_download=False)
                self.assertEqual(fake.initialized, (str(self.model(model_id).model_directory), (0, 1, 2, 3, 4), "checkpoint_final.pth"))
                manifest = json.loads((result / "run.json").read_text())
                self.assertEqual(manifest["ordered_lesion_channels"], list(spec.channel_names))
                self.assertEqual(manifest["crop"]["start"], [6, 7, 8])
                self.assertEqual(manifest["crop"]["stop"], [17, 18, 19])
                slices = tuple(slice(a, b) for a, b in zip(manifest["crop"]["start"], manifest["crop"]["stop"]))
                for name, channel in zip(spec.required_modalities, fake.channels[:-1]):
                    np.testing.assert_array_equal(channel.dataobj, np.asanyarray(validated[0][name].dataobj)[slices])
                expected_mask = (self.mask_data[slices] != 0).astype(np.uint8)
                np.testing.assert_array_equal(fake.channels[-1].dataobj, expected_mask)
                expected = np.zeros(self.reference.shape, np.uint8)
                expected[6, 7, 8] = expected[16, 17, 18] = 1
                lesions = load_nifti(result / "lesion/lesion_labelled.nii.gz")
                np.testing.assert_array_equal(lesions.dataobj, expected)
                compare_geometry(validated[2], lesions)
                self.assertEqual(before, {path: path.read_bytes() for path in self.inputs.iterdir()})
                self.assertEqual(source_bytes, self.mask_path.read_bytes())
                self.assertEqual((result / "spine/spine_supplied.nii.gz").read_bytes(), source_bytes)
                self.assertFalse((result / "work").exists())
                self.assertNotIn(str(self.root), json.dumps(manifest))

    def test_automatic_preflight_resolves_both_before_predictor(self):
        plan = self.plan(708, supplied=False)
        events = []
        installer = self.installer(events)
        def spine_factory():
            events.append(("predictor", 802))
            return FakePredictor()
        fake = LesionPredictor(plan.lesion_model)
        result = run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=installer,
                                    spine_predictor_factory=spine_factory, lesion_predictor_factory=lambda: fake)
        self.assertEqual(events[:3], [("resolve", 802), ("resolve", 708), ("predictor", 802)])
        manifest = json.loads((result / "run.json").read_text())
        self.assertEqual(manifest["model"]["folds"], ["all"])
        self.assertEqual(manifest["lesion_model"]["folds"], [0, 1, 2, 3, 4])
        self.assertTrue(manifest["split_effective"])
        self.assertEqual(manifest["spine_source"], "automatic")

    def test_missing_lesion_model_prevents_spine_prediction_and_output_creation(self):
        plan = self.plan(709, supplied=False)
        installer = self.installer()
        def resolve(dataset_id, **kwargs):
            if dataset_id == 802:
                return self.model(802)
            raise InstallationError("Missing lesion model; use --install_models")
        installer.resolve.side_effect = resolve
        factory = Mock(side_effect=AssertionError("must fail before inference"))
        with self.assertRaisesRegex(InstallationError, "--install_models"):
            run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=installer,
                                spine_predictor_factory=factory, lesion_predictor_factory=factory)
        factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_supplied_710_needs_no_convct_and_authorizes_only_selected_model(self):
        plan = self.plan(710, install=True)
        installer = self.installer()
        factory = Mock(side_effect=AssertionError("spine bypass"))
        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=installer,
                                spine_predictor_factory=factory, lesion_predictor_factory=lambda: LesionPredictor(plan.lesion_model))
        installer.resolve.assert_called_once_with(710, allow_download=True)
        factory.assert_not_called()
        self.assertFalse((self.inputs / "ConvCT.nii.gz").exists())

    def test_real_installer_missing_lesion_never_uses_network(self):
        plan = self.plan(710)
        opener = Mock(side_effect=AssertionError("network forbidden"))
        installer = ModelInstaller(self.root / "cache", opener=opener)
        with self.assertRaisesRegex(InstallationError, "--install_models"):
            run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=installer,
                                lesion_predictor_factory=Mock(side_effect=AssertionError("no inference")))
        opener.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(installer.root.rglob("*.part")))

    def test_real_installer_explicit_lesion_install_and_offline_reuse(self):
        import hashlib
        import zipfile
        from test_benchmark_installation import Response
        plan = self.plan(710, install=True)
        spec = plan.lesion_model
        model = self.model(710)
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in model.model_directory.rglob("*"):
                if path.is_file():
                    archive.writestr(spec.dataset_name + "/" + spec.trainer_directory + "/" + path.relative_to(model.model_directory).as_posix(), path.read_bytes())
        content = payload.getvalue()
        fixture_spec = replace(spec, zip_size=len(content), md5=hashlib.md5(content).hexdigest())
        opener = Mock(side_effect=lambda *args, **kwargs: Response(content))
        installer = ModelInstaller(self.root / "cache", opener=opener, progress=lambda *args: None)
        with patch("benchmark_model_installation.get_model", return_value=fixture_spec):
            result, _ = self.run_pipeline(plan, installer=installer)
            opener.assert_called_once_with(fixture_spec.download_url, timeout=60)
            opener.side_effect = AssertionError("network forbidden on reuse")
            self.run_pipeline(replace(plan, install_models=False), installer=installer)
            self.assertEqual(opener.call_count, 1)
        manifest = json.loads((result / "run.json").read_text())
        self.assertIsNone(manifest["model"])
        self.assertEqual(manifest["lesion_model"]["verification"], "pinned-archive-and-local-sha256")
        self.assertFalse((installer.root / "models/18598645").exists())

    def test_reconstruction_and_write_failures_publish_no_run(self):
        plan = self.plan(710)
        for function, error in (("reconstruct_lesions", ValueError("mock reconstruction failure")),
                                ("save_nifti_no_clobber", OSError("mock output write failure"))):
            with self.subTest(function=function), patch("benchmark_lesion_inference." + function, side_effect=error), patch("benchmark_lesion_inference._publish_directory") as publish:
                with self.assertRaisesRegex((ValueError, OSError), "mock"):
                    self.run_pipeline(plan)
                publish.assert_not_called()
            self.assertEqual(list(self.output.iterdir()), [])

    def test_even_volume_and_subtolerance_metadata_use_one_cropped_grid(self):
        self.reference = _new_image(np.zeros((24, 26, 28), np.float32), self.reference, self.reference.affine)
        self.mask_data = np.zeros(self.reference.shape, np.uint16)
        self.mask_data[12, 13, 0] = 300
        plan = self.plan(717)
        path = self.inputs / "VMI_80.nii.gz"
        source = load_nifti(path)
        affine = source.affine.copy()
        affine[0, 3] += 0.0004  # Within the unchanged 0.001 mm validation contract.
        nib.save(_new_image(np.asanyarray(source.dataobj).copy(), source, affine), path)
        original_bytes = path.read_bytes()
        result, fake = self.run_pipeline(plan)
        for channel in fake.channels[1:]:
            np.testing.assert_array_equal(channel.affine, fake.channels[0].affine)
        manifest = json.loads((result / "run.json").read_text())
        self.assertEqual(manifest["crop"]["start"], [7, 8, 0])
        self.assertEqual(manifest["crop"]["stop"], [18, 19, 6])
        self.assertEqual(load_nifti(result / "lesion/lesion_labelled.nii.gz").shape, (24, 26, 28))
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_loaded_lesion_trainer_and_configuration_are_checked(self):
        from types import SimpleNamespace
        plan = self.plan(710)
        fake = LesionPredictor(plan.lesion_model)
        fake.trainer_name = "WrongTrainer"
        with self.assertRaisesRegex(LesionInferenceError, "trainer"):
            self.run_pipeline(plan, fake)
        fake.trainer_name = "nnUNetTrainer"
        fake.configuration_manager = SimpleNamespace(configuration={"wrong": True})
        fake.plans_manager = SimpleNamespace(get_configuration=lambda name: SimpleNamespace(configuration={"fullres": True}))
        with self.assertRaisesRegex(LesionInferenceError, "configuration"):
            self.run_pipeline(plan, fake)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_missing_inference_dependencies_return_clear_cli_failure(self):
        from benchmark_spine_inference import SpineInferenceError
        self.plan(710)
        args = ["--input_dir", str(self.inputs), "--output_dir", str(self.output), "--lesion_model", "710", "--spine_segmentation", str(self.mask_path)]
        error = io.StringIO()
        with patch("benchmark_lesion_inference.ModelInstaller", return_value=self.installer()), patch("benchmark_lesion_inference.create_predictor", side_effect=SpineInferenceError("Install nnunetv2==2.6.4 and CUDA-enabled PyTorch")) as create, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            self.assertEqual(cli(args), 5)
        create.assert_called_once_with(inference_name="Benchmark lesion inference")
        self.assertIn("nnunetv2==2.6.4", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_boundary_crops_and_single_voxel_masks(self):
        for occupied in (((0, 0, 0),), ((22, 24, 26),), ((0, 0, 0), (22, 24, 26)), ((0, 12, 13),)):
            with self.subTest(occupied=occupied):
                self.mask_data[:] = 0
                for index in occupied:
                    self.mask_data[index] = 300
                result, _ = self.run_pipeline(self.plan(710))
                manifest = json.loads((result / "run.json").read_text())
                start = tuple(max(min(index[axis] for index in occupied)-5, 0) for axis in range(3))
                stop = tuple(min(max(index[axis] for index in occupied)+6, self.reference.shape[axis]) for axis in range(3))
                self.assertEqual(manifest["crop"]["start"], list(start))
                self.assertEqual(manifest["crop"]["stop"], list(stop))
                expected = np.zeros(self.reference.shape, np.uint8)
                expected[start] = 1
                expected[tuple(value-1 for value in stop)] = 1
                np.testing.assert_array_equal(load_nifti(result / "lesion/lesion_labelled.nii.gz").dataobj, expected)

    def test_qform_only_and_background_only_lesions(self):
        self.reference.set_sform(None, 0)
        plan = self.plan(710)
        result, _ = self.run_pipeline(plan, LesionPredictor(plan.lesion_model, empty=True))
        original = load_nifti(self.inputs / "VMI_40.nii.gz")
        for filename in ("lesion/lesion_labelled.nii.gz", "spine/spine_labelled.nii.gz", "spine/spine_binary.nii.gz"):
            output = load_nifti(result / filename)
            self.assertEqual(int(output.header["qform_code"]), int(original.header["qform_code"]))
            self.assertEqual(int(output.header["sform_code"]), int(original.header["sform_code"]))
            np.testing.assert_array_equal(output.affine, original.affine)
        self.assertEqual(load_nifti(result / "lesion/lesion_labelled.nii.gz").get_fdata().sum(), 0)

    def test_invalid_labels_geometry_and_predictor_failure_publish_nothing(self):
        plan = self.plan(710)
        transforms = []
        for value in (-1, 0.5, 2, np.nan, np.inf):
            transforms.append(lambda prediction, value=value: _new_image(np.full(prediction.shape, value, np.float32), prediction, prediction.affine))
        transforms.append(lambda prediction: _new_image(np.zeros((2, 2, 2), np.uint8), prediction, prediction.affine))
        def shifted(prediction):
            affine = prediction.affine.copy()
            affine[0, 3] += 0.01
            return _new_image(np.zeros(prediction.shape, np.uint8), prediction, affine)
        transforms.append(shifted)
        for transform in transforms:
            with self.subTest(transform=transform), self.assertRaises(LesionInferenceError):
                self.run_pipeline(plan, LesionPredictor(plan.lesion_model, transform=transform))
            self.assertEqual(list(self.output.iterdir()), [])
        with self.assertRaisesRegex(LesionInferenceError, "mock lesion predictor failure"):
            self.run_pipeline(plan, LesionPredictor(plan.lesion_model, fail=True))
        self.assertEqual(list(self.output.iterdir()), [])

    def test_lesion_metadata_validation_precedes_any_predictor(self):
        plan = self.plan(710, supplied=False)
        resolved = self.model(710)
        original = json.loads((resolved.model_directory / "dataset.json").read_text())
        for replacement in ({"channel_names": {"0": "wrong"}}, {"labels": {"background": 0, "lesion": 2}}, {"labels": []}):
            dataset = dict(original)
            dataset.update(replacement)
            (resolved.model_directory / "dataset.json").write_text(json.dumps(dataset))
            with self.subTest(replacement=replacement), self.assertRaises(InstallationError):
                self.run_pipeline(plan)
            self.assertFalse(self.output.exists())
        (resolved.model_directory / "dataset.json").write_text(json.dumps(original))
        (resolved.model_directory / "fold_4/checkpoint_final.pth").unlink()
        with self.assertRaisesRegex(InstallationError, "checkpoint"):
            self.run_pipeline(plan)
        self.assertFalse(self.output.exists())

    def test_prepared_order_count_and_nonbinary_mask_rejected(self):
        plan = self.plan(710)
        inputs, segmentation, reference = validate_plan_inputs(plan)
        prepared = prepare_lesion_channels(710, inputs, segmentation, reference)
        factory = Mock(side_effect=AssertionError("no predictor expected"))
        cases = (replace(prepared, channel_names=tuple(reversed(prepared.channel_names))),
                 replace(prepared, images=prepared.images[:-1]),
                 replace(prepared, images=(prepared.images[0], _new_image(np.full(prepared.crop.shape, 300, np.uint16), prepared.images[-1], prepared.images[-1].affine))))
        for invalid in cases:
            with tempfile.TemporaryDirectory(dir=self.root) as work, self.assertRaises(LesionInferenceError):
                infer_lesions(invalid, self.model(710), work, predictor_factory=factory)
        factory.assert_not_called()

    def test_publication_failure_preserves_existing_results(self):
        plan = self.plan(710)
        self.output.mkdir()
        sentinel = self.output / "old_result.txt"
        sentinel.write_bytes(b"old result")
        from benchmark_model_installation import _publish_directory
        def collide(stage, destination):
            destination.mkdir()
            (destination / "sentinel").write_bytes(b"existing run")
            return _publish_directory(stage, destination)
        with patch("benchmark_lesion_inference._publish_directory", side_effect=collide), self.assertRaises(FileExistsError):
            self.run_pipeline(plan)
        self.assertEqual(sentinel.read_bytes(), b"old result")
        self.assertFalse(list(self.output.glob(".v3-*")))
        self.assertFalse(list(self.output.rglob("run.json")))

    def test_concurrent_runs_do_not_collide(self):
        plan = self.plan(710)
        self.model(710)
        validated = validate_plan_inputs(plan)
        results, failures = [], []
        def run():
            try:
                results.append(run_lesion_pipeline(plan, validated, installer=self.installer(),
                                                   lesion_predictor_factory=lambda: LesionPredictor(plan.lesion_model)))
            except Exception as error:
                failures.append(error)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(failures)
        self.assertEqual(len(set(results)), 2)
        self.assertFalse(list(self.output.glob(".v3-*")))

    def test_cli_success_resolution_and_execution_errors(self):
        plan = self.plan(710)
        args = ["--input_dir", str(self.inputs), "--output_dir", str(self.output), "--lesion_model", "710", "--spine_segmentation", str(self.mask_path)]
        installer = self.installer()
        with patch("benchmark_lesion_inference.ModelInstaller", return_value=installer), patch("benchmark_lesion_inference.create_lesion_predictor", return_value=LesionPredictor(plan.lesion_model)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli(args), 0)
            installer.resolve.side_effect = InstallationError("missing model; use --install_models")
            self.assertEqual(cli(args), 4)
            with patch("benchmark_lesion_inference.ModelInstaller", side_effect=OSError("cache permission denied")):
                self.assertEqual(cli(args), 4)
            installer.resolve.side_effect = lambda dataset_id, **kwargs: self.model(dataset_id)
            with patch("benchmark_lesion_inference.create_lesion_predictor", return_value=LesionPredictor(plan.lesion_model, fail=True)):
                self.assertEqual(cli(args), 5)
            with patch("benchmark_lesion_inference._publish_directory", side_effect=InstallationError("native rename unavailable")):
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    self.assertEqual(cli(args), 5)
                self.assertIn("Cannot publish lesion run", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(len(list(self.output.glob("run-*"))), 1)
        self.assertFalse(list(self.output.glob(".v3-*")))

    def test_validation_only_imports_no_inference_dependencies_and_creates_nothing(self):
        self.plan(718)
        script = '''
import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','nnunetv2'):
        raise AssertionError('Inference import during validation')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from run_prediction_of_nnUNet_benchmark_models import cli
sys.exit(cli(sys.argv[1:]))
'''
        for supplied in (True, False):
            self.plan(718, supplied=supplied)
            args = [sys.executable, "-B", "-c", script, "--input_dir", str(self.inputs),
                    "--output_dir", str(self.output), "--lesion_model", "718", "--validate_only"]
            if supplied:
                args += ["--spine_segmentation", str(self.mask_path)]
            result = subprocess.run(args, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertFalse(self.output.exists())

    def test_prediction_label_policy_accepts_exact_float_and_rejects_invalid_values(self):
        dataset = {"labels": {"background": 0, "Lesions": 1}}
        for dtype in (np.uint8, np.int16, np.uint64, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                data = np.zeros(self.reference.shape, dtype)
                data[0, 0, 0] = 1
                prediction = _new_image(data, self.reference, self.reference.affine)
                original = np.asanyarray(prediction.dataobj).copy()
                self.assertIs(validate_lesion_prediction(prediction, self.reference, dataset), prediction)
                np.testing.assert_array_equal(prediction.dataobj, original)
        for value in (0.5, -1, 2, np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prediction = _new_image(np.full(self.reference.shape, value, np.float64), self.reference, self.reference.affine)
                validate_lesion_prediction(prediction, self.reference, dataset)
        empty = _new_image(np.zeros(self.reference.shape, np.float32), self.reference, self.reference.affine)
        self.assertIs(validate_lesion_prediction(empty, self.reference, dataset), empty)

    def test_exact_float_prediction_roundtrip_is_not_rounded(self):
        plan = self.plan(710)
        def as_float(prediction):
            return _new_image(np.asanyarray(prediction.dataobj).astype(np.float64), prediction, prediction.affine)
        result, _ = self.run_pipeline(plan, LesionPredictor(plan.lesion_model, transform=as_float))
        output = load_nifti(result / "lesion/lesion_labelled.nii.gz")
        self.assertEqual(output.get_data_dtype(), np.dtype("uint8"))
        self.assertEqual(int(np.asanyarray(output.dataobj).sum()), 2)

    def test_large_integer_validation_never_rounds_and_checks_allowed_values_exactly(self):
        from benchmark_spine_inference import validate_labels
        data = np.zeros(self.reference.shape, np.uint64)
        data[0, 0, 0] = 2**53 + 1
        data[-1, -1, -1] = np.iinfo(np.uint64).max
        segmentation = _new_image(data, self.reference, self.reference.affine)
        # np.rint(uint64) would allocate lossy float64. Integer validation must
        # not invoke it, even for the largest exactly represented integer label.
        with patch("benchmark_nifti.np.rint", side_effect=AssertionError("lossy rounding forbidden")):
            validate_labels(segmentation, allowed_labels={0, 2**53 + 1, 2**64 - 1})
            binary = binary_spine_mask(segmentation)
        self.assertEqual(int(binary.get_fdata().sum()), 2)
        with self.assertRaisesRegex(ValueError, "allowed set"):
            validate_labels(segmentation, allowed_labels={0, 2**53, 2**64 - 1})
        np.testing.assert_array_equal(segmentation.dataobj, data)

    def test_cli_model_metadata_failures_are_code_four_before_inference(self):
        plan = self.plan(710)
        resolved = self.model(710)
        directory = resolved.model_directory
        dataset_path, plans_path = directory / "dataset.json", directory / "plans.json"
        dataset_bytes, plans_bytes = dataset_path.read_bytes(), plans_path.read_bytes()
        checkpoint = directory / "fold_4/checkpoint_final.pth"
        checkpoint_bytes = checkpoint.read_bytes()
        args = ["--input_dir", str(self.inputs), "--output_dir", str(self.output),
                "--lesion_model", "710", "--spine_segmentation", str(self.mask_path)]
        cases = ("missing_dataset", "malformed_dataset", "wrong_channels", "wrong_labels", "missing_fold",
                 "missing_checkpoint", "empty_checkpoint", "wrong_configuration", "wrong_trainer_directory")
        for case in cases:
            with self.subTest(case=case):
                installer = self.installer()
                if case == "missing_dataset":
                    dataset_path.unlink()
                elif case == "malformed_dataset":
                    dataset_path.write_text("{not JSON")
                elif case in ("wrong_channels", "wrong_labels"):
                    dataset = json.loads(dataset_bytes)
                    dataset["channel_names" if case == "wrong_channels" else "labels"] = {"0": "wrong"}
                    dataset_path.write_text(json.dumps(dataset))
                elif case in ("missing_fold", "missing_checkpoint"):
                    checkpoint.unlink()
                    if case == "missing_fold":
                        checkpoint.parent.rmdir()
                elif case == "empty_checkpoint":
                    checkpoint.write_bytes(b"")
                elif case == "wrong_configuration":
                    plans = json.loads(plans_bytes)
                    plans["configurations"] = {"2d": {}}
                    plans_path.write_text(json.dumps(plans))
                else:
                    installer.resolve.side_effect = lambda *args, **kwargs: replace(resolved, model_directory=directory.parent / "WrongTrainer")
                error = io.StringIO()
                factory = Mock(side_effect=AssertionError("preflight must precede construction"))
                try:
                    with patch("benchmark_lesion_inference.ModelInstaller", return_value=installer), patch("benchmark_lesion_inference.create_lesion_predictor", factory), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                        self.assertEqual(cli(args), 4)
                    factory.assert_not_called()
                    self.assertFalse(self.output.exists())
                    self.assertNotIn("Traceback", error.getvalue())
                    self.assertIn("model resolution failed", error.getvalue())
                finally:
                    dataset_path.write_bytes(dataset_bytes)
                    plans_path.write_bytes(plans_bytes)
                    checkpoint.parent.mkdir(exist_ok=True)
                    checkpoint.write_bytes(checkpoint_bytes)

    def test_cli_execution_failure_matrix_preserves_outputs_and_cleans_staging(self):
        plan = self.plan(710)
        self.model(710)
        self.output.mkdir()
        sentinel = self.output / "unrelated.txt"
        sentinel.write_bytes(b"preserve")
        args = ["--input_dir", str(self.inputs), "--output_dir", str(self.output),
                "--lesion_model", "710", "--spine_segmentation", str(self.mask_path)]
        cases = ("initialization", "initialization_assertion", "execution", "execution_assertion",
                 "wrong_loaded_trainer", "wrong_loaded_configuration", "invalid_geometry", "fractional_labels",
                 "channel_preparation", "reconstruction", "output_write", "manifest_write", "native_rename", "collision")
        for case in cases:
            with self.subTest(case=case), contextlib.ExitStack() as stack:
                fake = LesionPredictor(plan.lesion_model)
                if case.startswith("initialization"):
                    exception = AssertionError("model rejected") if case.endswith("assertion") else RuntimeError("initialization failed")
                    fake.initialize_from_trained_model_folder = Mock(side_effect=exception)
                elif case.startswith("execution"):
                    exception = AssertionError("request rejected") if case.endswith("assertion") else RuntimeError("prediction failed")
                    fake.predict_from_files_sequential = Mock(side_effect=exception)
                elif case == "wrong_loaded_trainer":
                    fake.trainer_name = "WrongTrainer"
                elif case == "wrong_loaded_configuration":
                    from types import SimpleNamespace
                    fake.configuration_manager = SimpleNamespace(configuration={"wrong": True})
                    fake.plans_manager = SimpleNamespace(get_configuration=lambda name: SimpleNamespace(configuration={}))
                elif case == "invalid_geometry":
                    fake.transform = lambda prediction: _new_image(np.zeros((2, 2, 2), np.uint8), prediction, prediction.affine)
                elif case == "fractional_labels":
                    fake.transform = lambda prediction: _new_image(np.full(prediction.shape, 0.5, np.float32), prediction, prediction.affine)
                else:
                    function = {"channel_preparation": "prepare_lesion_channels", "reconstruction": "reconstruct_lesions",
                                "output_write": "save_nifti_no_clobber", "manifest_write": "json.dump",
                                "native_rename": "_publish_directory", "collision": "_publish_directory"}[case]
                    exception = InstallationError("native rename failed") if case == "native_rename" else OSError("mock " + case)
                    if case == "manifest_write":
                        def fail_manifest(value, stream, **kwargs):
                            stream.write('{"status":"completed"')
                            raise OSError("manifest write failed")
                        stack.enter_context(patch("benchmark_lesion_inference." + function, side_effect=fail_manifest))
                    elif case == "collision":
                        from benchmark_model_installation import _publish_directory
                        def collide(stage, destination):
                            destination.mkdir()
                            (destination / "sentinel").write_bytes(b"existing run")
                            return _publish_directory(stage, destination)
                        stack.enter_context(patch("benchmark_lesion_inference." + function, side_effect=collide))
                    else:
                        stack.enter_context(patch("benchmark_lesion_inference." + function, side_effect=exception))
                stack.enter_context(patch("benchmark_lesion_inference.ModelInstaller", return_value=self.installer()))
                stack.enter_context(patch("benchmark_lesion_inference.create_lesion_predictor", return_value=fake))
                error = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                    self.assertEqual(cli(args), 5)
                self.assertNotIn("Traceback", error.getvalue())
                self.assertIn("V3 lesion pipeline failed", error.getvalue())
                self.assertFalse(list(self.output.glob(".v3-*")))
                self.assertFalse(list(self.output.rglob("run.json")))
                self.assertEqual(sentinel.read_bytes(), b"preserve")
                for destination in self.output.glob("run-*"):
                    self.assertEqual((destination / "sentinel").read_bytes(), b"existing run")

    def test_automatic_spine_failure_cannot_publish_a_lesion_run(self):
        plan = self.plan(710, supplied=False)
        lesion_factory = Mock(side_effect=AssertionError("spine failure must stop lesion prediction"))
        with self.assertRaisesRegex(RuntimeError, "second split failed"):
            run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=self.installer(),
                                spine_predictor_factory=lambda: FakePredictor(fail_second=True),
                                lesion_predictor_factory=lesion_factory)
        lesion_factory.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])
        error = io.StringIO()
        with patch("benchmark_lesion_inference.ModelInstaller", return_value=self.installer()), patch("benchmark_spine_inference.create_predictor", return_value=FakePredictor(fail_second=True)), patch("benchmark_lesion_inference.create_lesion_predictor", lesion_factory), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            self.assertEqual(cli(["--input_dir", str(self.inputs), "--output_dir", str(self.output), "--lesion_model", "710"]), 5)
        self.assertNotIn("Traceback", error.getvalue())
        lesion_factory.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_common_channel_metadata_never_hides_above_tolerance_misalignment(self):
        plan = self.plan(717)
        path = self.inputs / "VMI_80.nii.gz"
        image = load_nifti(path)
        affine = image.affine.copy()
        affine[:3, 3] += 0.0007  # Each component <0.001, Euclidean offset >0.001 mm.
        nib.save(_new_image(np.asanyarray(image.dataobj).copy(), image, affine), path)
        source_bytes = path.read_bytes()
        images = {name: load_nifti(self.inputs / (name + ".nii.gz")) for name in plan.lesion_model.required_modalities}
        with self.assertRaisesRegex(ValueError, "alignment"):
            prepare_lesion_channels(717, images, load_nifti(self.mask_path))
        with patch("benchmark_lesion_inference.ModelInstaller", side_effect=AssertionError("validation must precede resolution")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli(["--input_dir", str(self.inputs), "--output_dir", str(self.output), "--lesion_model", "717", "--spine_segmentation", str(self.mask_path), "--validate_only"]), 2)
        self.assertEqual(path.read_bytes(), source_bytes)
        self.assertFalse(self.output.exists())

    def test_programmer_type_error_is_not_disguised_as_an_operational_failure(self):
        plan = self.plan(710)
        with self.assertRaisesRegex(TypeError, "programmer bug"):
            run_lesion_pipeline(plan, validate_plan_inputs(plan), installer=self.installer(),
                                lesion_predictor_factory=Mock(side_effect=TypeError("programmer bug")))
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
