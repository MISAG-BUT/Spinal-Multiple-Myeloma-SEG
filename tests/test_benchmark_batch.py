"""Prepared batch discovery, predictor lifecycle, and publication contracts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from run_prediction_of_nnUNet_benchmark_models import (
    cli, parse_arguments, prepared_filenames, validate_case_id,
)
import test_benchmark_lesion_inference as lesion_fixtures
from test_benchmark_lesion_inference import LesionPredictor
from utils.benchmark.batch_inference import discover_cases
from utils.benchmark.model_installation import InstallationError, _publish_directory
from utils.benchmark.model_registry import get_lesion_model
from utils.benchmark.nifti import _new_image, publish_case_run


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = lesion_fixtures.LesionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.inputs = self.fixture.inputs
        self.output = self.fixture.output

    def args(self, model=709):
        return ["--nnunet_input_dir", str(self.inputs), "--lesion_model", str(model),
                "--output_dir", str(self.output), "--batch", "--offline"]

    def write_case(self, case, model=709, *, omit=(), mask=1, offset=0):
        reference = self.fixture.reference
        for index, channel in enumerate(get_lesion_model(model).channel_names):
            if index in omit:
                continue
            affine = reference.affine.copy()
            data = np.full(reference.shape, index + 10, np.int16)
            if channel == "mask_spine":
                data.fill(0)
                data[11, 12, 13] = mask
                affine[0, 3] += offset
            nib.save(_new_image(data, reference, affine), self.inputs / f"{case}_{index:04d}.nii.gz")

    def call(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            result = cli(args)
        return result, out.getvalue() + err.getvalue()

    def test_discover_sorted_multichannel_cases(self):
        case_ids = ["Myel_3", "Myel-2_part", "A-1"]
        for case in case_ids:
            self.write_case(case, 708)
        plan = parse_arguments(self.args(708)).template
        discovered, cases, errors = discover_cases(plan, validate_case_id, prepared_filenames)
        self.assertEqual(discovered, tuple(sorted(case_ids)))
        self.assertEqual([p.case_id for p in cases], sorted(case_ids))
        self.assertEqual(errors, {})
        for case in cases:
            self.assertEqual(tuple(n for n, _ in case.modality_paths), get_lesion_model(708).channel_names)
            self.assertEqual(len(case.modality_paths), 9)

    def test_single_patient_validate_only_is_read_only(self):
        self.write_case("A")
        before = {p.name: p.read_bytes() for p in self.inputs.iterdir()}
        with patch("utils.benchmark.batch_inference.ModelInstaller") as installer:
            status, log = self.call(self.args() + ["--validate_only"])
        self.assertEqual(status, 3)
        self.assertIn("1 discovered, 1 valid, 0 invalid", log)
        installer.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.inputs.iterdir()})

    def test_empty_directory(self):
        code, log = self.call(self.args())
        self.assertEqual(code, 2)
        self.assertIn("0 discovered, 0 valid, 0 invalid", log)
        self.assertFalse(self.output.exists())

    def test_preflight_reports_every_bad_case_without_installing(self):
        self.write_case("good", 717)
        self.write_case("missing_first", 717, omit=(0,))
        self.write_case("missing_middle", 717, omit=(1,))
        self.write_case("missing_final", 717, omit=(3,))
        self.write_case("bad_geometry", 717, offset=5)
        self.write_case("bad_mask", 717, mask=2)
        self.write_case("empty_mask", 717, mask=0)
        self.write_case("bad prefix", 717)
        self.write_case("extra", 717)
        (self.inputs / "extra_0004.nii.gz").touch()
        (self.inputs / "malformed.nii.gz").touch()
        with patch("utils.benchmark.batch_inference.ModelInstaller") as installer:
            status, log = self.call(self.args(717))
        self.assertEqual(status, 2)
        self.assertIn("10 discovered, 1 valid, 9 invalid", log)
        for name in ("missing_first", "missing_middle", "missing_final", "bad_geometry",
                     "bad_mask", "empty_mask", "bad prefix", "extra", "malformed.nii.gz"):
            self.assertIn(name, log)
        installer.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_hard_link_alias_between_cases(self):
        self.write_case("A")
        self.write_case("B", omit=(0,))
        os.link(self.inputs / "A_0000.nii.gz", self.inputs / "B_0000.nii.gz")
        code, log = self.call(self.args() + ["--validate_only"])
        self.assertEqual(code, 2)
        self.assertIn("Aliased input paths", log)
        self.assertIn("2 discovered, 0 valid, 2 invalid", log)

    def test_batch_argument_conflicts(self):
        for extra in (["--case_id", "A"], ["--nnunet_case_prefix", "A"], ["--split", "False"],
                      ["--input_dir", str(self.inputs)], ["--list_models"],
                      ["--spine_segmentation", "mask.nii.gz"], ["--install_models"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(self.args() + extra)
        for args in (["--batch"], ["--batch", "--nnunet_input_dir", str(self.inputs)]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(args)

    def execute(self, fake=None, *, flags=(), offline=True):
        installer = self.fixture.installer()
        fake = fake or LesionPredictor(get_lesion_model(709))
        fake.initialize_from_trained_model_folder = Mock(wraps=fake.initialize_from_trained_model_folder)
        args = self.args() + list(flags)
        if not offline:
            args.remove("--offline")
        with patch("utils.benchmark.batch_inference.ModelInstaller", return_value=installer), patch(
            "utils.benchmark.lesion_inference.create_lesion_predictor", return_value=fake
        ) as factory:
            code, log = self.call(args)
        factory.assert_called_once()
        fake.initialize_from_trained_model_folder.assert_called_once()
        installer.resolve.assert_called_once_with(709, allow_download=not offline)
        return code, log

    def assert_manifest_files(self, folder):
        manifest = json.loads((folder / "run.json").read_text())
        expected = {"run.json", *manifest["intermediates"]}
        expected.update(manifest[k] for k in ("binary_output", "lesion_output", "supplied_spine_copy"))
        self.assertEqual({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}, expected)
        self.assertFalse(list(folder.rglob("*_spine_binary.nii")))
        return manifest

    def test_batch_reuses_model_and_publishes_versions_and_manifest(self):
        self.write_case("B-2")
        self.write_case("A_1")
        before = {p.name: p.read_bytes() for p in self.inputs.iterdir()}
        flags = ["--keep_intermediates"]
        self.assertEqual(self.execute(flags=flags)[0], 0)
        existing = (self.output / "batch_run.json").read_bytes()
        self.assertEqual(self.execute(flags=flags + ["--already_cropped"], offline=False)[0], 0)
        self.assertEqual((self.output / "batch_run.json").read_bytes(), existing)
        first = json.loads(existing)
        second = json.loads((self.output / "batch_run_v2.json").read_text())
        self.assertEqual(first["discovered_case_ids"], ["A_1", "B-2"])
        self.assertEqual(first["counts"], {"succeeded": 2, "failed": 0, "unprocessed": 0})
        self.assertEqual(first["predictor_initializations"], 1)
        self.assertEqual(first["model"]["ordered_channels"], ["ConvCT", "mask_spine"])
        self.assertNotEqual(first["batch_id"], second["batch_id"])
        self.assertTrue(first["started_at"] <= first["completed_at"])
        self.assertGreater(first["duration_seconds"], 0)
        for case_id in first["cases"]:
            one, two = (Path(r["cases"][case_id]["output_dir"]) for r in (first, second))
            self.assertEqual(one.name, f"{case_id}_output")
            self.assertEqual(two.name, f"{case_id}_output_v2")
            self.assertTrue(self.assert_manifest_files(one)["intermediates"])
            self.assertEqual(self.assert_manifest_files(two)["application_crop"], "already_cropped")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.inputs.iterdir()})

    def test_invalid_prediction_can_continue(self):
        self.write_case("A")
        self.write_case("B")
        calls = iter((False, True))
        def transform(image):
            if next(calls):
                return image
            return _new_image(np.full(image.shape, 2, np.uint8), image, image.affine)
        self.assertEqual(self.execute(LesionPredictor(get_lesion_model(709), transform=transform))[0], 5)
        report = json.loads((self.output / "batch_run.json").read_text())
        self.assertEqual(report["status"], "partial_failure")
        self.assertEqual(report["cases"]["A"]["status"], "failed")
        self.assertIn("labels", report["cases"]["A"]["error"])
        self.assertEqual(report["cases"]["B"]["status"], "succeeded")
        self.assertFalse((self.output / "A_output").exists())
        self.assertFalse(list(self.output.glob(".v3-*")))

    def test_predictor_error_aborts_and_records_unprocessed(self):
        self.write_case("A")
        self.write_case("B")
        self.assertEqual(self.execute(LesionPredictor(get_lesion_model(709), fail=True))[0], 5)
        report = json.loads((self.output / "batch_run.json").read_text())
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["counts"], {"succeeded": 0, "failed": 1, "unprocessed": 1})
        self.assertIn("mock lesion predictor failure", report["error"])
        self.assertIsNone(report["cases"]["B"]["duration_seconds"])

    def test_offline_missing_model_aborts_before_initialization(self):
        self.write_case("A")
        resolver = Mock(side_effect=InstallationError("offline cache missing"))
        with patch("utils.benchmark.batch_inference.ModelInstaller", return_value=Mock(resolve=resolver)), patch(
            "utils.benchmark.batch_inference.initialize_lesion_predictor"
        ) as initialize:
            self.assertEqual(self.call(self.args())[0], 4)
        initialize.assert_not_called()
        report = json.loads((self.output / "batch_run.json").read_text())
        self.assertEqual(report["cases"]["A"]["status"], "unprocessed")


class VersioningTests(unittest.TestCase):
    def test_gap_and_legacy_directory_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("scan_output", "scan_output_v5", "scan_output-abcdef"):
                directory = root / name
                directory.mkdir()
                (directory / "sentinel").write_text(name)
            stage = root / "stage"
            stage.mkdir()
            self.assertEqual(publish_case_run(stage, root, "scan", _publish_directory).name, "scan_output_v6")
            self.assertFalse((root / "scan_output_v2").exists())
            for name in ("scan_output", "scan_output_v5", "scan_output-abcdef"):
                self.assertEqual((root / name / "sentinel").read_text(), name)

    def test_concurrent_report_collision_and_gap(self):
        from utils.benchmark.batch_inference import _publish_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _publish_report(root, {"id": "original"})
            self.assertEqual(first.name, "batch_run.json")
            (root / "batch_run_v4.json").write_text('{"id": "preserve"}')
            barrier = threading.Barrier(2)
            def publish(value):
                barrier.wait(timeout=5)
                return _publish_report(root, {"id": value})
            with ThreadPoolExecutor(2) as pool:
                paths = list(pool.map(publish, (1, 2)))
            self.assertEqual({p.name for p in paths}, {"batch_run_v5.json", "batch_run_v6.json"})
            self.assertEqual({json.loads(p.read_text())["id"] for p in paths}, {1, 2})
            self.assertEqual(json.loads(first.read_text())["id"], "original")
            self.assertEqual(json.loads((root / "batch_run_v4.json").read_text())["id"], "preserve")
            self.assertFalse(list(root.glob(".v3-*")))
