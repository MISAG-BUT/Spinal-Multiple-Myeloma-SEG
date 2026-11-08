"""Synthetic offline geometry and preprocessing tests; no models or GPU needed."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from benchmark_nifti import (
    Crop, binary_spine_mask, compare_geometry, crop_image, load_nifti,
    padded_bounding_box, prepare_lesion_channels, reconstruct_lesions,
    save_prepared_channels, validate_image, validate_plan_inputs,
    validate_spine_segmentation, save_nifti_no_clobber,
)
from benchmark_model_registry import get_lesion_model
from run_prediction_of_nnUNet_benchmark_models import cli, parse_arguments


def image(data=None, affine=None):
    if data is None:
        data = np.zeros((21, 23, 25), dtype=np.float32)
    if affine is None:
        affine = np.diag([0.7, 0.9, 1.4, 1.0])
        affine[:3, 3] = [12, -8, 32]
    result = nib.Nifti1Image(np.asarray(data), affine)
    result.header.set_xyzt_units("mm")
    result.set_qform(affine, 1)
    result.set_sform(affine, 1)
    return result


def spine():
    data = np.zeros((21, 23, 25), dtype=np.int16)
    data[10, 11, 12] = 7
    data[11, 12, 13] = 28
    return image(data)


class GeometryTests(unittest.TestCase):
    def test_matching_and_subtolerance_geometry(self):
        compare_geometry(image(), image())
        affine = image().affine.copy()
        affine[0, 3] += 0.0005
        compare_geometry(image(), image(affine=affine))

    def test_shape_spacing_translation_orientation(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_geometry(image(), image(np.zeros((20, 23, 25), np.float32)))
        for axis, value, message in ((0, 0.01, "spacing"), (3, 0.002, "alignment")):
            affine = image().affine.copy()
            affine[0, axis] += value
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                compare_geometry(image(), image(affine=affine))
        affine = image().affine.copy()
        affine[0, 0] *= -1
        with self.assertRaisesRegex(ValueError, "RAS"):
            validate_image(image(affine=affine))

    def test_corner_accumulation_and_qform_conflict(self):
        affine = image().affine.copy()
        affine[0, 0] += 0.0001
        with self.assertRaisesRegex(ValueError, "alignment"):
            compare_geometry(image(), image(affine=affine))
        conflicting = image()
        affine = conflicting.affine.copy()
        affine[0, 3] += 1
        conflicting.set_qform(affine, 1)
        with self.assertRaisesRegex(ValueError, "qform/sform"):
            validate_image(conflicting)

    def test_invalid_data_dimensions_affines_and_units(self):
        for shape in ((2, 2), (2, 2, 2, 1), (0, 3, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "3D"):
                validate_image(image(np.zeros(shape, np.float32)))
        for value in (np.nan, np.inf, -np.inf):
            data = np.zeros((2, 2, 2), np.float32)
            data[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                validate_image(image(data))
        invalid = image()
        invalid._affine[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_image(invalid)
        invalid = image()
        invalid._affine[0, :3] = 0
        with self.assertRaisesRegex(ValueError, "invertible"):
            validate_image(invalid)
        invalid = image()
        invalid.header.set_xyzt_units("unknown")
        with self.assertRaisesRegex(ValueError, "units"):
            validate_image(invalid)

    def test_declared_unit_conversion(self):
        source = image()
        affine = source.affine.copy()
        affine[:3] /= 1000
        meters = image(affine=affine)
        meters.header.set_xyzt_units("meter")
        compare_geometry(source, meters)

    def test_unreadable_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.nii.gz"
            nib.save(spine(), path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            source = load_nifti(path)
            crop_image(binary_spine_mask(source), padded_bounding_box(binary_spine_mask(source)))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                load_nifti(path)


class PreprocessingTests(unittest.TestCase):
    def test_independent_boundary_lesions_and_atomic_writes(self):
        reference = image()
        source_data = reference.get_fdata().copy()
        crop = Crop((0, 4, 5), (10, 23, 25), reference.shape)
        cropped = crop_image(reference, crop)
        data = np.zeros(crop.shape, np.uint8)
        data[0, 0, 0] = data[-1, -1, -1] = 1
        prediction = nib.Nifti1Image(data, cropped.affine, cropped.header)
        expected = np.zeros(reference.shape, np.uint8)
        expected[0, 4, 5] = expected[9, 22, 24] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lesions.nii.gz"
            result = reconstruct_lesions(prediction, crop, reference, path)
            np.testing.assert_array_equal(result.dataobj, expected)
            np.testing.assert_array_equal(result.affine, reference.affine)
            self.assertEqual(result.shape, reference.shape)
            original = path.read_bytes()
            for _ in range(2):
                with self.assertRaises(FileExistsError):
                    reconstruct_lesions(prediction, crop, reference, path)
                self.assertEqual(path.read_bytes(), original)
            failed = Path(tmp) / "failed.nii.gz"
            with patch("benchmark_nifti.nib.save", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    reconstruct_lesions(prediction, crop, reference, failed)
            self.assertFalse(failed.exists())
            # A writer that returns without completing a NIfTI must not publish.
            with patch("benchmark_nifti.nib.save"):
                with self.assertRaisesRegex(ValueError, "Cannot read"):
                    reconstruct_lesions(prediction, crop, reference, failed)
            self.assertFalse(failed.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            # Simulate a destination appearing between temporary write and publish.
            from benchmark_nifti import _rename_no_clobber
            def racing_rename(src, dst):
                Path(dst).write_bytes(b"concurrent writer")
                return _rename_no_clobber(src, dst)
            with patch("benchmark_nifti._rename_no_clobber", side_effect=racing_rename):
                with self.assertRaises(FileExistsError):
                    reconstruct_lesions(prediction, crop, reference, failed)
            self.assertEqual(failed.read_bytes(), b"concurrent writer")
            self.assertFalse(list(Path(tmp).glob(".v3-*")))
        np.testing.assert_array_equal(reference.dataobj, source_data)

    def test_multilabel_binarization_and_invalid_masks(self):
        source = spine()
        original = np.asanyarray(source.dataobj).copy()
        binary = binary_spine_mask(source)
        self.assertEqual(binary.get_data_dtype(), np.dtype("uint8"))
        np.testing.assert_array_equal(binary.dataobj, original != 0)
        np.testing.assert_array_equal(source.dataobj, original)
        validate_spine_segmentation(binary)
        for value, message in ((0, "empty"), (0.5, "integer"), (np.nan, "finite"), (np.inf, "finite")):
            data = np.zeros((3, 3, 3), np.float32)
            data[1, 1, 1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                binary_spine_mask(image(data))
        data = np.zeros((3, 3, 3), np.int16)
        data[1, 1, 1] = -2
        self.assertEqual(binary_spine_mask(image(data)).get_fdata().sum(), 1)

    def test_padding_exclusive_bounds_and_edges(self):
        binary = binary_spine_mask(spine())
        crop = padded_bounding_box(binary)
        self.assertEqual(crop.start, (5, 6, 7))
        self.assertEqual(crop.stop, (17, 18, 19))
        self.assertEqual(crop.shape, (12, 12, 12))
        data = np.zeros((13, 15, 17), np.uint8)
        data[0, 0, 0] = data[-1, -1, -1] = 1
        edges = padded_bounding_box(image(data))
        self.assertEqual(edges.start, (0, 0, 0))
        self.assertEqual(edges.stop, data.shape)
        with self.assertRaisesRegex(ValueError, "binary"):
            padded_bounding_box(spine())
        with self.assertRaises(ValueError):
            Crop((0, 0, 0), (99, 1, 1), (3, 3, 3))

    def test_translated_affine_and_physical_locations(self):
        source = image()
        crop = padded_bounding_box(binary_spine_mask(spine()))
        cropped = crop_image(source, crop)
        for voxel in ((0, 0, 0), (4, 5, 6), (11, 11, 11)):
            np.testing.assert_allclose(nib.affines.apply_affine(cropped.affine, voxel),
                                       nib.affines.apply_affine(source.affine, np.array(voxel) + crop.start))
        self.assertEqual(cropped.header.get_xyzt_units(), source.header.get_xyzt_units())
        np.testing.assert_allclose(cropped.get_qform(), cropped.get_sform(), atol=1e-5)

    def test_shear_and_uncoded_qform_preserve_exact_sform(self):
        affine = image().affine.copy()
        affine[0, 1] = 0.1
        source = image(affine=affine)
        source.set_qform(None, 0)
        validate_image(source)
        crop = padded_bounding_box(binary_spine_mask(spine()))
        result = crop_image(source, crop)
        validate_image(result)
        self.assertEqual(int(result.header["qform_code"]), 0)
        translation = np.eye(4)
        translation[:3, 3] = crop.start
        np.testing.assert_allclose(result.get_sform(), affine @ translation, atol=1e-5)

    def test_header_spacing_and_prediction_nonfinite_rejection(self):
        invalid = image()
        invalid.header["pixdim"][1] = 0
        with self.assertRaisesRegex(ValueError, "spacing"):
            validate_image(invalid)
        prepared = prepare_lesion_channels(710, {"VMI_40": image()}, spine())
        prediction = prepared.images[-1]
        invalid = nib.Nifti1Image(np.full(prepared.crop.shape, np.nan, np.float32), prediction.affine, prediction.header)
        with self.assertRaisesRegex(ValueError, "finite"):
            reconstruct_lesions(invalid, prepared.crop, prepared.reference)

    def test_model_710_without_convct_and_shared_crop(self):
        data = np.indices((21, 23, 25))[2].astype(np.float32)
        prepared = prepare_lesion_channels(710, {"VMI_40": image(data)}, spine())
        self.assertEqual(prepared.channel_names, ("VMI_40", "mask_spine"))
        self.assertEqual(prepared.filenames, ("case_0001_0000.nii.gz", "case_0001_0001.nii.gz"))
        np.testing.assert_array_equal(prepared.images[0].dataobj, data[prepared.crop.slices])
        compare_geometry(*prepared.images)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_prepared_channels(prepared, Path(tmp) / "channels")
            self.assertEqual([p.name for p in paths], list(prepared.filenames))
            compare_geometry(load_nifti(paths[0]), prepared.images[0])
            with self.assertRaises(FileExistsError):
                save_prepared_channels(prepared, Path(tmp) / "channels")

    def test_model_708_order_and_missing_channels(self):
        names = get_lesion_model(708).required_modalities
        inputs = {name: image(np.full((21, 23, 25), index + 1, np.float32)) for index, name in reversed(list(enumerate(names)))}
        prepared = prepare_lesion_channels(708, inputs, spine())
        self.assertEqual(prepared.channel_names, names + ("mask_spine",))
        self.assertEqual(len(prepared.images), 9)
        for index, channel in enumerate(prepared.images[:-1]):
            self.assertTrue(np.all(channel.get_fdata() == index + 1))
        with self.assertRaisesRegex(ValueError, "Missing required"):
            prepare_lesion_channels(708, {}, spine())

    def test_reconstruction_roundtrip_and_rejection(self):
        reference = image()
        prepared = prepare_lesion_channels(710, {"VMI_40": reference}, spine())
        prediction = prepared.images[-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lesions.nii.gz"
            full = reconstruct_lesions(prediction, prepared.crop, reference, path)
            np.testing.assert_array_equal(full.dataobj, binary_spine_mask(spine()).dataobj)
            compare_geometry(reference, full)
            compare_geometry(reference, load_nifti(path))
            with self.assertRaises(FileExistsError):
                reconstruct_lesions(prediction, prepared.crop, reference, path)
        with self.assertRaisesRegex(ValueError, "shape"):
            reconstruct_lesions(image(), prepared.crop, reference)
        wrong = image(np.zeros(prepared.crop.shape, np.uint8))
        with self.assertRaisesRegex(ValueError, "alignment"):
            reconstruct_lesions(wrong, prepared.crop, reference)
        bad = nib.Nifti1Image(np.full(prepared.crop.shape, 2, np.uint8), prediction.affine, prediction.header)
        with self.assertRaisesRegex(ValueError, "binary"):
            reconstruct_lesions(bad, prepared.crop, reference)


class NiftiPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="V3 NIfTI ž ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_success_without_hard_links_preserves_data_and_coded_geometry(self):
        data = np.zeros((5, 7, 9), np.uint16)
        data[0, 0, 0], data[-1, -1, -1] = 300, 65535
        source = image(data)
        source.set_sform(None, 0)
        path = self.root / "labelled ž spine.nii.gz"
        from benchmark_nifti import _rename_no_clobber
        def publish(temporary, destination):
            self.assertEqual(temporary.parent, destination.parent)
            self.assertTrue(str(temporary).endswith(".nii.gz"))
            self.assertFalse(destination.exists())
            np.testing.assert_array_equal(load_nifti(temporary).dataobj, data)
            return _rename_no_clobber(temporary, destination)
        with patch("benchmark_nifti.os.link", side_effect=OSError("hard links unsupported")) as link, patch("benchmark_nifti._rename_no_clobber", side_effect=publish):
            save_nifti_no_clobber(source, path)
        link.assert_not_called()
        written = load_nifti(path)
        np.testing.assert_array_equal(written.dataobj, data)
        np.testing.assert_array_equal(written.affine, source.affine)
        self.assertEqual(written.get_data_dtype(), data.dtype)
        self.assertEqual(written.header.get_zooms(), source.header.get_zooms())
        self.assertEqual(written.header.get_xyzt_units(), source.header.get_xyzt_units())
        for kind in ("qform", "sform"):
            expected, code = getattr(source, "get_" + kind)(coded=True)
            actual, actual_code = getattr(written, "get_" + kind)(coded=True)
            self.assertEqual(int(code), int(actual_code))
            if expected is not None:
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source.dataobj, data)
        self.assertEqual(list(self.root.iterdir()), [path])
        original_bytes = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_nifti_no_clobber(image(np.zeros(data.shape, np.uint16)), path)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_native_rename_flags_and_unsupported_platform_fail_closed(self):
        import errno
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock
        from benchmark_nifti import _rename_no_clobber
        source, destination = self.root / "temporary", self.root / "final"
        for platform, function, arguments in (
            ("linux", "renameat2", (-100, os.fsencode(source), -100, os.fsencode(destination), 1)),
            ("darwin", "renamex_np", (os.fsencode(source), os.fsencode(destination), 4)),
        ):
            operation = Mock(return_value=0)
            library = SimpleNamespace(**{function: operation})
            with self.subTest(platform=platform), patch("benchmark_nifti.os.name", "posix"), patch("benchmark_nifti.sys.platform", platform), patch("benchmark_nifti.ctypes.CDLL", return_value=library):
                _rename_no_clobber(source, destination)
                operation.assert_called_once_with(*arguments)
                operation.return_value = -1
                with patch("benchmark_nifti.ctypes.get_errno", return_value=errno.EEXIST):
                    with self.assertRaises(FileExistsError):
                        _rename_no_clobber(source, destination)
                with patch("benchmark_nifti.ctypes.get_errno", return_value=errno.ENOSYS):
                    with self.assertRaises(OSError):
                        _rename_no_clobber(source, destination)
        with patch("benchmark_nifti.os.name", "posix"), patch("benchmark_nifti.sys.platform", "linux"), patch("benchmark_nifti.ctypes.CDLL", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(OSError, "unavailable"):
                _rename_no_clobber(source, destination)

    def test_spine_cli_publication_failure_is_execution_error_and_cleans_run(self):
        import errno
        inputs = self.root / "inputs"
        inputs.mkdir()
        mask = self.root / "supplied.nii.gz"
        nib.save(image(np.ones((5, 7, 9), np.uint16)), mask)
        original_bytes = mask.read_bytes()
        output = self.root / "outputs"
        error = io.StringIO()
        with patch("benchmark_nifti._rename_no_clobber", side_effect=OSError(errno.ENOTSUP, "rename unavailable")), patch("benchmark_spine_inference.ModelInstaller", side_effect=AssertionError("supplied mask bypasses models")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            result = cli(["--input_dir", str(inputs), "--output_dir", str(output),
                          "--spine_segmentation", str(mask), "--skip_lesion_segmentation"])
        self.assertEqual(result, 5)
        self.assertIn("rename unavailable", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(list(output.iterdir()), [])
        self.assertEqual(mask.read_bytes(), original_bytes)

    def test_repeated_publication_preserves_existing_destinations(self):
        source = image()
        for existing in (b"", b"existing result"):
            path = self.root / "existing.nii.gz"
            path.write_bytes(existing)
            for _ in range(2):
                with self.assertRaises(FileExistsError):
                    save_nifti_no_clobber(source, path)
                self.assertEqual(path.read_bytes(), existing)
            path.unlink()
        destination = self.root / "directory.nii.gz"
        destination.mkdir()
        with self.assertRaises(OSError):
            save_nifti_no_clobber(source, destination)
        self.assertTrue(destination.is_dir())
        self.assertFalse(list(self.root.glob(".v3-nifti-*")))

    def test_publication_and_fsync_failures_cleanup_without_final_file(self):
        import errno
        sentinel = self.root / "unrelated.txt"
        sentinel.write_bytes(b"preserve")
        path = self.root / "failed.nii.gz"
        for operation, error in (("_rename_no_clobber", PermissionError("denied")),
                                 ("_rename_no_clobber", OSError(errno.EXDEV, "cross-device")),
                                 ("_rename_no_clobber", OSError(errno.ENOTSUP, "exclusive rename unavailable")),
                                 ("os.fsync", OSError("fsync failed"))):
            with self.subTest(operation=operation, error=error):
                with patch("benchmark_nifti." + operation, side_effect=error):
                    with self.assertRaises(OSError) as caught:
                        save_nifti_no_clobber(image(), path)
                self.assertIsInstance(caught.exception, OSError)
                self.assertFalse(path.exists())
                self.assertEqual(list(self.root.iterdir()), [sentinel])
                self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_changed_saved_voxels_never_publish(self):
        original_save = nib.save
        def corrupt(source, filename):
            original_save(image(np.ones(source.shape, np.float32)), filename)
        with patch("benchmark_nifti.nib.save", side_effect=corrupt), patch("benchmark_nifti._rename_no_clobber") as publish:
            with self.assertRaisesRegex(ValueError, "voxel verification"):
                save_nifti_no_clobber(image(), self.root / "invalid.nii.gz")
        publish.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_concurrent_publishers_have_one_winner_and_no_partial_file(self):
        import threading
        from benchmark_nifti import _rename_no_clobber
        barrier = threading.Barrier(2)
        outcomes, failures = [], []
        destination = self.root / "shared.nii.gz"
        def publish(temporary, path):
            barrier.wait(timeout=10)
            return _rename_no_clobber(temporary, path)
        def work(value):
            try:
                save_nifti_no_clobber(image(np.full((5, 7, 9), value, np.uint16)), destination)
                outcomes.append(value)
            except Exception as error:
                failures.append(error)
        with patch("benchmark_nifti._rename_no_clobber", side_effect=publish):
            threads = [threading.Thread(target=work, args=(value,)) for value in (300, 400)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=15)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], FileExistsError)
        np.testing.assert_array_equal(load_nifti(destination).dataobj,
                                      np.full((5, 7, 9), outcomes[0], np.uint16))
        self.assertEqual(list(self.root.iterdir()), [destination])


class CliGeometryTests(unittest.TestCase):
    def test_explicit_installation_happens_only_after_validation(self):
        from benchmark_model_installation import InstallationError, ResolvedModel
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            path = inputs / "ConvCT.nii.gz"
            path.write_bytes(b"invalid NIfTI")
            args = ["--input_dir", str(inputs), "--output_dir", str(root / "outputs"),
                    "--lesion_model", "709", "--install_models", "--cache_dir", str(root / "cache")]
            with patch("benchmark_model_installation.ModelInstaller") as installer, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from benchmark_model_installation import ModelInstaller
                def execute(plan, validated):
                    ModelInstaller(plan.cache_dir).resolve(plan.lesion_model.dataset_id, allow_download=plan.install_models)
                    return root / "run"
                with patch("benchmark_lesion_inference.run_lesion_pipeline", side_effect=execute) as run:
                    self.assertEqual(cli(args), 2)
                    installer.assert_not_called()
                    run.assert_not_called()
                    nib.save(image(), path)
                    self.assertEqual(cli(args), 0)
                    installer.assert_called_once_with(root / "cache")
                    installer.return_value.resolve.assert_called_once_with(709, allow_download=True)
                    installer.return_value.resolve.side_effect = InstallationError("mock failure")
                    self.assertEqual(cli(args), 4)
            self.assertFalse((root / "outputs").exists())
            self.assertFalse((root / "cache").exists())

    def test_modes_validate_geometry_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            mask = root / "spine.nii.gz"
            nib.save(spine(), mask)
            nib.save(image(), inputs / "VMI_40.nii.gz")
            output = root / "output"
            base = ["--input_dir", str(inputs), "--output_dir", str(output)]
            supplied = base + ["--lesion_model", "710", "--spine_segmentation", str(mask)]
            before = {p: p.read_bytes() for p in root.rglob("*.nii.gz")}
            with patch("socket.socket", side_effect=AssertionError("network forbidden")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli(supplied + ["--validate_only"]), 3)
                self.assertFalse(output.exists())
                _, segmentation, reference = validate_plan_inputs(parse_arguments(supplied))
                compare_geometry(reference, segmentation)
                spine_only = base + ["--skip_lesion_segmentation", "--spine_segmentation", str(mask)]
                self.assertEqual(cli(spine_only + ["--validate_only"]), 3)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*.nii.gz")})
            nib.save(image(), inputs / "ConvCT.nii.gz")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for split in ("True", "False"):
                    self.assertEqual(cli(base + ["--lesion_model", "710", "--split", split, "--validate_only"]), 3)
            affine = image().affine.copy()
            affine[0, 3] += 1
            nib.save(image(affine=affine), inputs / "VMI_40.nii.gz")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(cli(supplied), 2)
            self.assertIn("alignment", error.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
