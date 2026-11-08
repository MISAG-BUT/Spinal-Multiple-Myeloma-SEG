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
    validate_spine_segmentation,
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
            import os
            real_link = os.link
            def racing_link(src, dst):
                Path(dst).write_bytes(b"concurrent writer")
                return real_link(src, dst)
            with patch("benchmark_nifti.os.link", side_effect=racing_link):
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
                self.assertEqual(cli(args), 2)
                installer.assert_not_called()
                nib.save(image(), path)
                installer.return_value.resolve_plan.return_value = {802: ResolvedModel(root / "model", 802, "structural-only", {})}
                self.assertEqual(cli(args), 3)
                installer.assert_called_once_with(root / "cache")
                installer.return_value.resolve_plan.side_effect = InstallationError("mock failure")
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
                self.assertEqual(cli(supplied), 3)
                self.assertFalse(output.exists())
                _, segmentation, reference = validate_plan_inputs(parse_arguments(supplied))
                compare_geometry(reference, segmentation)
                spine_only = base + ["--skip_lesion_segmentation", "--spine_segmentation", str(mask)]
                self.assertEqual(cli(spine_only + ["--validate_only"]), 3)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*.nii.gz")})
            nib.save(image(), inputs / "ConvCT.nii.gz")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for split in ("True", "False"):
                    self.assertEqual(cli(base + ["--lesion_model", "710", "--split", split]), 3)
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
