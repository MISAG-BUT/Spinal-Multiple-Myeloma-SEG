"""Independent V3 preprocessing; no registration, resampling, or normalization.

Geometry comparisons use voxel-centre corners and a Euclidean tolerance of
0.001 mm. Spatial units must be declared (mm, meter, or micron); unknown units
cannot establish physical alignment. Crop upper bounds are always exclusive.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import ctypes
import errno
import os
import sys
import tempfile
from typing import Tuple

import nibabel as nib
import numpy as np

from benchmark_model_registry import get_lesion_model


GEOMETRY_TOLERANCE_MM = 0.001
_MM_SCALE = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}


def _corners(shape):
    return np.array(list(product(*[(0, size - 1) for size in shape])), dtype=float)


def _distance(first, second, shape):
    points = _corners(shape)
    return float(np.max(np.linalg.norm(
        nib.affines.apply_affine(first, points) - nib.affines.apply_affine(second, points), axis=1)))


def _affine_mm(image, affine=None):
    unit = image.header.get_xyzt_units()[0]
    if unit not in _MM_SCALE:
        raise ValueError("Spatial units are unknown; declare the correct NIfTI spatial units before using V3")
    result = np.array(image.affine if affine is None else affine, dtype=float, copy=True)
    result[:3, :] *= _MM_SCALE[unit]
    return result


def _check_affine(affine, label):
    if affine is None or np.shape(affine) != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("{}: affine must be finite and 4x4".format(label))
    if not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("{}: invalid affine homogeneous row".format(label))
    if np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError("{}: affine is not invertible".format(label))


def validate_image(image, label="image"):
    """Validate an in-memory NIfTI and return it without modifying it."""
    try:
        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
            raise ValueError("Expected a NIfTI image")
        if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
            raise ValueError("Require nonempty 3D image dimensions; got {}".format(image.shape))
        _check_affine(image.affine, label)
        spacing = np.asarray(image.header.get_zooms()[:3])
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Voxel spacing must be positive and finite")
        if not np.allclose(spacing, nib.affines.voxel_sizes(image.affine), rtol=0, atol=GEOMETRY_TOLERANCE_MM / _MM_SCALE.get(image.header.get_xyzt_units()[0], 1)):
            raise ValueError("Header voxel spacing conflicts with affine spacing")
        affine_mm = _affine_mm(image)
        if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
            raise ValueError("Require RAS orientation; got {}. Prepare aligned RAS inputs externally; V3 does not reorient".format(nib.aff2axcodes(image.affine)))
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        for transform, code in ((qform, qcode), (sform, scode)):
            if code:
                _check_affine(transform, label)
                if _distance(affine_mm, _affine_mm(image, transform), image.shape) > GEOMETRY_TOLERANCE_MM:
                    raise ValueError("Coded qform/sform conflicts with effective affine")
        if qcode and scode and _distance(_affine_mm(image, qform), _affine_mm(image, sform), image.shape) > GEOMETRY_TOLERANCE_MM:
            raise ValueError("Materially conflicting qform and sform transforms")
        data = np.asanyarray(image.dataobj)
        if data.dtype.kind not in "biuf" or not np.isfinite(data).all():
            raise ValueError("Voxel values must be real and finite (no NaN or infinity)")
    except (ValueError, OSError, nib.spatialimages.HeaderDataError) as error:
        raise ValueError("{}: {}".format(label, error)) from error
    return image


def load_nifti(path):
    """Load and fully validate a 3D RAS NIfTI, including scaled voxel values."""
    try:
        return validate_image(nib.load(str(path)), str(path))
    except Exception as error:
        raise ValueError("Cannot read valid NIfTI {}: {}".format(path, error)) from error


def compare_geometry(reference, image, label="image"):
    """Raise on unequal dimensions/orientation/spacing or >0.001 mm alignment error."""
    validate_image(reference, "reference")
    validate_image(image, label)
    if reference.shape != image.shape:
        raise ValueError("{}: shape {} differs from reference {}; supply aligned inputs".format(label, image.shape, reference.shape))
    first, second = _affine_mm(reference), _affine_mm(image)
    if not np.allclose(nib.affines.voxel_sizes(first), nib.affines.voxel_sizes(second), rtol=0, atol=GEOMETRY_TOLERANCE_MM):
        raise ValueError("{}: voxel spacing differs from reference; no resampling is performed".format(label))
    distance = _distance(first, second, reference.shape)
    if distance > GEOMETRY_TOLERANCE_MM:
        raise ValueError("{}: affine alignment differs by {:.6g} mm (tolerance 0.001 mm); supply aligned inputs".format(label, distance))


def validate_label_values(image, label="segmentation", *, nonnegative=True,
                          allow_empty=True, allowed_labels=None):
    """Validate exact labels without converting integer arrays to floating point.

    Integer-valued floating storage is accepted, but fractional values are never
    rounded. Python integers preserve uint64 precision during allowed-set checks.
    """
    validate_image(image, label)
    data = np.asanyarray(image.dataobj)
    if data.dtype.kind == "f" and np.any(data != np.trunc(data)):
        raise ValueError("{} must contain exact integer labels, not probabilities".format(label))
    if nonnegative and np.any(data < 0):
        raise ValueError("{} must contain nonnegative labels".format(label))
    if not allow_empty and not np.any(data != 0):
        raise ValueError("{} is empty (no nonzero labels)".format(label))
    if allowed_labels is not None:
        allowed = set(allowed_labels)
        if any(int(value) not in allowed for value in np.unique(data)):
            raise ValueError("{} contains labels outside the model-specific allowed set".format(label))
    return image


def validate_spine_segmentation(image):
    # Preserve Phase 2's nonzero-mask contract; execution validation additionally
    # rejects negative spine labels through validate_labels().
    return validate_label_values(image, "spine segmentation", nonnegative=False, allow_empty=False)


def _new_image(data, source, affine):
    """Retain spatial units and compatible transform codes, without stale scaling."""
    header = source.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_slope_inter(None, None)
    result = source.__class__(data, affine, header)
    _, qcode = source.get_qform(coded=True)
    _, scode = source.get_sform(coded=True)
    # qform cannot encode shear. Keep it only if its approximation is compatible.
    result.set_qform(affine, int(qcode))
    if qcode and _distance(_affine_mm(result, result.get_qform()), _affine_mm(result, affine), data.shape) > GEOMETRY_TOLERANCE_MM:
        result.set_qform(None, 0)
    result.set_sform(affine, int(scode) or 2)
    return result


def binary_spine_mask(segmentation):
    validate_spine_segmentation(segmentation)
    return _new_image((np.asanyarray(segmentation.dataobj) != 0).astype(np.uint8), segmentation, segmentation.affine)


def image_on_reference_grid(data, reference):
    """Copy already-aligned voxel data onto a reference, retaining coded forms.

    No interpolation is performed. Callers must first validate grid alignment.
    """
    if data.shape != reference.shape:
        raise ValueError("Data dimensions do not match reference grid")
    result = _new_image(data, reference, reference.affine)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    result.set_qform(qform, int(qcode))
    result.set_sform(sform, int(scode))
    return result


@dataclass(frozen=True)
class Crop:
    start: Tuple[int, int, int]
    stop: Tuple[int, int, int]
    original_shape: Tuple[int, int, int]

    def __post_init__(self):
        if any(len(values) != 3 for values in (self.start, self.stop, self.original_shape)) or any(
            not isinstance(value, (int, np.integer)) for values in (self.start, self.stop, self.original_shape) for value in values
        ) or any(not 0 <= lo < hi <= size for lo, hi, size in zip(self.start, self.stop, self.original_shape)):
            raise ValueError("Invalid crop: require integer 0 <= start < stop <= original_shape")

    @property
    def slices(self):
        return tuple(slice(lo, hi) for lo, hi in zip(self.start, self.stop))

    @property
    def shape(self):
        return tuple(hi - lo for lo, hi in zip(self.start, self.stop))


def padded_bounding_box(binary_mask, padding=5):
    validate_spine_segmentation(binary_mask)
    data = np.asanyarray(binary_mask.dataobj)
    if not np.isin(data, (0, 1)).all():
        raise ValueError("Bounding box requires a binary mask; binarize spine labels first")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("Padding must be a nonnegative integer")
    occupied = np.nonzero(data)
    start = tuple(max(int(axis.min()) - padding, 0) for axis in occupied)
    stop = tuple(min(int(axis.max()) + padding + 1, size) for axis, size in zip(occupied, data.shape))
    return Crop(start, stop, data.shape)


def crop_image(image, crop):
    validate_image(image)
    if image.shape != crop.original_shape:
        raise ValueError("Image dimensions do not match stored crop original_shape")
    translation = np.eye(4)
    translation[:3, 3] = crop.start
    data = np.asanyarray(image.dataobj)[crop.slices].copy()
    return _new_image(data, image, image.affine @ translation)


@dataclass(frozen=True)
class PreparedChannels:
    channel_names: Tuple[str, ...]
    images: tuple
    crop: Crop
    reference: object

    @property
    def filenames(self):
        return tuple("case_0001_{:04d}.nii.gz".format(i) for i in range(len(self.images)))


def prepare_lesion_channels(dataset_id, reconstructions, segmentation, reference=None):
    """Prepare ordered cropped images in memory; source images are never changed."""
    model = get_lesion_model(dataset_id)
    missing = [name for name in model.required_modalities if name not in reconstructions]
    if missing:
        raise ValueError("Missing required modalities: " + ", ".join(missing))
    if model.channel_names != model.required_modalities + ("mask_spine",):
        raise ValueError("Registry channel order/count does not match reconstruction and mask inputs")
    if reference is None:
        reference = reconstructions[model.required_modalities[0]]
    binary = binary_spine_mask(segmentation)
    compare_geometry(reference, binary, "spine segmentation")
    for name in model.required_modalities:
        compare_geometry(reference, reconstructions[name], name)
    crop = padded_bounding_box(binary)
    cropped_reference = crop_image(reference, crop)
    # Geometry validation establishes matching voxel indices. Use the single
    # reference metadata for every channel, including sub-tolerance differences
    # and physically equivalent declared spatial units; do not resample data.
    images = tuple(image_on_reference_grid(np.asanyarray(crop_image(source, crop).dataobj), cropped_reference)
                   for source in tuple(reconstructions[name] for name in model.required_modalities) + (binary,))
    return PreparedChannels(model.channel_names, images, crop, reference)


def save_prepared_channels(prepared, output_dir):
    """Explicit writer: require a fresh directory and never overwrite existing files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    paths = tuple(directory / name for name in prepared.filenames)
    for image, path in zip(prepared.images, paths):
        nib.save(image, str(path))
    return paths


def reconstruct_lesions(prediction, crop, reference, output_path=None):
    """Reconstruct strict 0/1 lesions; caller verifies the model's output convention.

    Explicit saving uses a complete validated temporary NIfTI and an exclusive
    same-filesystem rename. No hard-link support is required.
    """
    expected = crop_image(reference, crop)
    compare_geometry(expected, prediction, "cropped lesion prediction")
    validate_label_values(prediction, "Lesion prediction (binary 0/1)", allowed_labels=(0, 1))
    data = np.asanyarray(prediction.dataobj)
    full = np.zeros(reference.shape, dtype=np.uint8)
    full[crop.slices] = data.astype(np.uint8)
    result = image_on_reference_grid(full, reference)
    if output_path is not None:
        save_nifti_no_clobber(result, output_path)
    return result


def _rename_no_clobber(source, destination):
    """Publish on one filesystem, without importing model-installation code.

    Keep this small OS primitive independent to preserve the validated Phase 5C
    installer unchanged. Native flag tests guard its no-overwrite semantics.
    """
    if os.name == "nt":
        os.rename(str(source), str(destination))
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        operation = library.renameat2
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        operation = library.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(source_bytes, destination_bytes, 4)
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-clobber NIfTI rename is unavailable")
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, "Atomic no-clobber NIfTI rename failed: {}".format(os.strerror(code)), str(destination))


def save_nifti_no_clobber(image, output_path):
    """Validate/fsync a temporary NIfTI, then exclusively rename it in-place.

    The temporary and final paths share a directory. Native exclusive rename
    must be supported; never fall back to overwriting or cross-device copying.
    This protects publication visibility, not durability after power loss.
    """
    validate_image(image)
    path = Path(output_path)
    if not str(path).endswith(".nii.gz"):
        raise ValueError("NIfTI output must end with .nii.gz")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".v3-nifti-", suffix=".nii.gz", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nib.save(image, str(temporary))
        written = load_nifti(temporary)
        compare_geometry(image, written, "saved NIfTI")
        if not np.array_equal(np.asanyarray(written.dataobj), np.asanyarray(image.dataobj)):
            raise ValueError("Saved NIfTI voxel verification failed")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        _rename_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_plan_inputs(plan):
    """Load required CLI inputs and validate them without writing intermediates."""
    reconstructions = {name: load_nifti(path) for name, path in plan.modality_paths}
    segmentation = load_nifti(plan.spine_segmentation) if plan.spine_segmentation is not None else None
    if segmentation is not None:
        validate_spine_segmentation(segmentation)
    reference = (reconstructions["ConvCT"] if segmentation is None else
                 reconstructions[plan.lesion_model.required_modalities[0]] if plan.lesion_model else segmentation)
    for name, image in reconstructions.items():
        compare_geometry(reference, image, name)
    if segmentation is not None:
        compare_geometry(reference, segmentation, "spine segmentation")
    return reconstructions, segmentation, reference
