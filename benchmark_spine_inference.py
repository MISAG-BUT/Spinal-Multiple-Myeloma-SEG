"""V3 model-802 spine execution. No Torch/nnU-Net imports until construction.

V2 splits every requested volume at Z//2, without overlap or label reconciliation.
Only the impossible Z=1 split falls back to a full-volume input in V3.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import nibabel as nib
import numpy as np

from benchmark_model_registry import SPINE_MODEL
from benchmark_model_installation import ModelInstaller, InstallationError, _publish_directory
from benchmark_nifti import (
    Crop, _new_image, binary_spine_mask, compare_geometry, crop_image,
    load_nifti, save_nifti_no_clobber, validate_image, image_on_reference_grid,
    validate_label_values,
)


class SpineInferenceError(RuntimeError):
    """Model execution or output validation failed; no completed run published."""


def create_predictor(inference_name="Automatic spine inference"):
    """Use V2 sliding-window settings; sequential file API avoids worker spawning."""
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except (ImportError, OSError, RuntimeError) as error:
        raise SpineInferenceError("Cannot import nnU-Net inference dependencies. Install nnunetv2==2.6.4 and a compatible PyTorch/CUDA environment: {}".format(error)) from error
    if not torch.cuda.is_available():
        raise SpineInferenceError("{} requires a CUDA-enabled PyTorch installation and an available GPU".format(inference_name))
    return nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device("cuda", 0),
        verbose=False, verbose_preprocessing=False, allow_tqdm=True,
    )


def validate_labels(image, *, allow_empty=False, allowed_labels=None):
    """Keep exact discrete nonnegative labels; permit background-only split parts."""
    validate_label_values(image, "Spine prediction", allow_empty=allow_empty, allowed_labels=allowed_labels)
    data = np.asanyarray(image.dataobj)
    maximum = int(data.max())
    if maximum > np.iinfo(np.uint64).max:
        raise ValueError("Spine labels exceed the supported uint64 range")
    return image


def _label_image(data, reference):
    maximum = int(data.max())
    dtype = next(dtype for dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
                 if maximum <= np.iinfo(dtype).max)
    return image_on_reference_grid(data.astype(dtype), reference)


def split_crops(reference, split=True):
    """Return ordered Z slices. No threshold, overlap, or voxel changes."""
    validate_image(reference, "ConvCT")
    x, y, z = reference.shape
    ranges = ((0, z // 2), (z // 2, z)) if split and z >= 2 else ((0, z),)
    return tuple(Crop((0, 0, start), (x, y, stop), reference.shape) for start, stop in ranges)


def infer_spine(reference, resolved_model, work_dir, *, split=True, predictor_factory=None):
    """Run separate per-part requests and concatenate exact voxel grids in Z order.

    Factory injection accepts a fake implementing the two nnUNetPredictor methods.
    Prediction files must restore each input grid; no resampling is used here.
    """
    if resolved_model.dataset_id != SPINE_MODEL.dataset_id:
        raise SpineInferenceError("Automatic spine inference requires resolved model 802")
    crops = split_crops(reference, split)
    try:
        with (resolved_model.model_directory / "dataset.json").open(encoding="utf-8") as stream:
            dataset = json.load(stream)
        if not isinstance(dataset, dict) or dataset.get("channel_names") != {"0": SPINE_MODEL.channel_names[0]}:
            raise ValueError("Model 802 must have exactly one ConvCT channel")
        label_map = dataset.get("labels")
        if not isinstance(label_map, dict) or not label_map:
            raise ValueError("Model 802 dataset.json labels must be a nonempty object")
        values = label_map.values()
        if any(type(value) is not int or value < 0 for value in values) or 0 not in values or not any(value > 0 for value in values):
            raise ValueError("Model 802 dataset.json must declare discrete nonnegative labels")
        labels = set(values)
        predictor = (predictor_factory or create_predictor)()
        predictor.initialize_from_trained_model_folder(
            str(resolved_model.model_directory), use_folds=SPINE_MODEL.folds,
            checkpoint_name=SPINE_MODEL.checkpoint_filename,
        )
        expected_trainer, _, expected_configuration = SPINE_MODEL.trainer_directory.split("__")
        if getattr(predictor, "trainer_name", expected_trainer) != expected_trainer:
            raise ValueError("Loaded checkpoint trainer does not match model 802")
        manager = getattr(predictor, "configuration_manager", None)
        if manager is not None:
            expected = predictor.plans_manager.get_configuration(expected_configuration)
            if manager.configuration != expected.configuration:
                raise ValueError("Loaded checkpoint does not use the expected 3d_fullres configuration")
        predictions = []
        for index, crop in enumerate(crops):
            part = crop_image(reference, crop)
            directory = Path(work_dir) / ("part_{:02d}".format(index))
            inputs, outputs = directory / "inputs", directory / "predictions"
            inputs.mkdir(parents=True, exist_ok=False)
            outputs.mkdir()
            nib.save(part, str(inputs / "case_0001_0000.nii.gz"))
            predictor.predict_from_files_sequential(
                # v2.6.4 directory discovery starts a Pool even in sequential
                # mode. Explicit lists keep this one-channel request sequential.
                [[str(inputs / "case_0001_0000.nii.gz")]],
                [str(outputs / "case_0001")], save_probabilities=False,
                overwrite=False, folder_with_segs_from_prev_stage=None,
            )
            prediction = load_nifti(outputs / "case_0001.nii.gz")
            compare_geometry(part, prediction, "model-802 part {}".format(index))
            validate_labels(prediction, allow_empty=True, allowed_labels=labels)
            predictions.append(np.asanyarray(prediction.dataobj))
        # Use one unsigned integer type to avoid uint64/int64 -> float promotion.
        maximum = max(int(data.max()) for data in predictions)
        dtype = next(dtype for dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
                     if maximum <= np.iinfo(dtype).max)
        merged = np.concatenate([data.astype(dtype) for data in predictions], axis=2)
        if merged.shape != reference.shape:
            raise ValueError("Merged spine shape differs from original ConvCT")
        result = _label_image(merged, reference)
        compare_geometry(reference, result, "full spine prediction")
        validate_labels(result, allowed_labels=labels)
        return result, len(crops) == 2
    except SpineInferenceError:
        raise
    except Exception as error:
        raise SpineInferenceError("Model-802 spine inference failed: {}".format(error)) from error


def _geometry(image):
    return {"shape": list(image.shape), "affine": image.affine.tolist(),
            "spacing": [float(value) for value in image.header.get_zooms()],
            "spatial_units": image.header.get_xyzt_units()[0],
            "orientation": list(nib.aff2axcodes(image.affine)),
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"])}


def validate_spine_inputs(plan, validated_inputs):
    """Extend Phase 2 validation for spine execution, including optional ConvCT."""
    reconstructions, segmentation, reference = validated_inputs
    reconstructions = dict(reconstructions)
    if segmentation is not None:
        validate_labels(segmentation)
        exact_names = {path.name for path in plan.input_dir.iterdir()}
        if "ConvCT.nii.gz" in exact_names:
            if "ConvCT" not in reconstructions:
                reconstructions["ConvCT"] = load_nifti(plan.input_dir / "ConvCT.nii.gz")
            reference = reconstructions["ConvCT"]
        compare_geometry(reference, segmentation, "supplied spine segmentation")
    else:
        validate_image(reference, "ConvCT")
    return reconstructions, segmentation, reference


def run_spine_only(plan, validated_inputs, *, installer=None, predictor_factory=None):
    """Publish a complete unique run, including labelled and binary spine masks.

    Caller must validate inputs first. Supplied masks never resolve models or
    construct predictors. No lesion model is executed or installed here.
    """
    if plan.lesion_model is not None:
        raise SpineInferenceError("Lesion inference is not implemented in Phase 4A")
    reconstructions, segmentation, reference = validate_spine_inputs(plan, validated_inputs)
    resolved = None
    if segmentation is None:
        resolved = (installer or ModelInstaller(plan.cache_dir)).resolve(
            802, spine_model_dir=plan.spine_model_dir, allow_download=plan.install_models,
        )
    output = Path(plan.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ("run-" + uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix=".v3-spine-", dir=str(output)) as temporary:
        stage = Path(temporary)
        work = stage / "work"
        work.mkdir()
        if segmentation is None:
            labelled, split_effective = infer_spine(
                reference, resolved, work, split=plan.split, predictor_factory=predictor_factory,
            )
        else:
            labelled = _label_image(np.asanyarray(segmentation.dataobj), reference)
            split_effective = None
        validate_labels(labelled)
        compare_geometry(reference, labelled, "final spine segmentation")
        spine_dir = stage / "spine"
        spine_dir.mkdir()
        save_nifti_no_clobber(labelled, spine_dir / "spine_labelled.nii.gz")
        binary = binary_spine_mask(labelled)
        binary = _label_image(np.asanyarray(binary.dataobj), reference)
        save_nifti_no_clobber(binary, spine_dir / "spine_binary.nii.gz")
        if segmentation is not None:
            shutil.copyfile(str(plan.spine_segmentation), str(spine_dir / "spine_supplied.nii.gz"))
        # Work directory is owned by this run, under its unique staging directory.
        if work.resolve().parent != stage.resolve():
            raise SpineInferenceError("Temporary work directory escaped its run directory")
        shutil.rmtree(work)
        manifest = {
            "schema_version": 1, "phase": "4A", "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "spine_source": "automatic" if resolved else "supplied",
            "source_convct": "ConvCT.nii.gz" if "ConvCT" in reconstructions else None,
            "source_segmentation": plan.spine_segmentation.name if segmentation is not None else None,
            "split_requested": plan.split, "split_effective": split_effective,
            "split_fallback": "single_z_slice" if resolved and plan.split and reference.shape[2] == 1 else None,
            "model": {"dataset_id": 802, "verification": resolved.verification,
                      "artifact": resolved.provenance.get("artifact"),
                      "folds": list(SPINE_MODEL.folds), "checkpoint": SPINE_MODEL.checkpoint_filename,
                      "trainer_directory": SPINE_MODEL.trainer_directory} if resolved else None,
            "reference_geometry": _geometry(reference), "output_geometry": _geometry(labelled),
            "labelled_output": "spine/spine_labelled.nii.gz",
            "binary_output": "spine/spine_binary.nii.gz", "lesion_inference": "not_requested",
        }
        with (stage / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        try:
            _publish_directory(stage, destination)
        except InstallationError as error:
            # The shared native rename helper uses InstallationError, but this
            # is patient-output publication, not model installation (CLI code 5).
            raise SpineInferenceError("Cannot publish spine run: {}. Use an output filesystem supporting atomic no-clobber directory rename; existing results were preserved.".format(error)) from error
    return destination
