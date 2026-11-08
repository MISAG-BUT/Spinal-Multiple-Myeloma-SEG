"""V3 five-fold lesion inference and private end-to-end run orchestration.

Registry metadata, V3 geometry/preparation, the Phase 4A spine adapter, and the
verified installer remain authoritative. No Torch/nnU-Net imports occur here.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from benchmark_model_registry import get_lesion_model, SPINE_MODEL
from benchmark_model_installation import (
    ModelInstaller, InstallationError, validate_model, _read_json, _publish_directory,
)
from benchmark_nifti import (
    binary_spine_mask, compare_geometry, load_nifti, prepare_lesion_channels,
    reconstruct_lesions, save_nifti_no_clobber, save_prepared_channels,
    validate_label_values,
)
from benchmark_spine_inference import (
    create_predictor, infer_spine, validate_labels, _label_image, _geometry,
)


class LesionInferenceError(RuntimeError):
    """Lesion prediction, reconstruction, or completed-run publication failed."""


def lesion_model_metadata(resolved, spec):
    """Validate model-specific structure/channels and the binary output convention."""
    if resolved.dataset_id != spec.dataset_id:
        raise InstallationError("Resolved lesion model ID disagrees with selected registry model")
    directory = validate_model(resolved.model_directory, spec)
    dataset = _read_json(directory / "dataset.json")
    labels = dataset.get("labels")
    if (not isinstance(labels, dict) or not labels
            or any(type(value) is not int for value in labels.values())
            or set(labels.values()) != {0, 1}):
        raise InstallationError("Lesion model {} dataset.json must declare binary output labels 0 and 1".format(spec.dataset_id))
    return dataset


def create_lesion_predictor():
    return create_predictor(inference_name="Benchmark lesion inference")


def validate_lesion_prediction(prediction, reference, dataset):
    """Validate the restored crop grid and exact model labels, including 0.0/1.0."""
    compare_geometry(reference, prediction, "cropped lesion prediction")
    return validate_label_values(prediction, "Lesion prediction (binary 0/1)",
                                 allowed_labels=set(dataset["labels"].values()))


def infer_lesions(prepared, resolved, work_dir, *, predictor_factory=None):
    """Execute one registry-ordered case with all five benchmark folds.

    Empty binary predictions are valid. Invalid labels are never binarized.
    """
    try:
        spec = get_lesion_model(resolved.dataset_id)
        dataset = lesion_model_metadata(resolved, spec)
        if prepared.channel_names != spec.channel_names or len(prepared.images) != len(spec.channel_names):
            raise ValueError("Prepared lesion channel count/order disagrees with registry")
        cropped_reference = prepared.images[0]
        for channel in prepared.images:
            compare_geometry(cropped_reference, channel, "cropped lesion channel")
        if not np.isin(np.asanyarray(prepared.images[-1].dataobj), (0, 1)).all():
            raise ValueError("Final lesion channel must be a binary spine mask")
        paths = save_prepared_channels(prepared, Path(work_dir) / "inputs")
        output = Path(work_dir) / "predictions"
        output.mkdir(exist_ok=False)
        predictor = (predictor_factory or create_lesion_predictor)()
        try:
            predictor.initialize_from_trained_model_folder(
                str(resolved.model_directory), use_folds=spec.folds,
                checkpoint_name=spec.checkpoint_filename,
            )
        except AssertionError as error:
            raise LesionInferenceError("Lesion predictor initialization rejected the model: {}".format(error)) from error
        trainer, _, configuration = spec.trainer_directory.split("__")
        if getattr(predictor, "trainer_name", trainer) != trainer:
            raise ValueError("Loaded lesion checkpoint trainer disagrees with registry")
        manager = getattr(predictor, "configuration_manager", None)
        if manager is not None and manager.configuration != predictor.plans_manager.get_configuration(configuration).configuration:
            raise ValueError("Loaded lesion checkpoint configuration disagrees with registry")
        try:
            predictor.predict_from_files_sequential(
                [[str(path) for path in paths]], [str(output / "case_0001")],
                save_probabilities=False, overwrite=False, folder_with_segs_from_prev_stage=None,
            )
        except AssertionError as error:
            raise LesionInferenceError("Lesion predictor rejected the inference request: {}".format(error)) from error
        prediction = load_nifti(output / "case_0001.nii.gz")
        return validate_lesion_prediction(prediction, cropped_reference, dataset)
    except LesionInferenceError:
        raise
    except (InstallationError, OSError, ValueError, RuntimeError, ImportError) as error:
        raise LesionInferenceError("Benchmark lesion inference failed: {}".format(error)) from error


def _model_provenance(resolved, spec):
    trainer, plans, configuration = spec.trainer_directory.split("__")
    return {"dataset_id": spec.dataset_id, "dataset_name": spec.dataset_name,
            "trainer": trainer, "plans": plans, "configuration": configuration,
            "trainer_directory": spec.trainer_directory, "folds": list(spec.folds),
            "checkpoint": spec.checkpoint_filename, "verification": resolved.verification,
            "artifact": resolved.provenance.get("artifact")}


def run_lesion_pipeline(plan, validated_inputs, *, installer=None,
                        spine_predictor_factory=None, lesion_predictor_factory=None):
    """Preflight every model, then stage and publish one complete spine+lesion run."""
    if plan.lesion_model is None:
        raise ValueError("Select a lesion model for end-to-end lesion inference")
    spec = get_lesion_model(plan.lesion_model.dataset_id)
    reconstructions, segmentation, reference = validated_inputs
    if segmentation is not None:
        validate_labels(segmentation)
        compare_geometry(reference, segmentation, "supplied spine segmentation")
    for name in spec.required_modalities:
        if name not in reconstructions:
            raise ValueError("Missing required lesion modality: {}".format(name))
        compare_geometry(reference, reconstructions[name], name)
    # No predictor or patient output directory is created before both resolve.
    try:
        installer = installer or ModelInstaller(plan.cache_dir)
        spine_model = installer.resolve(802, spine_model_dir=plan.spine_model_dir,
                                        allow_download=plan.install_models) if segmentation is None else None
        lesion_model = installer.resolve(spec.dataset_id, allow_download=plan.install_models)
        lesion_model_metadata(lesion_model, spec)
    except OSError as error:
        raise InstallationError("Model preflight failed: {}".format(error)) from error
    output = Path(plan.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ("run-" + uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix=".v3-lesion-", dir=str(output)) as temporary:
        stage = Path(temporary)
        work = stage / "work"
        work.mkdir()
        if segmentation is None:
            labelled, split_effective = infer_spine(reference, spine_model, work / "spine",
                                                   split=plan.split, predictor_factory=spine_predictor_factory)
        else:
            labelled = _label_image(np.asanyarray(segmentation.dataobj), reference)
            split_effective = None
        validate_labels(labelled)
        compare_geometry(reference, labelled, "full spine segmentation")
        binary = _label_image(np.asanyarray(binary_spine_mask(labelled).dataobj), reference)
        prepared = prepare_lesion_channels(spec.dataset_id, reconstructions, labelled, reference)
        cropped_prediction = infer_lesions(prepared, lesion_model, work / "lesion",
                                           predictor_factory=lesion_predictor_factory)
        lesions = reconstruct_lesions(cropped_prediction, prepared.crop, reference)
        compare_geometry(reference, lesions, "full lesion segmentation")
        spine_dir, lesion_dir = stage / "spine", stage / "lesion"
        spine_dir.mkdir()
        lesion_dir.mkdir()
        save_nifti_no_clobber(labelled, spine_dir / "spine_labelled.nii.gz")
        save_nifti_no_clobber(binary, spine_dir / "spine_binary.nii.gz")
        save_nifti_no_clobber(lesions, lesion_dir / "lesion_labelled.nii.gz")
        if segmentation is not None:
            shutil.copyfile(str(plan.spine_segmentation), str(spine_dir / "spine_supplied.nii.gz"))
        if work.resolve().parent != stage.resolve():
            raise LesionInferenceError("Temporary work directory escaped its run directory")
        shutil.rmtree(work)
        manifest = {
            "schema_version": 2, "phase": "4B", "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "spine_source": "automatic" if spine_model else "supplied",
            "source_convct": "ConvCT.nii.gz" if "ConvCT" in reconstructions else None,
            "source_segmentation": plan.spine_segmentation.name if segmentation is not None else None,
            "model": _model_provenance(spine_model, SPINE_MODEL) if spine_model else None,
            "lesion_model": _model_provenance(lesion_model, spec),
            "split_requested": plan.split, "split_effective": split_effective,
            "split_fallback": "single_z_slice" if spine_model and plan.split and reference.shape[2] == 1 else None,
            "input_modalities": list(reconstructions), "lesion_modalities": list(spec.required_modalities),
            "ordered_lesion_channels": list(spec.channel_names),
            "reference_geometry": _geometry(reference), "cropped_geometry": _geometry(prepared.images[0]),
            "output_geometry": _geometry(lesions), "spine_geometry": _geometry(labelled),
            "crop": {"start": list(prepared.crop.start), "stop": list(prepared.crop.stop),
                     "upper_bounds": "exclusive", "padding_voxels": 5,
                     "original_shape": list(prepared.crop.original_shape)},
            "labelled_output": "spine/spine_labelled.nii.gz", "binary_output": "spine/spine_binary.nii.gz",
            "lesion_output": "lesion/lesion_labelled.nii.gz", "lesion_inference": "completed",
        }
        with (stage / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        try:
            _publish_directory(stage, destination)
        except InstallationError as error:
            raise LesionInferenceError("Cannot publish lesion run: {}. Existing results were preserved; use a filesystem supporting atomic no-clobber directory rename.".format(error)) from error
    return destination
