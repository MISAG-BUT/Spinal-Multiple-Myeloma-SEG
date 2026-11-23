"""Prepared nnU-Net batch preflight and sequential lesion execution."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from time import perf_counter
from uuid import uuid4

from .lesion_inference import (
    LesionInferenceError, LesionSession, PredictorExecutionError,
    initialize_lesion_predictor, lesion_model_metadata, run_lesion_pipeline,
)
from .model_installation import (
    InstallationError, ModelInstaller, _rename_no_clobber, cache_location,
)
from .nifti import publish_versioned, validate_plan_inputs


def discover_cases(template, validate_case_id, prepared_filenames):
    """Group every NIfTI candidate by its last underscore-delimited channel index."""
    groups = {}
    identities = {}
    errors = {}
    for path in sorted(template.nnunet_input_dir.iterdir()):
        if not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        match = re.fullmatch(r"(.*)_([0-9]{4})\.nii\.gz", path.name)
        if match is None:
            errors[path.name] = [f"Malformed channel filename {path.name!r}, expected <case>_NNNN.nii.gz"]
            continue
        groups.setdefault(match[1], {})[path.name] = path

    plans = []
    folded = {}
    for case_id, channels in sorted(groups.items()):
        try:
            validate_case_id(case_id)
            expected = prepared_filenames(template.lesion_model, case_id)
            missing = sorted(set(expected) - channels.keys())
            extra = sorted(channels.keys() - set(expected))
            if missing or extra:
                raise ValueError(f"Invalid prepared channels: missing {missing}, unexpected {extra}")
            paths = tuple(channels[name] for name in expected)
            folded.setdefault(case_id.casefold(), []).append(case_id)
            for path in paths:
                if not path.is_file():
                    raise ValueError(f"Channel is not a readable file: {path.name}")
                stat = path.stat()
                identities.setdefault((stat.st_dev, stat.st_ino), []).append((case_id, path.name))
            plans.append(replace(template, case_id=case_id,
                                 modality_paths=tuple(zip(template.lesion_model.channel_names, paths))))
        except (ValueError, OSError) as error:
            errors.setdefault(case_id, []).append(str(error))
    for cases in folded.values():
        if len(cases) > 1:
            for case_id in cases:
                errors.setdefault(case_id, []).append(f"Case identifiers collide by capitalization: {cases}")
    for entries in identities.values():
        if len(entries) > 1:
            names = ", ".join(name for _, name in entries)
            for case_id in {case for case, _ in entries}:
                errors.setdefault(case_id, []).append(f"Aliased input paths: {names}")
    return tuple(sorted(set(groups) | errors.keys())), plans, errors


def _publish_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".v3-batch-",
                                         suffix=".json", dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return publish_versioned(temporary, output, "batch_run", ".json", _rename_no_clobber)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_batch(template, validate_case_id, prepared_filenames):
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    try:
        cache, source = cache_location(template.cache_dir)
        if cache.resolve().is_relative_to(template.nnunet_input_dir.resolve()):
            raise ValueError("Batch model cache must be outside the prepared input directory")
        discovered, plans, errors = discover_cases(template, validate_case_id, prepared_filenames)
    except (ValueError, OSError) as error:
        print(f"V3 batch preflight failed: {error}", file=sys.stderr)
        return 2

    for plan in plans:
        if plan.case_id not in errors:
            try:
                validate_plan_inputs(plan)
            except (ValueError, OSError) as error:
                errors[plan.case_id] = [str(error)]
    print(f"V3 batch preflight: {len(discovered)} discovered, "
          f"{len(discovered) - len(errors)} valid, {len(errors)} invalid")
    for case_id, diagnostics in sorted(errors.items()):
        for message in diagnostics:
            print(f"V3 batch case {case_id!r}: {message}", file=sys.stderr)
    if not discovered:
        print("V3 batch preflight failed: no prepared NIfTI cases found", file=sys.stderr)
    if errors or not discovered:
        return 2
    if template.validate_only:
        return 3

    spec = template.lesion_model
    records = {plan.case_id: {"status": "unprocessed", "duration_seconds": None} for plan in plans}
    code = 0
    failure = None
    initializations = 0
    resolved = None
    try:
        resolved = ModelInstaller(template.cache_dir).resolve(spec.dataset_id, allow_download=template.install_models)
        dataset = lesion_model_metadata(resolved, spec)
        predictor = initialize_lesion_predictor(resolved, spec)
        initializations = 1
        session = LesionSession(resolved, dataset, predictor)
        for plan in plans:
            print(f"V3 batch case: {plan.case_id}", flush=True)
            case_started = perf_counter()
            record = records[plan.case_id]
            try:
                destination = run_lesion_pipeline(plan, validate_plan_inputs(plan), lesion_session=session)
                record.update(status="succeeded", output_dir=str(destination))
            except Exception as error:
                record.update(status="failed", error=str(error))
                code = 5
                print(f"V3 batch case {plan.case_id!r} failed: {error}", file=sys.stderr)
                if isinstance(error, PredictorExecutionError) or not isinstance(
                    error, (LesionInferenceError, ValueError, OSError)
                ):
                    failure = f"Case {plan.case_id}: {error}"
                    break
            finally:
                record["duration_seconds"] = perf_counter() - case_started
    except Exception as error:
        code = 4 if isinstance(error, InstallationError) else 5
        failure = str(error)
        print(f"V3 batch aborted: {error}", file=sys.stderr)

    counts = {key: sum(record["status"] == key for record in records.values())
              for key in ("succeeded", "failed", "unprocessed")}
    report = {
        "schema_version": 1,
        "batch_id": uuid4().hex,
        "model": {"dataset_id": spec.dataset_id, "dataset_name": spec.dataset_name,
                  "ordered_channels": list(spec.channel_names), "folds": list(spec.folds),
                  "checkpoint": spec.checkpoint_filename,
                  "cache_status": resolved.cache_status if resolved is not None else None},
        "settings": {"nnunet_input_dir": str(template.nnunet_input_dir),
                     "output_dir": str(template.output_dir), "cache_dir": str(cache),
                     "cache_source": source, "offline": template.offline,
                     "allow_download": template.install_models, "already_cropped": template.already_cropped,
                     "keep_intermediates": template.keep_intermediates},
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": perf_counter() - started,
        "discovered_case_ids": list(discovered),
        "counts": counts,
        "cases": records,
        "predictor_initializations": initializations,
        "status": "aborted" if failure is not None else "partial_failure" if code else "completed",
        "error": failure,
    }
    try:
        destination = _publish_report(template.output_dir, report)
    except (InstallationError, OSError) as error:
        print(f"Cannot publish V3 batch report: {error}", file=sys.stderr)
        return 5
    print(f"V3 batch report: {destination}")
    return code
