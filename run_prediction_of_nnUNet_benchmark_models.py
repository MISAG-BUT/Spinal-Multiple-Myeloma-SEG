"""V3 Phase 4B: standalone spine and benchmark lesion execution.

Later stages must binarize nonzero spine labels before a shared five-voxel crop,
translate cropped affines, retain published normalization/plans, and reconstruct
on the original grid. Model installation must use isolated record-specific roots.
Imaging utilities are loaded only on execution, never during argument parsing.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Optional, Tuple

from benchmark_model_registry import LESION_MODELS, ModelSpec, get_lesion_model


MODALITY_OPTIONS = {
    "ConvCT": "--path_to_convCT_nifti",
    "VMI_40": "--path_to_VMI40_nifti",
    "VMI_80": "--path_to_VMI80_nifti",
    "VMI_120": "--path_to_VMI120_nifti",
    "CaSupp_25": "--path_to_CaSupp25_nifti",
    "CaSupp_50": "--path_to_CaSupp50_nifti",
    "CaSupp_75": "--path_to_CaSupp75_nifti",
    "CaSupp_100": "--path_to_CaSupp100_nifti",
}
MODALITY_FILENAMES = {name: name + ".nii.gz" for name in MODALITY_OPTIONS}
_RESERVED_CASE_IDS = {"CON", "PRN", "AUX", "NUL"} | {
    prefix + str(number) for prefix in ("COM", "LPT") for number in range(1, 10)
}


def validate_case_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value) or value.upper() in _RESERVED_CASE_IDS:
        raise ValueError("Unsafe case ID; use --case_id with 1–80 ASCII letters, digits, underscores or hyphens, starting with a letter or digit")
    return value


@dataclass(frozen=True)
class ExecutionPlan:
    input_dir: Optional[Path]
    output_dir: Path
    lesion_model: Optional[ModelSpec]
    spine_segmentation: Optional[Path]
    spine_model_dir: Optional[Path]
    cache_dir: Optional[Path]
    split: Optional[bool]
    modality_paths: Tuple[Tuple[str, Path], ...]
    install_models: bool = False
    validate_only: bool = False
    case_id: str = "case"


def build_parser():
    parser = argparse.ArgumentParser(
        description="V3 benchmark prediction (Phase 4B: spine and benchmark lesions)",
        allow_abbrev=False,
    )
    inputs = parser.add_argument_group("Input and case")
    inputs.add_argument("--input_dir", type=Path, help="Folder containing standard modality filenames")
    for modality, option in MODALITY_OPTIONS.items():
        inputs.add_argument(option, dest=modality, type=Path, metavar="NIFTI", action="append",
                            help=f"{modality} file (.nii or .nii.gz); use explicit paths without --input_dir")
    inputs.add_argument("--case_id", help="Case ID; defaults to the input folder name and is required without --input_dir")
    inputs.add_argument("--spine_segmentation", type=Path, help="Existing spine labels; skips model 802")

    execution = parser.add_argument_group("Prediction")
    selection = execution.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lesion_model", type=int, choices=sorted(LESION_MODELS), metavar="700–718")
    selection.add_argument("--skip_lesion_segmentation", action="store_true", help="Publish spine masks only")
    execution.add_argument("--split", type=lambda x: str(x).lower() == "true",
                           default=argparse.SUPPRESS, help="Split automatic spine inference in two (default: True)")
    execution.add_argument("--validate_only", action="store_true",
                           help="Check inputs without model installation, prediction or output creation")
    execution.add_argument("--output_dir", required=True, type=Path, help="Parent folder for completed case runs")

    models = parser.add_argument_group("Model storage")
    models.add_argument("--spine_model_dir", type=Path, help="External model 802 trainer directory")
    models.add_argument("--cache_dir", type=Path, help="Override the default model cache")
    models.add_argument("--install_models", action="store_true", help="Authorize downloading missing models")
    return parser


def validate_arguments(args):
    """Validate paths only, without reading NIfTI content or creating directories."""
    supplied = args.spine_segmentation is not None
    if args.validate_only and args.install_models:
        raise ValueError("--validate_only cannot be combined with --install_models")
    if supplied and args.spine_model_dir is not None:
        raise ValueError("--spine_model_dir cannot be combined with --spine_segmentation")
    if supplied and hasattr(args, "split"):
        raise ValueError("Explicit --split is not applicable with --spine_segmentation")
    if args.input_dir is not None and not args.input_dir.is_dir():
        raise ValueError("Input directory does not exist: {}".format(args.input_dir))
    if supplied and not args.spine_segmentation.is_file():
        raise ValueError("Supplied spine segmentation file does not exist: {}".format(args.spine_segmentation))
    if args.spine_model_dir is not None and not args.spine_model_dir.is_dir():
        raise ValueError("Spine model directory does not exist: {}".format(args.spine_model_dir))
    for label, path in (("Output", args.output_dir), ("Cache", args.cache_dir)):
        if path is not None:
            ancestor = path
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            if not ancestor.is_dir():
                raise ValueError("{} path is not a directory or has a file as an ancestor: {}".format(label, path))

    model = get_lesion_model(args.lesion_model) if args.lesion_model is not None else None
    required = list(model.required_modalities) if model else []
    if not supplied and "ConvCT" not in required:
        required.insert(0, "ConvCT")
    explicit = {}
    for name, option in MODALITY_OPTIONS.items():
        paths = getattr(args, name)
        if paths:
            if len(paths) != 1:
                raise ValueError(f"{option} may only be specified once")
            explicit[name] = paths[0]
    if args.input_dir is not None and explicit:
        raise ValueError("--input_dir cannot be combined with explicit reconstruction paths")
    optional_convct = supplied and model is None
    if args.input_dir is None:
        if args.case_id is None:
            raise ValueError("--case_id is required without --input_dir")
        entries = explicit
        unused = entries.keys() - set(required) - ({"ConvCT"} if optional_convct else set())
        if unused:
            raise ValueError("Unused explicit reconstructions: " + ", ".join(sorted(unused)))
        for name, path in entries.items():
            if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
                raise ValueError(f"Expected an existing .nii or .nii.gz file for {name}: {path}")
        missing = [MODALITY_OPTIONS[name] for name in required if name not in entries]
    else:
        # Exact spelling matters even on case-insensitive filesystems.
        by_filename = {entry.name: entry for entry in args.input_dir.iterdir()}
        canonical = {filename.casefold(): filename for filename in MODALITY_FILENAMES.values()}
        incorrect = [f"{name} (expected {canonical[name.casefold()]})" for name in by_filename
                     if name.casefold() in canonical and name != canonical[name.casefold()]]
        if incorrect:
            raise ValueError("Incorrect capitalization of reconstruction filenames: " + ", ".join(sorted(incorrect)))
        entries = {name: by_filename[filename] for name, filename in MODALITY_FILENAMES.items()
                   if filename in by_filename}
        if optional_convct and "ConvCT" in entries and not entries["ConvCT"].is_file():
            raise ValueError("Optional ConvCT is not a file")
        missing = [MODALITY_FILENAMES[name] for name in required
                   if name not in entries or not entries[name].is_file()]
    if missing:
        raise ValueError("Missing required reconstruction files: " + ", ".join(missing))
    if optional_convct and "ConvCT" in entries:
        required.append("ConvCT")
    case_id = validate_case_id(args.case_id if args.case_id is not None else args.input_dir.absolute().name)
    return ExecutionPlan(
        args.input_dir, args.output_dir, model, args.spine_segmentation,
        args.spine_model_dir, args.cache_dir,
        None if supplied else getattr(args, "split", True),
        tuple((name, entries[name]) for name in required),
        args.install_models, args.validate_only, case_id,
    )


def parse_arguments(argv=None):
    """Parse and validate CLI input, returning an immutable execution plan."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return validate_arguments(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


def cli(argv=None):
    plan = parse_arguments(argv)
    try:
        from benchmark_nifti import case_output_paths, validate_plan_inputs
        validated_inputs = validate_plan_inputs(plan)
        if plan.lesion_model is None:
            from benchmark_spine_inference import validate_spine_inputs
            validated_inputs = validate_spine_inputs(plan, validated_inputs)
        else:
            _, segmentation, _ = validated_inputs
            if segmentation is not None:
                from benchmark_spine_inference import validate_labels
                validate_labels(segmentation)
    except (ValueError, OSError) as error:
        print("V3 input validation failed: {}".format(error), file=sys.stderr)
        return 2
    print("V3 Phase 4B execution plan (NIfTI content validated and inputs aligned in RAS)")
    print(f"Case: {plan.case_id}")
    if plan.input_dir is not None:
        print(f"Input: {plan.input_dir}")
    else:
        for name, path in plan.modality_paths:
            print(f"Input {name}: {path}")
    print(f"Output destination (not created): {plan.output_dir}")
    if plan.spine_segmentation is None:
        print("Automatic spine segmentation: model 802, fold_all; split={}".format(plan.split))
        print("Spine model source: {}".format(plan.spine_model_dir or "managed model cache"))
    else:
        print("Supplied spine segmentation: {}; model 802 bypassed; split not applicable".format(plan.spine_segmentation))
    if plan.lesion_model is None:
        print("Lesion segmentation: skipped")
    else:
        print("Lesion model: {}; folds: 0 1 2 3 4".format(plan.lesion_model.dataset_name))
        print("Ordered lesion channels: " + ", ".join(plan.lesion_model.channel_names))
    print("Required reconstructions: " + (", ".join(name for name, _ in plan.modality_paths) or "none"))
    from benchmark_model_installation import default_cache_dir
    print("Cache: {}".format(plan.cache_dir or default_cache_dir()))
    if plan.validate_only:
        print("Validation only: no installation, inference, or output creation was attempted.")
        return 3
    if plan.lesion_model is None:
        from benchmark_spine_inference import run_spine_only, SpineInferenceError
        from benchmark_model_installation import InstallationError
        try:
            result = run_spine_only(plan, validated_inputs)
        except InstallationError as error:
            print("V3 model resolution failed: {}".format(error), file=sys.stderr)
            return 4
        except (SpineInferenceError, ValueError, OSError) as error:
            print("V3 spine execution failed: {}".format(error), file=sys.stderr)
            if isinstance(error, FileExistsError):
                print("Output publication collided with an existing destination, which was preserved. Retry to obtain a new run ID or choose another output directory.", file=sys.stderr)
            return 5
        print("Spine segmentation completed: {}".format(result / case_output_paths(plan)["labelled_output"]))
        print("Run manifest: {}".format(result / "run.json"))
        return 0
    from benchmark_lesion_inference import run_lesion_pipeline, LesionInferenceError
    from benchmark_spine_inference import SpineInferenceError
    from benchmark_model_installation import InstallationError
    try:
        result = run_lesion_pipeline(plan, validated_inputs)
    except InstallationError as error:
        print("V3 model resolution failed: {}".format(error), file=sys.stderr)
        return 4
    except (LesionInferenceError, SpineInferenceError, ValueError, OSError) as error:
        print("V3 lesion pipeline failed: {}".format(error), file=sys.stderr)
        if isinstance(error, FileExistsError):
            print("Output destination already exists and was preserved. Retry for a new run ID or choose another output directory.", file=sys.stderr)
        return 5
    print("Lesion segmentation completed: {}".format(result / case_output_paths(plan)["lesion_output"]))
    print("Run manifest: {}".format(result / "run.json"))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
