"""V3 Phase 4B: standalone spine and benchmark lesion execution.

Later stages must binarize nonzero spine labels before a shared five-voxel crop,
translate cropped affines, retain published normalization/plans, and reconstruct
on the original grid. Model installation must use isolated record-specific roots.
Imaging utilities are loaded only on execution, never during argument parsing.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple

from benchmark_model_registry import LESION_MODELS, ModelSpec, get_lesion_model


MODALITY_FILENAMES = {name: name + ".nii.gz" for name in (
    "ConvCT", "VMI_40", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_50",
    "CaSupp_75", "CaSupp_100",
)}


@dataclass(frozen=True)
class ExecutionPlan:
    input_dir: Path
    output_dir: Path
    lesion_model: Optional[ModelSpec]
    spine_segmentation: Optional[Path]
    spine_model_dir: Optional[Path]
    cache_dir: Optional[Path]
    split: Optional[bool]
    modality_paths: Tuple[Tuple[str, Path], ...]
    install_models: bool = False
    validate_only: bool = False


def build_parser():
    parser = argparse.ArgumentParser(
        description="V3 benchmark prediction (Phase 4B: spine and benchmark lesions)",
        allow_abbrev=False,
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--lesion_model", type=int, choices=sorted(LESION_MODELS))
    selection.add_argument("--skip_lesion_segmentation", action="store_true")
    parser.add_argument("--spine_segmentation", type=Path)
    parser.add_argument("--spine_model_dir", type=Path)
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--install_models", action="store_true",
                        help="Authorize missing-model installation; without this flag no downloads are allowed")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate inputs and print the plan without installation, inference, or output creation")
    # Suppression distinguishes explicit input from the implicit V2 default.
    parser.add_argument("--split", type=lambda x: str(x).lower() == "true",
                        default=argparse.SUPPRESS,
                        help="V2 parsing: case-insensitive 'true' enables splitting; all other values disable it. Default: True")
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
    if not args.input_dir.is_dir():
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
    # Enumerate actual names: Path.exists alone is case-insensitive on Windows.
    entries = {entry.name: entry for entry in args.input_dir.iterdir()}
    canonical = {filename.casefold(): filename for filename in MODALITY_FILENAMES.values()}
    incorrect = ["{} (expected {})".format(name, canonical[name.casefold()])
                 for name in entries if name.casefold() in canonical
                 and name != canonical[name.casefold()]]
    if incorrect:
        raise ValueError("Incorrect capitalization of reconstruction filenames: " + "; ".join(sorted(incorrect)))
    missing = [MODALITY_FILENAMES[name] for name in required
               if MODALITY_FILENAMES[name] not in entries
               or not entries[MODALITY_FILENAMES[name]].is_file()]
    if missing:
        raise ValueError("Missing required reconstruction files: " + ", ".join(missing))
    return ExecutionPlan(
        args.input_dir, args.output_dir, model, args.spine_segmentation,
        args.spine_model_dir, args.cache_dir,
        None if supplied else getattr(args, "split", True),
        tuple((name, entries[MODALITY_FILENAMES[name]]) for name in required),
        args.install_models,
        args.validate_only,
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
        from benchmark_nifti import validate_plan_inputs
        validated_inputs = validate_plan_inputs(plan)
        if plan.lesion_model is None:
            from benchmark_spine_inference import validate_spine_inputs
            validated_inputs = validate_spine_inputs(plan, validated_inputs)
        elif validated_inputs[1] is not None:
            from benchmark_spine_inference import validate_labels
            validate_labels(validated_inputs[1])
    except (ValueError, OSError) as error:
        print("V3 input validation failed: {}".format(error), file=sys.stderr)
        return 2
    print("V3 Phase 4B execution plan (NIfTI content and geometry validated)")
    print("Input: {}\nOutput destination (not created): {}".format(plan.input_dir, plan.output_dir))
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
        print("Spine segmentation completed: {}".format(result / "spine" / "spine_labelled.nii.gz"))
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
    print("Lesion segmentation completed: {}".format(result / "lesion" / "lesion_labelled.nii.gz"))
    print("Run manifest: {}".format(result / "run.json"))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
