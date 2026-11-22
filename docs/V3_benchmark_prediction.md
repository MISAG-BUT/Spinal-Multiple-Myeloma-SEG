# V3 · Benchmark model segmentation

[Back to main README](../README.md)

[Pipeline](#pipeline) · [Inputs](#input-data) · [Running V3](#running-v3) · [CLI reference](#command-line-reference) · [Cache](#model-installation-and-cache) · [Outputs](#output-structure) · [Troubleshooting](#limitations-and-troubleshooting)

V3 applies 19 spectral lesion models to aligned dual-energy CT NIfTI data. It can predict the spine with model 802, predict lesions using supplied spine information or run both steps together. V3 handles orientation and manages required model downloads. The [Model Zoo](Model_zoo.md) lists exactly which reconstructions each checkpoint requires.

The [benchmarking project](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking) used manually corrected vertebral masks and patient-grouped five-fold cross-validation, predicting held-out examinations with the corresponding fold. For a new case V3 uses the five-fold lesion ensemble. V3 can generate a spine mask automatically and applies its own five-voxel padded crop by default. The public benchmark sources do not establish the exact training-data crop margin. See [scientific context](Model_zoo.md#scientific-context) for the scope of reported benchmark results.

## Pipeline

1. Select a lesion model or spine-only mode. The model registry determines the required reconstructions and their order.
2. Validate the input files and compare their spatial grids after orientation normalization. V3 permutes and flips voxel axes into internal RAS. This operation updates geometry and does not interpolate voxel values or register images. nnU-Net still performs its own configured preprocessing and resampling for prediction.
3. Resolve only the required models after input validation. Run 802 on ConvCT or use supplied spine information. Automatic spine prediction splits the internal RAS volume along Z by default and rejoins the two predictions.
4. Convert nonzero anatomical labels to a binary spine mask. For lesion prediction, use its bounding box, five voxels of padding and bounds clipped to the RAS grid. Crop all required images consistently and build the lesion model's numbered channels. With `--already_cropped`, use the provided grid directly.
5. Run lesion inference on folds 0–4. Place the label volume into the full reference grid when an application crop was made, then undo the internal orientation transform. Spine-only mode ends after the spine outputs are prepared.
6. Write the masks, manifest and any requested intermediates within staging. Atomically publish the completed case directory.

## Input data

Use 3D NIfTI data from one examination. Declare spatial units as millimeters, meters or microns in each NIfTI header. Unknown spatial units are rejected. The reconstructions and any mask must share shape and physical grid after orientation normalization. The geometry check uses spacing, affine and coded qform/sform and rejects physical differences above its 0.001 mm tolerance. An image in a different orientation can pass if axis permutation and flipping align it to the same grid. Misaligned modalities must be aligned outside V3.

| Reconstruction | Exact filename in `--input_dir` | Explicit argument |
|---|---|---|
| Conventional CT | `ConvCT.nii.gz` | `--path_to_convCT_nifti` |
| VMI 40 keV | `VMI_40.nii.gz` | `--path_to_VMI40_nifti` |
| VMI 80 keV | `VMI_80.nii.gz` | `--path_to_VMI80_nifti` |
| VMI 120 keV | `VMI_120.nii.gz` | `--path_to_VMI120_nifti` |
| Calcium suppression 25 | `CaSupp_25.nii.gz` | `--path_to_CaSupp25_nifti` |
| Calcium suppression 50 | `CaSupp_50.nii.gz` | `--path_to_CaSupp50_nifti` |
| Calcium suppression 75 | `CaSupp_75.nii.gz` | `--path_to_CaSupp75_nifti` |
| Calcium suppression 100 | `CaSupp_100.nii.gz` | `--path_to_CaSupp100_nifti` |

Choose one input form:

- **Standard directory:** use these exact, case-sensitive filenames. Include ConvCT for automatic spine prediction and the modalities required by the lesion model. The directory basename is the default case ID.
- **Explicit paths:** pass one `.nii` or `.nii.gz` file per required modality. Include `--case_id`. Unused explicit reconstructions are rejected. A supplied-mask spine-only run can accept optional ConvCT as its reference.
- **Prepared nnU-Net channels:** `--nnunet_input_dir` requires a lesion model and explicit case ID. Supply exactly one file per channel as `<prefix>_0000.nii.gz`, `<prefix>_0001.nii.gz` and so on, using `data` as the default prefix. Files under that prefix must match the exact expected suffixes and resolve to distinct input paths. Follow [the model-specific order](Model_zoo.md#exact-channel-order). The final channel must be a nonempty binary mask with values 0 and 1. This mode bypasses model 802.

Case IDs and prepared channel prefixes accept 1–80 ASCII letters, digits, underscores or hyphens, starting with a letter or digit. Windows device names such as `CON` and `LPT1` are rejected.

With `--spine_segmentation`, pass a nonempty, spatially aligned spine NIfTI containing nonnegative integer labels. V3 preserves the anatomical labels, derives the binary channel and bypasses model 802. For model 710, the only other required input in this mode is VMI 40 keV.

## Configuration

[config.py](../config.py) offers `V3_MODEL_CACHE_DIR` for V3. `None` selects the repository cache or the installed-package fallback. Set an absolute path there to override it. `--cache_dir` takes precedence. Choose the case inputs, prediction mode, output path, external spine directory and split behavior using the CLI.

## Running V3

Run these examples from the repository root after [installation](Installation.md). Substitute your own data paths. If the GUI extra is installed, include `--extra gui` in each `uv run` so synchronization retains it.

The launcher and Python script accept the same arguments. In any prediction command below, you can replace `uv run --locked spinal-run-nnunet-benchmark-prediction` with `uv run --locked python run_prediction_of_nnUNet_benchmark_models.py` or `.venv/bin/python run_prediction_of_nnUNet_benchmark_models.py`. In VS Code, open [the V3 script](../run_prediction_of_nnUNet_benchmark_models.py), select the project interpreter and pass the arguments in the integrated terminal or debugger. See [VS Code setup](Installation.md#run-in-vs-code).

```bash
.venv/bin/python run_prediction_of_nnUNet_benchmark_models.py --list_models
uv run --locked python run_prediction_of_nnUNet_benchmark_models.py --model_info 710
```

These direct-script commands inspect the registry and require no models or images.

### Inspect models and validate inputs

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction --list_models
uv run --locked spinal-run-nnunet-benchmark-prediction --model_info 710

uv run --locked spinal-run-nnunet-benchmark-prediction \
  --input_dir /path/to/Myel_001 --output_dir /path/to/output \
  --lesion_model 710 --validate_only
```

The first two commands need no input or weights. The last command checks ConvCT and VMI 40 keV, prints the plan and returns **3** on success. It creates no output, resolves no weights and runs no inference. This is useful before any download or run.

### End-to-end spine and lesion prediction

Use `ConvCT.nii.gz` and `VMI_40.nii.gz` in the standard directory for 802 → 710. Missing weights are resolved automatically. The case receives anatomical spine, binary spine and lesion masks in the original ConvCT grid.

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction \
  --input_dir /path/to/Myel_001 --output_dir /path/to/output --lesion_model 710
```

For explicit paths and retained working images, use:

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction \
  --case_id Myel_001 --output_dir /path/to/output --lesion_model 710 \
  --path_to_convCT_nifti /path/to/conventional.nii.gz \
  --path_to_VMI40_nifti /path/to/vmi40.nii.gz --keep_intermediates
```

To use a different spectral model, supply its complete reconstruction set. This example uses 717, with ConvCT for 802 and all three VMI reconstructions for lesions, and publishes to the ConvCT grid.

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction \
  --input_dir /path/to/case-with-convct-and-all-vmi --output_dir /path/to/output --lesion_model 717
```

### Spine-only and custom spine checkpoint

These two forms use only ConvCT and write anatomical and binary masks in the source grid. The first uses managed model 802, which is downloaded if needed. The second uses an existing trusted trainer directory containing `dataset.json`, `plans.json` and the `fold_all` checkpoint. It also illustrates a full-volume spine pass with `--split False`.

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction \
  --input_dir /path/to/Myel_001 --output_dir /path/to/output --skip_lesion_segmentation

uv run --locked spinal-run-nnunet-benchmark-prediction \
  --input_dir /path/to/Myel_001 --output_dir /path/to/output --skip_lesion_segmentation \
  --spine_model_dir /path/to/model802/nnUNetTrainer__nnUNetPlans__3d_fullres --split False
```

### Lesion-only with supplied spine labels

Only model 710 is needed. The following example uses a custom cache, requires its verified installed model and writes in the original VMI 40 keV grid. The labelled and binary spine outputs accompany the lesion mask and the original supplied mask is copied for provenance.

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction \
  --case_id Myel_001 --output_dir /path/to/output --lesion_model 710 \
  --path_to_VMI40_nifti /path/to/vmi40.nii.gz \
  --spine_segmentation /path/to/spine-labels.nii.gz \
  --cache_dir /path/to/model-cache --offline
```

With an existing mask, `--skip_lesion_segmentation` can instead publish just the labelled and binary spine masks without resolving a model. Pass the mask, case ID and output directory. Optional ConvCT selects its grid as the reference.

### Prepared and already cropped channels

For 710, prepare `data_0000.nii.gz` (VMI 40 keV) and `data_0001.nii.gz` (binary spine). No ConvCT or model 802 is used. The first command applies the standard crop and restores to the original prepared image grid. The second expects `caseA_0000.nii.gz` and `caseA_0001.nii.gz` of an already cropped case. The final masks stay on that cropped grid in its original orientation. Both commands publish a lesion mask, the binary spine mask and the copied input spine channel. The required lesion model is downloaded unless an offline policy is selected.

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction \
  --nnunet_input_dir /path/to/prepared --case_id Myel_001 \
  --output_dir /path/to/output --lesion_model 710

uv run --locked spinal-run-nnunet-benchmark-prediction \
  --nnunet_input_dir /path/to/cropped --nnunet_case_prefix caseA --already_cropped \
  --case_id Myel_001 --output_dir /path/to/output --lesion_model 710
```

## Command-line reference

### Case and input paths

| Argument | Value | Default and constraints |
|---|---|---|
| `--input_dir` | Directory | None. Exact filenames from [Input data](#input-data). Exclusive with explicit and prepared input forms. |
| `--case_id` | Case identifier | Standard input directory basename. Required for explicit or prepared inputs. |
| Eight modality options | One `.nii` or `.nii.gz` path per option | None. See [the complete argument and filename table](#input-data). Supply only the required reconstructions. |
| `--spine_segmentation` | Labelled spine NIfTI | None. Bypasses 802, requires an aligned, nonempty label mask. Conflicts with prepared inputs, `--spine_model_dir` and an explicit `--split`. |
| `--nnunet_input_dir` | Directory | None. Requires a lesion model and case ID. Conflicts with standard or explicit input paths, supplied spine labels, external spine models and explicit splitting. |
| `--nnunet_case_prefix` | Filename prefix | `data`. Only for prepared inputs. |
| `--already_cropped` | Flag | Off. Only for prepared inputs. Skips the application crop and full-volume restoration. |

### Prediction and output

| Argument | Value | Default and constraints |
|---|---|---|
| `--lesion_model` | Integer 700–718 | None. Choose this or `--skip_lesion_segmentation` for prediction and validation. |
| `--skip_lesion_segmentation` | Flag | Off. Publish spine masks without lesion inference. |
| `--spine_model_dir` | Model 802 trainer directory | None, use managed model 802 when needed. Automatic spine prediction only. |
| `--split` | `True` or `False` | `True` for automatic spine prediction. A single RAS Z slice is handled as unsplit. |
| `--output_dir` | Parent directory | Required for prediction and validation. Completed case runs are published underneath. |
| `--keep_intermediates` | Flag | Off. Retain working NIfTIs where generated by the selected execution path. |

### Cache, validation and information

| Argument | Value | Default and constraints |
|---|---|---|
| `--cache_dir` | Directory | Configured cache or platform default. Overrides `V3_MODEL_CACHE_DIR`. |
| `--install_models` | Flag | Compatibility option. Automatic installation already occurs when models are missing. Conflicts with `--offline` and `--validate_only`. Does not stop before prediction. |
| `--offline` | Flag | Off. Use verified installed models in the managed cache. Explicit external spine directories are still structurally checked. |
| `--validate_only` | Flag | Off. Checks inputs without model resolution, prediction or output creation. Success returns `3`. |
| `--list_models` | Flag | List model IDs, datasets and ordered channels. Standalone. |
| `--model_info` | 802 or 700–718 | Show channels and numbered filenames, folds, archive size and record. Standalone, exclusive with `--list_models`. |
| `-h`, `--help` | Flag | Display help and exit. |

## Model installation and cache

The managed downloader resolves the selected weights from [the original spine collection](https://doi.org/10.5281/zenodo.18598645) and [the benchmark lesion collection](https://doi.org/10.5281/zenodo.19708429). It checks free disk space, verifies archive size and the pinned MD5, checks paths before extraction and publishes with an exclusive rename. Installed files are checked against a local SHA-256 receipt before reuse. Keep the cache writable only by trusted users. These checks establish consistency with the expected artifact and local receipt. Load checkpoint files only from trusted sources.

The default source-checkout cache is `Model_cache/` at the repository root, independent of the current working directory. The standard installed package uses a platform user cache, for example `$XDG_CACHE_HOME/spinal-multiple-myeloma-seg` on Linux or `~/.cache/spinal-multiple-myeloma-seg` if that variable is unset. The cache can contain both compressed archives and extracted files. The original archive is about 3.68 GB and an individual lesion archive about 1.66 GB, before extraction. Allow additional room for installation and image staging.

`uv sync` installs Python software. `uv run --offline` restricts uv's own dependency resolution. The application argument `--offline`, placed after the launcher, prevents V3 from downloading model weights. Offline prediction needs a verified model installation. Validation-only can still inspect inputs without installed models.

For model installation **without inference**, use the existing Python API. The CLI's compatibility flag `--install_models` still runs prediction. From the checkout, this command resolves or installs exactly 802 and 710 in the default cache and needs no image inputs:

```bash
.venv/bin/python -c "from utils.benchmark.model_installation import ModelInstaller; installer=ModelInstaller(); print(installer.resolve(802).model_directory); print(installer.resolve(710).model_directory)"
```

Use `ModelInstaller(cache_dir='/path/to/model-cache')` for a different destination. Pass `allow_download=False` to `resolve()` to check a managed installation without downloads.

## Output structure

```text
output/<case>_output/
├── run.json
├── Segmentation/
│   ├── Spine_segmentation_final/
│   │   ├── <case>_spine_segmentation.nii.gz
│   │   ├── <case>_spine_binary.nii.gz
│   │   └── <case>_spine_supplied.nii.gz
│   └── Lesion_segmentation_final/
│       └── <case>_lesions_segmentation.nii.gz
└── Intermediates/
    ├── RAS/<modality>.nii.gz
    ├── Spine_chunks/part_*/
    │   ├── ConvCT.nii.gz
    │   └── spine.nii.gz
    └── Lesion_crop/
        ├── inputs/case_0001_0000.nii.gz
        ├── inputs/...
        └── lesions.nii.gz
```

- Anatomical and binary masks are written for automatic or supplied-label spine runs. Prepared channels produce the binary mask only.
- The original supplied or prepared spine file is copied only when present and retains its `.nii` or `.nii.gz` suffix.
- A lesion mask is present only in lesion modes.
- `Intermediates/` is opt-in and contains only artifacts produced in the selected mode. RAS copies are written for inputs that needed reorientation. Spine chunks are saved when splitting actually occurred. Cropped channels and prediction are saved when the application crop was performed. In `--already_cropped` mode no `Lesion_crop/` directory is retained.
- `run.json` records paths, source inputs, orientation transforms, model provenance and geometry. Lesion runs record crop bounds with exclusive upper limits, padding and reference grid, or mark the input as already cropped.

Automatic spine runs use the original ConvCT as reference. With supplied spine and lesion modes, the first CT modality required by the lesion model is the reference. Supplied spine-only uses ConvCT if present, otherwise the mask. Prepared channels use their first CT modality. Final published masks restore the reference shape and orientation, affine, spacing, and qform/sform transforms and codes. With already cropped data this applies to the supplied cropped grid. The original supplied copy is preserved independently.

V3 stages a complete case before atomic publication. A name collision at `<case>_output` causes a retry with a unique suffix and the previous run remains in place. The filesystem must support the required exclusive rename. This differs from the working-directory replacement performed by V1 and V2.

## Limitations and troubleshooting

| Issue | What to check |
|---|---|
| Missing files, extra inputs or wrong channels | Check the exact filename spelling, model ID and [channel order](Model_zoo.md#exact-channel-order). An automatic spine run needs ConvCT even if the lesion model omits it. |
| Geometry or orientation error | Check spatial correspondence and coded transforms on every NIfTI. Reorientation is an axis transformation. Register or resample inputs separately when necessary. |
| Invalid or empty spine mask | Check that supplied labels are nonnegative integers and that prepared masks are binary and nonempty. A lesion prediction may validly be empty. |
| Offline model error | Check the configured cache, the selected version and the installation receipt. An archive alone is insufficient for offline prediction. |
| Memory pressure | Use spine splitting where applicable and monitor GPU and host memory. nnU-Net may switch prediction accumulation to CPU, which can increase RAM use. |
| Output already cropped | The application cannot reconstruct a missing full-volume grid from cropped channels. Use full prepared channels to get a full-grid prediction. |
| Publication fails | Check output permissions and support for the platform's atomic exclusive rename. An existing completed run is preserved. |

Execution returns `0` on success, `2` for argument or input errors, `3` for a successful validation-only plan, `4` for model resolution errors and `5` for prediction or publication failures.

Automated coverage includes input handling, registry configurations, cache and publication behavior. Real end-to-end regression covered models 802 → 710. The remaining lesion configurations have not all been run end to end in this setup. Technical agreement on a reference examination does not establish clinical performance or the number of clinically confirmed lesions.

[Installation](Installation.md) · [Model Zoo](Model_zoo.md) · [Original NIfTI workflow](V2_new_data_prediction.md) · [Dataset Viewer](Dataset_viewer.md)
