# Spinal-Multiple-Myeloma-SEG

## V3 benchmark prediction — Phase 4B (end-to-end spine and lesions)

V3 adds benchmark inference alongside the V1 TCIA/DICOM and V2 NIfTI workflows.
**Spine-only and end-to-end benchmark lesion execution are implemented.**
End-to-end models 802 and 710 were validated on real RAS and LPS inputs on
Linux with an RTX 5080.
`--skip_lesion_segmentation` executes the spine workflow and exits **0** only
after publishing a validated completed run. `--lesion_model 700–718` executes
spine preparation and the selected five-fold lesion model, also exiting **0**
only after the entire run has been published.
Invalid input validation exits **2**, model resolution/installation errors exit
**4**, and inference/reconstruction/publication errors exit **5**. Prediction
installs missing required models automatically. Use `--offline` to require an
existing verified installation. `--validate_only` checks the input geometry and
prints the plan without resolving models or writing outputs (exit **3**).

After package installation, use `spinal-run-nnunet-benchmark-prediction`, or run
`python run_prediction_of_nnUNet_benchmark_models.py` directly.

```bash
# Automatic model-802 spine inference plus benchmark lesion model 710:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results --lesion_model 710

# Supplied spine mask: model 710 needs VMI_40, but no ConvCT:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --spine_segmentation /data/masks/spine.nii.gz --lesion_model 710

# Automatic spine-only inference (requires an installed model 802 and CUDA):
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --skip_lesion_segmentation --split True
```

One input directory represents one examination. Exact case-sensitive filenames
are `ConvCT.nii.gz`, `VMI_40.nii.gz`, `VMI_80.nii.gz`, `VMI_120.nii.gz`,
`CaSupp_25.nii.gz`, `CaSupp_50.nii.gz`, `CaSupp_75.nii.gz`, and
`CaSupp_100.nii.gz`. Only selected-model reconstructions are required, plus
ConvCT when no spine segmentation is supplied. Discovery is not recursive;
unrelated files are ignored. Sources are never modified.

Alternatively, provide the original NIfTI paths directly. Both `.nii` and
`.nii.gz` are accepted. Do not combine explicit paths with `--input_dir`.
Each explicit modality may be specified once. Unused explicit modality options
are rejected.

```bash
uv run spinal-run-nnunet-benchmark-prediction \
  --case_id Myel_001 --output_dir /data/results --lesion_model 710 \
  --path_to_convCT_nifti /data/original/Myel_001_conv.nii.gz \
  --path_to_VMI40_nifti /data/original/Myel_001_monoe_40kev.nii.gz
```

| Reconstruction | Explicit option |
|---|---|
| ConvCT | `--path_to_convCT_nifti` |
| VMI_40 | `--path_to_VMI40_nifti` |
| VMI_80 | `--path_to_VMI80_nifti` |
| VMI_120 | `--path_to_VMI120_nifti` |
| CaSupp_25 | `--path_to_CaSupp25_nifti` |
| CaSupp_50 | `--path_to_CaSupp50_nifti` |
| CaSupp_75 | `--path_to_CaSupp75_nifti` |
| CaSupp_100 | `--path_to_CaSupp100_nifti` |

With `--input_dir`, the folder name is the default case ID. Supply `--case_id`
to override it or to identify a run using explicit paths. An ID has 1–80 ASCII
letters, digits, underscores or hyphens and starts with a letter or digit.
Reserved device names are rejected. Case IDs appear in output paths and
filenames.

The registry is authoritative for exact channel order. The required lesion
reconstructions are listed below; a binary spine mask is appended as the **last
channel** for every model. Automatic spine inference additionally needs ConvCT
even when the lesion model excludes it. Supplied-mask inference needs only the
selected model's reconstructions, so model 710 can run without ConvCT.

| Lesion model | Required reconstructions |
|---|---|
| 700 | All eight except ConvCT |
| 701 | All eight except VMI_40 |
| 702 | All eight except VMI_80 |
| 703 | All eight except VMI_120 |
| 704 | All eight except CaSupp_25 |
| 705 | All eight except CaSupp_50 |
| 706 | All eight except CaSupp_75 |
| 707 | All eight except CaSupp_100 |
| 708 | All eight |
| 709 | ConvCT |
| 710 | VMI_40 |
| 711 | VMI_80 |
| 712 | VMI_120 |
| 713 | CaSupp_25 |
| 714 | CaSupp_50 |
| 715 | CaSupp_75 |
| 716 | CaSupp_100 |
| 717 | VMI_40, VMI_80, VMI_120 |
| 718 | CaSupp_25, CaSupp_50, CaSupp_75, CaSupp_100 |

Prediction requires `--output_dir` and exactly one of
`--lesion_model 700–718` or `--skip_lesion_segmentation`. Choose directory,
explicit-file, or prepared-channel inputs as described above and below.
Validation-only and installation flags cannot be combined.
`--spine_model_dir` must name an existing directory and conflicts with
`--spine_segmentation`. With model resolution requested, it must point to the
model-802 `nnUNetTrainer__nnUNetPlans__3d_fullres` directory, whose metadata and
`fold_all/checkpoint_final.pth` are structurally validated. Use this option only
with checkpoints from a trusted source. The pinned nnU-Net loads these files with
PyTorch's pickle-capable checkpoint loader. Structural validation establishes
neither checkpoint authenticity nor training provenance.
Explicit `--split` also conflicts with a supplied segmentation. Otherwise its
default is True, with V2 parsing: case-insensitive `true` enables splitting and
every other supplied value disables it. See the cache configuration below.
Validation-only runs do not create a cache.

### Model information, installation and cache

```bash
uv run spinal-run-nnunet-benchmark-prediction --list_models
uv run spinal-run-nnunet-benchmark-prediction --model_info 710

# Use only verified local models in an alternate cache.
uv run spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/Myel_001 --output_dir /results --lesion_model 710 \
  --cache_dir /data/model-cache --offline
```

Model queries use only the registry. Before a missing-model download, the CLI
reports the model, archive size, cache path and available disk space. The archive
plus extraction estimate is preliminary. The verified archive provides exact
selected-member sizes for the extraction check. A verified installed model is
reused. An incomplete or corrupt installation is rejected and preserved for
inspection. Supplied spine labels and prepared binary spine inputs bypass model
802. Spine-only runs select only model 802, and `--spine_model_dir` can provide
its external trainer directory.

`--cache_dir` overrides `V3_MODEL_CACHE_DIR` in `config.py`. Set that constant to
an absolute path or leave it as `None` to use the ignored `Model_cache/` directory
beside the source project. A non-editable installation uses the OS user cache
instead when no override is set. On Linux the user fallback follows
`XDG_CACHE_HOME` or `~/.cache`. The CLI prints the absolute path and its source.

The older `--install_models` option remains accepted as a compatibility flag and
conflicts with `--offline`. Prediction with `--offline` fails if a required
installation is missing or invalid. `--validate_only` works even with an empty
cache and never downloads or initializes a model. Launcher `--offline` controls
model access. `uv run --offline` controls Python dependency access.

### Prepared nnU-Net channels

```bash
uv run spinal-run-nnunet-benchmark-prediction \
  --nnunet_input_dir /data/prepared --case_id Myel_001 \
  --output_dir /results --lesion_model 710 --offline
```

Use a single selected filename prefix (default `data`). For model 710,
`data_0000.nii.gz` is VMI_40 and `data_0001.nii.gz` is a nonempty binary
`mask_spine` with values 0 and 1. Use `--nnunet_case_prefix` to select another
prefix. The user must map physical modalities to the registered indices shown
by `--model_info`. Image intensity and filenames alone cannot establish which
physical modality an image represents. This mode conflicts with other input
options, `--spine_segmentation`, `--spine_model_dir` and `--split`. Supply
multilabel spine through the normal `--spine_segmentation` workflow.

Full prepared inputs use the same application crop and five-voxel padding as
other lesion runs. Add `--already_cropped` when the input grid is already the
desired output grid. This skips the application crop, while nnU-Net still runs
its model-defined preprocessing, resampling and normalization. The lesion mask
returns to the supplied grid. Prepared runs publish the binary spine mask and a
byte-for-byte original mask copy. Only runs with anatomical spine labels publish
the multilabel spine output.

```bash
uv run spinal-run-nnunet-benchmark-prediction \
  --nnunet_input_dir /results/Myel_001_output/Intermediates/Lesion_crop/inputs \
  --nnunet_case_prefix case_0001 --already_cropped --case_id Myel_001_crop \
  --output_dir /results --lesion_model 710 --offline
```

### Optional working images

`--keep_intermediates` retains relevant NIfTI images in the same atomic run.

```bash
uv run spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/Myel_001 --output_dir /results --lesion_model 710 \
  --offline --keep_intermediates
```

```text
Intermediates/
  RAS/<channel>.nii.gz
  Spine_chunks/part_00/{ConvCT,spine}.nii.gz
  Spine_chunks/part_01/{ConvCT,spine}.nii.gz
  Lesion_crop/inputs/case_0001_0000.nii.gz
  Lesion_crop/inputs/...                  # ordered channels, including mask_spine
  Lesion_crop/lesions.nii.gz
```

Only nonidentity RAS conversions, actual split predictions and actual application
crops are retained. Staged split and crop files are moved into the retained
folders. The manifest records their relative paths and geometry. This option
can substantially increase disk use for large or multichannel volumes.

### End-to-end lesion execution

```bash
# All eight reconstructions, automatic spine:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results --lesion_model 708

# Supplied spine; ConvCT plus binary conditioning mask for model 709:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --spine_segmentation /data/masks/spine.nii.gz --lesion_model 709

# Validate supplied-mask CaSupp-only model inputs; no model resolution or CUDA:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --spine_segmentation /data/masks/spine.nii.gz --lesion_model 718 --validate_only
```

`utils/benchmark/lesion_inference.py` reuses the Phase 4A spine adapter or the supplied
labels, converts nonzero spine labels to binary, and uses the existing common
five-voxel padded crop. Crop bounds are clipped to the original image dimensions
and stored with exclusive upper bounds. Ordered channel files are constructed
internally; users never need `_0000` filenames. All channels use one cropped
reference geometry after passing the existing 0.001 mm alignment checks. This
assigns metadata to already-aligned voxel indices; it does not resample voxel
data or change intensities. The final channel is binary, never vertebral labels.

The lesion adapter checks `dataset.json` channels against the registry and
requires the published binary output convention (`0` background, `1` lesion).
It initializes `nnUNetTrainer__nnUNetPlans__3d_fullres` using
`checkpoint_final.pth` and folds **0, 1, 2, 3, 4**. It uses sequential explicit
file lists and the same CUDA sliding-window settings as the spine adapter.
Normalization/resampling inside nnU-Net follows the installed published plans;
no external normalization, morphology, or modality substitution is performed.

The cropped prediction must match the cropped input geometry and contain only
the model's valid binary labels. A **background-only lesion prediction is valid**
and produces a successful all-zero full-volume lesion mask. Fractional, negative,
nonfinite, or other labels are rejected rather than binarized. Reconstruction
inserts the crop by exact voxel indices, leaves background outside the crop, and
restores the reference dimensions, affine, qform/sform codes, and spatial units.
The reference is ConvCT for automatic spine mode, or the first required lesion
modality for supplied-mask lesion mode. No interpolation is performed here.
Integer-valued floating NIfTI storage (for example, exact `0.0` and `1.0`) is
accepted consistently with spine labels. There is no label rounding or epsilon
tolerance. Integer arrays, including uint64 spine labels above `2**53`, are
validated without converting them to floating point. Prediction-domain checks
use each model's allowed labels, separately from geometry tolerance.

A completed lesion run for `Myel_001` has this structure:

```text
<output_dir>/Myel_001_output/
  run.json
  Segmentation/
    Spine_segmentation_final/
      Myel_001_spine_segmentation.nii.gz
      Myel_001_spine_binary.nii.gz
      Myel_001_spine_supplied.nii.gz  # supplied mode only, retains .nii or .nii.gz
    Lesion_segmentation_final/
      Myel_001_lesions_segmentation.nii.gz
```

The run stays in private staging until both stages, output validation and the
manifest complete. Publication is atomic. If `Myel_001_output` already exists
or another process wins the publication race, the run is published under
`Myel_001_output-<unique-id>`. Completed runs and older V3 layouts stay in place.
Working CT images and intermediate predictions are removed by default.
The supplied-spine copy is byte-identical to the source.

Schema 5 of `run.json` records the case ID, original source basenames, relative
output paths, model verification and cache status, cache location and policy,
folds, input channel mapping, application crop status, split, orientation
transforms and retained intermediate geometry. The final spine and lesion filenames match V1/V2.
The legacy DICOM viewer still expects its own flat `<segmentations>/<case>/`
layout. For direct inspection, overlay the final NIfTI masks on the original
reference image in a NIfTI viewer.

### Scientific reproducibility review

The review used the clean local benchmarking repository at commit
`9cd13bd7d31eba930d6e925fd37cc8f126e63dd3` and the cached nnU-Net v2.6.4 tag
identified below. It did not inspect patient images or load model weights.

- **Channels and mask:** all 19 ordered channel lists in the benchmark's
  [model-configuration table](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking/blob/9cd13bd7d31eba930d6e925fd37cc8f126e63dd3/results/tables/model_configuration/model_configurations.csv)
  match the V3 registry. The benchmark [input documentation](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking/blob/9cd13bd7d31eba930d6e925fd37cc8f126e63dd3/README.md#input-data-and-output-segmentation)
  requires aligned reconstructions and a final binary mask, and states that
  vertebral masks were manually corrected. Automatic V3 masks are a different
  input-quality condition; comparable accuracy has not been demonstrated.
- **Crop and external preparation:** `prepare_full_model_folds()` in the
  [benchmark inference script](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking/blob/9cd13bd7d31eba930d6e925fd37cc8f126e63dd3/inference/run_predictions.sh#L143)
  copies existing `imagesTr` channel files into held-out fold folders. No original
  dataset-generation implementation is present in that checkout. Binary-mask
  bounding-box calculation, exact padding, inclusive/exclusive bounds, clamping,
  additional slice restrictions, prior intensity clipping/shifting, external
  resampling/reorientation, and training crop-affine handling therefore cannot
  be established from that script. V3 retains the requested five-voxel padded
  nonzero-mask crop; the legacy V2 implementation is not proof of the benchmark's
  original training preparation.
- **Normalization:** the benchmark's
  [configuration summary](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking/blob/9cd13bd7d31eba930d6e925fd37cc8f126e63dd3/results/tables/model_configuration/model_configuration_summary.csv)
  reports `ZScoreNormalization` for all models. nnU-Net v2.6.4's
  `DefaultPreprocessor` handles its internal nonzero crop, configured
  normalization, and resampling. V3 adds no intensity transforms and changes no
  published plans. This does not establish how the original `imagesTr` files
  were prepared before nnU-Net consumed them.
- **Inference is intentionally different from paper evaluation:**
  `run_full_models()` predicts held-out cases with `-f "$FOLD"`, one fold at a
  time. The benchmark README separately recommends `-f 0 1 2 3 4` for new-data
  inference. V3 follows the requested five-fold ensemble. It must not be used to
  reproduce held-out-fold scores on training-set cases, where other ensemble
  folds may have trained on those cases. Trainer/configuration and final
  checkpoint selection agree with the documented framework/defaults.
- **Reconstruction:** nnU-Net export restores predictions to its input grid.
  V3 additionally restores its external spinal crop by voxel insertion into a
  zero-filled full-size reference, preserving coded forms and spatial units.
  A corresponding benchmark full-volume restoration implementation is absent,
  so exact agreement with that additional step cannot be asserted.
- **Environment provenance:** the benchmark setup script pins nnU-Net commit
  `f6d221d1b79cd2173650f78f97ecfee273e0cf86`, which was not available in the local
  nnU-Net object cache. The local v2.6.4 tag verifies the V3 API contract, but not
  identity with that historical environment. No network retrieval was needed
  for this review.

Shared channel metadata is assigned only **after** all source grids pass the
0.001 mm physical-corner tolerance. Sub-tolerance differences are intentionally
reconciled to one reference without changing voxel values; larger Euclidean
offsets are rejected even when individual affine components differ by less than
0.001. This does not perform registration or prove anatomical correspondence.

Regression tests distinguish preflight model/metadata failures (exit 4) from
predictor, reconstruction, NIfTI/manifest writing, and publication failures
(exit 5). All failed runs retain existing results and remove their own staging
files, including partially written manifests. Unexpected programmer errors are
not broadly relabelled as ordinary prediction failures. The model 802 and 710
CUDA runs verify end-to-end operation on the tested inputs. Clinical accuracy
and reproduction of the published benchmarks require separate validation.

### Spine-only execution and outputs

```bash
# Use an already installed managed model; no automatic download:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --skip_lesion_segmentation --split True --cache_dir /data/v3-cache --offline

# Alternatively, use an external, structurally validated model-802 trainer folder:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --skip_lesion_segmentation --spine_model_dir /models/Dataset802/nnUNetTrainer__nnUNetPlans__3d_fullres

# Supplied mask: no model resolution, CUDA, or nnU-Net is needed:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --skip_lesion_segmentation --spine_segmentation /data/masks/spine.nii.gz
```

Automatic spine inference uses only ConvCT, model 802 `fold_all`, and
`checkpoint_final.pth`. Inference dependencies load lazily and require an
installed nnU-Net and CUDA-enabled PyTorch. The constructor, initialization,
sequential method, manager attributes, and export behavior were verified from a
clean cached nnU-Net **v2.6.4** tag at commit
`273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0`. The offline source-contract checks
run without loading checkpoints. The real model-802 CUDA runs separately
exercise that contract on the tested inputs. V3 leaves the global path and
nnU-Net environment configuration unchanged. V3 uses the supported sequential
file-prediction API with explicit one-channel input lists and truncated output
filename lists. In v2.6.4, passing directories would invoke a multiprocessing
pool during filename discovery even from the sequential method; explicit lists
avoid that path. This retains V2 sliding
window settings (step 0.5, Gaussian weighting, mirroring, CUDA device 0).

V2 has no size threshold: with splitting enabled, it partitions at `Z // 2` and
concatenates first-half then second-half predictions without overlap or label
reconciliation. V3 preserves that algorithm for `Z >= 2`, with the odd extra
slice in the second part. A one-slice volume is processed unsplit to avoid V2's
empty-part failure. Each part is submitted independently. Its affine is
`ras_affine @ voxel_translation(0, 0, z_start)`; V2's second-half origin
adjustment used only the world-Z component. V3 splits along Z in the internal RAS
grid and restores the merged prediction to the original reference grid.

Predictions must match their part's input grid, contain nonnegative discrete
labels from model 802's `dataset.json`, and merge to the original dimensions.
Background-only split parts are allowed. An entirely empty automatic result or
supplied segmentation is rejected, as is binary-mask conversion from an empty
segmentation: no spinal crop or anatomical conditioning region could be derived
for downstream lesion preparation, so an empty result is not treated as success.
Full-grid outputs restore the original reference affine, units, and coded
qform/sform transforms. Supplied masks retain their discrete label values
(including values above 255); negative/probabilistic/nonfinite masks are rejected
for spine execution. Their source is never modified. In supplied-mask mode,
ConvCT remains optional; when present it must match the segmentation within the
existing 0.001 mm contract and is the output reference. Otherwise the supplied
mask is the spine-only reference. Compatible supplied voxel labels are copied
by index onto the chosen reference grid without resampling; the original source
is also preserved byte-for-byte.

Spine-only runs use the same case directory and spine filenames shown above,
with no lesion directory. The manifest records the final relative paths.

`run.json` records completion status/time, automatic/supplied provenance, source
filenames (without absolute patient paths), requested/effective split and any
one-slice fallback, model ID/verification/artifact provenance/folds/checkpoint,
reference/output geometries, and relative output paths. The binary mask contains
1 for every nonzero spine label. In spine-only mode no lesion inference is
performed. All temporary inputs and predictions are
run-specific and cleaned on failure. A failure in either split publishes no
completed run and preserves existing results. An empty output root may remain
after a failed execution. No success is reported until the complete run is
published using the existing no-clobber directory mechanism.
Expected publication failures are reported without a traceback using exit code
5, including native filesystem errors raised by the shared installation helper;
they are not misclassified as model-resolution errors (code 4).

`utils.benchmark.model_installation.ModelInstaller(cache_dir).resolve(id)` also provides
the installation API; `resolve_plan(plan)` resolves only the plan's required
models. It uses registry metadata, streams HTTPS downloads into `.part` files,
restarts interrupted transfers from zero with bounded retries, and checks the
complete archive size and published MD5 before extraction. Verified archives are
retained under `archives/<record>/` for reuse. It extracts only the selected
dataset subtree (only model 802 from the combined original-model archive).

Managed models live under `models/<record>/<dataset>/`, separate from legacy
installations. Each contains a `.v3-installation.json` receipt with record/release
identity, artifact provenance, installation timestamp, and extracted-file SHA-256
hashes. Reuse validates structure, registry identity, and all recorded file hashes
without requiring a network connection. These receipts are local integrity
records, not publisher signatures. External spine models are explicitly reported
as **structural-only**, without claiming cryptographic verification.

Extraction rejects traversal, absolute paths, links/special files, duplicate and
case-colliding paths, Windows device/alternate-stream names, encryption, and
unsupported compression. Installations are validated in `staging/` and promoted
only when complete. Per-model exclusive locks under `locks/` serialize installation
of the same dataset, while separate artifact locks protect shared downloads.
Different lesion models can install concurrently within the same record.
A timed-out stale lock requires manual inspection after
confirming its process has stopped. Corrupt or incomplete installed directories
are preserved and rejected, never silently overwritten. Only invalid cached
archives/partial downloads for the requested artifact are removed. Keep the cache
user-owned and inaccessible to untrusted writers. Linked cache paths are rejected.
The installer does not deserialize checkpoints. Verified archives and complete
staged model directories (including validated receipts) are published with
atomic no-clobber renames: Windows rename, Linux `RENAME_NOREPLACE`, or
macOS `RENAME_EXCL`. Existing destinations, including empty directories created
concurrently, are never replaced. Unsupported native/filesystem operations fail
closed; no fallback uses an overwriting rename. Ordinary explicit ZIP directory
entries are accepted, including DOS/Unix attributes and directories listed after
their children. Unsafe entries in unselected models are still rejected.

Inputs must be readable, nonempty, finite, real-valued 3D NIfTI images. V3
converts voxel axes to RAS in memory by permutation and flipping, with affine
updates that retain physical voxel locations and any obliquity. Alignment is
checked on the resulting grids with a **0.001 mm** tolerance (maximum Euclidean distance
between corresponding voxel-centre corners). Spatial units must be declared as
mm, meter, or micron; unknown units are rejected rather than assumed. Conflicting
coded qform/sform transforms, singular affines, and NaN/infinity are rejected.
Axis reorientation uses no interpolation, registration, or resampling.

Supplied spine masks must have nonempty, finite, integer-valued labels and match
the required reconstruction grid. Probabilistic masks are rejected. All nonzero
labels become 1 in a separate uint8 binary mask; original labels and source files
remain unchanged. Supplied-mask model 710 still needs only VMI_40, without ConvCT.

`utils/benchmark/nifti.py` implements in-memory binary-mask preparation, a common
five-voxel padded bounding box (clipped to boundaries, exclusive Python slice
upper bounds), and ordered model-specific channel preparation. The binary spine
mask is always last. Cropped affines are translated by the crop start so that
physical voxel locations remain correct. Spatial units and compatible qform/sform
codes are retained; a qform that cannot represent shear compatibly is disabled
while the exact sform is retained. Explicit channel writing uses deterministic
`case_0001_0000.nii.gz`, etc., in a fresh directory. Reconstruction validates the
cropped binary lesion grid before insertion into the original reference grid;
explicit reconstruction writing validates a complete temporary `.nii.gz`, then
publishes via an atomic no-clobber rename. Existing destinations, including
concurrently created ones, remain untouched and failed temporary files are cleaned
up. Strict lesion labels remain 0/1;
the lesion adapter verifies that convention against each installed model's
metadata before inference. No external normalization or
morphology is applied. These utilities do not perform inference.

The reference is ConvCT when spine prediction is automatic. For lesion prediction
with a supplied spine mask it is the model's first required reconstruction
(VMI_40 for model 710). For spine-only mode with a supplied mask it is ConvCT if
present, otherwise the supplied mask. All generated masks are returned to the
reference's original voxel grid. The byte-for-byte copy of a supplied mask retains
its original grid. The manifest records each input's original and RAS geometry,
forward transforms and the reference inverse. Split and crop indices refer to
the internal reference RAS grid. `--validate_only` checks these grids without
producing predictions.
Published plans remain unchanged. Exact benchmark
training crop padding/affine handling remains unverified; five-voxel padding
preserves the existing pipeline's principle. Legacy V1/V2 utilities are unchanged.
Managed models from Zenodo records 18598645 (spine 802, `fold_all`) and 19708429
(lesions 700–718) will remain isolated from legacy installations.

Run all offline registry, preprocessing, mock-archive installation, and fake-predictor spine/lesion tests with
`python -B -m unittest discover -s tests -v`.
An optional source-level contract test checks signatures and filename mapping
against exactly nnU-Net 2.6.4 without importing its inference dependencies. It
uses an installed 2.6.4 distribution or a cached checkout specified by
`NNUNET_V264_SOURCE`; otherwise it skips cleanly. No test installs packages or
loads real checkpoints. Synthetic/fake-predictor tests validate orchestration and
output safety, not real-model segmentation performance or CUDA compatibility.

![Graphical Abstract](graphical_abstract.png "Overview of the dataset creation pipeline and data structure.")

## Overview
Welcome to the repository accompanying the published *Scientific Data* article **"Spinal-Multiple-Myeloma-SEG: Segmentation of spinal multiple myeloma lesions in dual-energy CT"**. This repository provides code for dataset visualization and segmentation workflows based on the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet).

### Published article

**Nohel et al. (2026).** *Spinal-Multiple-Myeloma-SEG: Segmentation of spinal multiple myeloma lesions in dual-energy CT.* **Scientific Data**. [Read the published article](https://doi.org/10.1038/s41597-026-08061-x).

**Public dataset:** [Spinal-Multiple-Myeloma-SEG on TCIA](https://doi.org/10.7937/k4qv-hh78). 

**Authors:** Michal Nohel<sup>1,†</sup>, Vlastimil Valek<sup>2,3,†</sup>, Tomas Rohan<sup>2,3</sup>, Martin Stork<sup>4</sup>, Roman Jakubicek<sup>1</sup>, Jiri Chmelik<sup>1,††,\*</sup> , Marek Dostal<sup>2,5,††,\*</sup>

**Author Affiliations:**  
<sup>1</sup> Department of Biomedical Engineering, Faculty of Electrical Engineering and Communication, Brno University of Technology, Brno, Czech Republic  
<sup>2</sup> Department of Radiology and Nuclear Medicine, University Hospital Brno, Brno, Czech Republic  
<sup>3</sup> Department of Radiology and Nuclear Medicine, Faculty of Medicine, Masaryk University, Brno, Czech Republic  
<sup>4</sup> Internal Hematology and Oncology Clinic, University Hospital Brno, Brno, Czech Republic  
<sup>5</sup> Department of Biophysics, Masaryk University, Brno, Czech Republic  

<sup>*</sup> Corresponding authors: Marek Dostal (Dostal.Marek@fnbrno.cz), Jiri Chmelik (chmelikj@vut.cz)  
<sup>†</sup> These authors contributed equally to this work  
<sup>††</sup> These authors contributed equally to this work



The repository contains two main scripts:  
1. A script for visualizing the dataset, allowing you to explore the spinal and myeloma images used.  
2. A script for performing segmentation predictions on the spine and myeloma regions. These predictions were utilized to create the myeloma database used in the study.  

Together, these scripts allow researchers to reproduce the dataset preparation and segmentation workflows described in the paper.

---

## Trained Models / Zenodo
Pre-trained nnU-Net models for this project are available on Zenodo. These models were trained for segmentation of spine, vertebrae, and osteolytic spinal multiple myeloma lesions in dual-energy CT data.

- The models were created as part of the dataset preparation and annotation pipeline for **Spinal-Multiple-Myeloma-SEG**, released via The Cancer Imaging Archive (TCIA) [https://doi.org/10.7937/k4qv-hh78](https://doi.org/10.7937/k4qv-hh78).
- Spine and vertebrae segmentation models were trained on **conventional CT data** from the VerSe2020 dataset, as well as on a combined VerSe2020 + multiple myeloma dataset. These masks were used to localize the spinal region and define spatial cropping for downstream lesion segmentation.
- Lesion segmentation models were trained exclusively on **VMI 40 keV images** using an iterative semi-automatic annotation workflow.
- All models operate on **NIfTI (.nii.gz)** inputs and are intended for **research and development purposes only**.

**Links:**
- Original dataset-preparation models: [Spinal-Multiple-Myeloma-SEG_nnUNet_models](https://doi.org/10.5281/zenodo.18598645)
- Zenodo snapshot of GitHub repo: [https://zenodo.org/records/19369118](https://zenodo.org/records/19369118)
- Zenodo DOI: [https://doi.org/10.5281/zenodo.15878952](https://doi.org/10.5281/zenodo.15878952)

A more detailed description of the models, training data, and processing pipeline is provided in the accompanying Zenodo README file.

---

## Usage
### Installation

The reproducible profile is tested on **Linux x86-64 with Python 3.12, an
NVIDIA CUDA 13.0-compatible driver and an RTX 5080**. It installs the three
prediction launchers and uses the exact nnU-Net v2.6.4 Git revision recorded in
`pyproject.toml`. Git and network access are needed for the initial dependency
installation. The nnU-Net source is installed as a Python dependency.

The tested profile uses Ubuntu 24.04 (Python 3.12.3), PyTorch 2.10.0 and
torchvision 0.25.0 (both CUDA 13.0 builds), and pydicom 3.0.2. The lockfile
pins the CUDA runtime libraries used by PyTorch.
pydicom 3.0.2 addresses [CVE-2026-32711](https://github.com/pydicom/pydicom/releases/tag/v3.0.2).
Use only trusted nnU-Net checkpoints.

The checked-in lock and pip export target this Linux profile. Windows, macOS,
other Python versions and alternative Torch builds require their own dependency
resolution and validation. Keep any existing legacy environments for those
configurations.

Real V3 inference has been tested with models 802 and 710. The other 18 lesion
models are covered by automated registry and input-channel contract tests.
V1/V2 retain their original implementation. The three launchers, DICOM metadata
selection, and geometry-preserving legacy NIfTI split/merge passed smoke tests.
Full V1/V2 inference and interactive GUI use remain outside that test coverage.

#### Recommended installation with uv

From the repository root, with Python 3.12 and uv available:

```bash
uv sync
uv run spinal-run-nnunet-TCIA --help
uv run spinal-run-nnunet-new-prediction --help
uv run spinal-run-nnunet-benchmark-prediction --help
```

Use `uv sync --locked` to require the checked-in resolution. Torch and
torchvision use the explicit official CUDA 13.0 index. The local environment is
`.venv`. For the optional DICOM viewer and its Qt dependencies:

```bash
uv sync --locked --extra gui
uv run --locked --extra gui spinal-db-viewer --help
```

An interactive viewer session also needs a working display and system Qt
libraries for the selected backend. Any external NIfTI viewer can overlay the
final V3 masks on the original reference image.

After installing the GUI extra, keep using `--extra gui` with `uv sync` and
`uv run`. A sync without the extra can remove its packages from the
environment. Running `.venv/bin/python` directly does not synchronize it.

#### Run the Python scripts directly

Run these commands from the repository root. Each script accepts the same
arguments as its corresponding launcher.

```bash
uv run python run_prediction_of_nnUNet_networks_on_TCIA_data_final.py --help
uv run python run_prediction_of_nnUNet_networks_on_new_data.py --help
uv run python run_prediction_of_nnUNet_benchmark_models.py --help
uv run --locked --extra gui python Database_viewer_final.py --help
```

To use the installed interpreter directly, replace `uv run python` (or
`uv run --locked --extra gui python`) with `.venv/bin/python`. If you keep
the GUI extra installed, include `--extra gui` on the prediction commands too.

In VS Code, open the repository folder and select `.venv/bin/python` via
**Python: Select Interpreter**. Open the script, set a breakpoint, then use
a Python Debugger configuration with the workspace root as its working
directory. Put each command-line argument in a separate entry in
`args`. Start with `["--help"]` or use the relevant prediction example with
your own input and output paths.

The bundled database viewer uses DICOM series and the legacy segmentation
layout. To inspect V3 results, open the original reference CT and the published
V3 masks together in a NIfTI viewer that respects image affines.
Connected components in a predicted lesion mask do not establish a clinically
confirmed lesion count. Technical regression checks measure reproducibility on
the tested inputs.

#### Alternative installation with pip

Use a fresh Python 3.12 environment from the repository root on Linux x86-64:

```bash
python3.12 -m venv .venv-pip
source .venv-pip/bin/activate
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
spinal-run-nnunet-TCIA --help
spinal-run-nnunet-new-prediction --help
spinal-run-nnunet-benchmark-prediction --help
```

The final install registers the launchers and the installed project reads the
same dependencies from `pyproject.toml`. For the optional viewer in the pip
environment, run
`python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -c requirements.txt -e '.[gui]'`.

`pyproject.toml` defines the dependencies, `uv.lock` records their resolution,
and `requirements.txt` is a generated export of the base environment. To
refresh the export after an intentional lock update:

```bash
uv export --locked --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

#### V3 quick start

For model 710, pass the original ConvCT and VMI40 files directly or use a
per-case input folder containing `ConvCT.nii.gz` and `VMI_40.nii.gz`. V3
reorients each required image internally and publishes masks in the original
reference grid.
Choose model IDs 700–718 according to the reconstructions listed in the
[V3 model table](#v3-benchmark-prediction--phase-4b-end-to-end-spine-and-lesions).

```bash
# Automatic spine (802) and lesions (710), authorizing model installation if missing.
uv run spinal-run-nnunet-benchmark-prediction \
  --path_to_convCT_nifti /data/original/Myel_001_conv.nii.gz \
  --path_to_VMI40_nifti /data/original/Myel_001_monoe_40kev.nii.gz \
  --case_id Myel_001 --output_dir /data/results \
  --lesion_model 710 --split True

# Spine only, reusing an installed model.
uv run spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/case --output_dir /data/results --skip_lesion_segmentation

# Lesions from VMI_40 and an existing spine mask.
uv run spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/case --output_dir /data/results \
  --lesion_model 710 --spine_segmentation /data/masks/spine.nii.gz
```

Use `--cache_dir /data/models` to share a V3 model cache across runs. Add
`--offline` to reuse verified installed models without a download. Add
`--validate_only` to check inputs and alignment without prediction. Successful
validation-only execution exits with code 3. Each successful prediction publishes
a new case output directory containing the final masks and `run.json`.

## Path Configuration
V1, V2 and the viewer accept explicit CLI paths or defaults from `config.py`. The installed nnU-Net dependency supplies imports even when the legacy `NNUNET_REPO_PATH` points to a missing checkout. Point the legacy prediction commands at the appropriate separately installed legacy model directory. V3 uses its command-line paths and model cache.

---

Running the Pipeline / Inference
===============================

This repository provides two complementary ways of working with the
Spinal-Multiple-Myeloma-SEG data:

1. Data visualization and qualitative inspection using Python scripts (recommended first step)
2. Full nnU-Net–based segmentation pipeline for automatic inference
3. Optional command-line inference using the default nnU-Net CLI


---------------------------------------------------------------------
Data Visualization and Inspection
---------------------------------------------------------------------

Visualization Script
--------------------
**`Database_viewer_final.py`**

This script is intended for qualitative inspection of the dataset and segmentation
results. It loads multiple CT reconstructions from DICOM data, overlays spine and
lesion segmentation masks, and visualizes everything using **Napari**.

The script can be executed both:
- directly from the command line, and
- interactively from an IDE such as VS Code.


Loaded Data
-----------
For a selected patient, the script automatically detects and loads the following
DICOM series based on the SeriesDescription metadata:

- Conventional CT (ConvCT)
- Virtual Monoenergetic Images (VMI):
  - 40 keV
  - 80 keV
  - 120 keV
- Calcium Suppression reconstructions:
  - 25 Index
  - 50 Index
  - 75 Index
  - 100 Index

In addition, the script loads segmentation masks in **NIfTI (.nii.gz)** format:

- Spine segmentation mask
- Osteolytic lesion segmentation mask


Visualization Features
----------------------
All volumes and segmentation masks are displayed together in a single Napari viewer:

- CT volumes are shown as grayscale image layers
- Spine segmentation is overlaid in blue
- Lesion segmentation is overlaid in red
- Individual layers can be toggled on/off for interactive comparison
- Opacity and visibility can be adjusted directly in Napari

This setup enables detailed visual comparison between different energy reconstructions
and segmentation outputs.


Running the Visualization Script from the Command Line
-----------------------------------------------------
The visualization tool can be executed from the command line.  
You can either configure paths in the `config.py` file or provide them directly as arguments.

#### Option 1 – Use `config.py` (recommended)

If all paths and the patient ID are correctly set in `config.py`, the viewer can be launched without any arguments:

```bash
spinal-db-viewer
```

Before running, make sure to edit the following variables in `config.py`:
- `PATH_TO_DICOM_FOLDERS`
- `PATH_TO_SEGMENTATIONS`
- `ID_PATIENT`
- `PATIENT_SERIES_TAG` (leave blank by default; set only for multi-series patients like `Myel_012_a` / `Myel_012_b`)
- `NNUNET_REPO_PATH` (required for nnU-Net integration)

This is the simplest and recommended setup for regular use.

#### Option 2 – Installed CLI Tool (manual paths)

```bash
spinal-db-viewer \
  --path_to_DICOM_folders "/path/to/Spinal-Multiple-Myeloma-SEG" \
  --path_to_segmentations "/path/to/MM_NIfTI_Segmentation" \
  --ID_patient "Myel_001" \
  --series_tag ""
```

#### Option 3 – Run as Python Script

```bash
python Database_viewer_final.py \
  --path_to_DICOM_folders "/path/to/Spinal-Multiple-Myeloma-SEG" \
  --path_to_segmentations "/path/to/MM_NIfTI_Segmentation" \
  --ID_patient "Myel_001" \
  --series_tag ""
```

All three options are equivalent.
If arguments are not provided, the values defined in `config.py` are used as defaults.

#### Arguments

- `--path_to_DICOM_folders`  
  Path to the DICOM folders, organized by patient ID and then by series description.

- `--path_to_segmentations`  
  Path to the NIfTI segmentation masks, organized by patient ID and mask type (`spine` or `lesions`).

- `--ID_patient`  
  Patient identifier (e.g., `Myel_001`).

- `--series_tag`  
  Optional series variant tag for multi-acquisition patients. Use `a` or `b` for names like `Myel_012_a` / `Myel_012_b`. Leave blank for standard single-series patients.

---------------------------------------------------------------------
Full Segmentation Pipeline (Python)
---------------------------------------------------------------------
---------------------------------------------------------------------
Segmentation Pipelines
---------------------------------------------------------------------

This repository provides two segmentation modes based on the input data format.
Both pipelines use the same trained nnU-Net models but differ in preprocessing.

- **TCIA Mode (DICOM input)** – full pipeline from raw DICOM data  
- **New Prediction Mode (NIfTI input)** – direct inference on prepared NIfTI files  


## 1. TCIA Mode (DICOM → Full Pipeline)


**Script:** `run_prediction_of_nnUNet_networks_on_TCIA_data_final.py`  
**CLI command:** `spinal-run-nnunet-TCIA`

This script runs the complete nnU-Net–based segmentation pipeline for a single patient,
starting directly from DICOM data and producing final lesion segmentations in the
original image space.

The pipeline always starts from a clean working directory to ensure reproducibility.
Any existing intermediate results for the selected patient are removed before processing.


Pipeline Steps
--------------
1. Conversion of ConvCT and VMI 40 keV DICOM series to NIfTI format
2. Spine segmentation from ConvCT data using nnU-Net
3. Reorientation of the spine segmentation back to the original image space
4. Lesion segmentation from VMI 40 keV images
5. Final reconstruction of lesion segmentation in the original image space

### Running

The segmentation pipeline can be executed from the command line.  
You can either configure paths in the `config.py` file or provide them directly as arguments.
You can run the pipeline in three equivalent ways:

#### Option 1 – Use `config.py` (recommended)

If all paths and settings are correctly defined in `config.py`, the pipeline can be launched without any arguments:

```bash
spinal-run-nnunet-TCIA
```

Before running, make sure to set the following variables in `config.py`:
- `PATH_TO_DICOM_FOLDERS`
- `PATH_TO_NNUNET_RESULTS`
- `ID_PATIENT`
- `PATIENT_SERIES_TAG` (leave blank by default; set only for multi-series patients, e.g. `a` or `b` for `Myel_012_a` / `Myel_012_b`)
- `NNUNET_REPO_PATH` (required for nnU-Net integration)
- `SPLIT_CONVCT_DEFAULT` (default split behavior)

This is the simplest and recommended setup for regular use.

#### Option 2 – Installed CLI Tool

```bash
spinal-run-nnunet-TCIA \
  --path_to_DICOM_folders "/path/to/Spinal-Multiple-Myeloma-SEG" \
  --path_to_nnunet_results "/path/to/nnUNet_trained_models" \
  --ID_patient "Myel_001" \
  --split True \
  --series_tag ""
```

#### Option 3 – Run as Python Script

```bash
python run_prediction_of_nnUNet_networks_on_TCIA_data_final.py \
  --path_to_DICOM_folders "/path/to/Spinal-Multiple-Myeloma-SEG" \
  --path_to_nnunet_results "/path/to/nnUNet_trained_models" \
  --ID_patient "Myel_001" \
  --split True \
  --series_tag ""
```
All three options are equivalent. 
If arguments are not provided, the values defined in `config.py` are used as defaults.

#### Arguments

- `--path_to_DICOM_folders`  
  Path to the DICOM folders organized by patient ID and series description.

- `--path_to_nnunet_results`  
  Path to the folder containing trained nnU-Net models.

- `--ID_patient`  
  Patient identifier (e.g., `S840`).

- `--series_tag`  
  Optional series variant tag for patients with multiple acquisition series. Use `a` or `b` for names like `Myel_012_a` / `Myel_012_b`. Leave blank for normal cases.

- `--split`  
  If `True`, the data are split along the Z-axis to reduce memory requirements.


## 2. New Prediction Mode (NIfTI input)


**CLI command:** `spinal-run-nnunet-new-prediction`

This mode is intended for already preprocessed data:
- Input must be NIfTI (.nii.gz)
- Data must be in RAS orientation
- No DICOM conversion is performed

### Running

#### Option 1 – Use `config.py` (recommended)

```bash
spinal-run-nnunet-new-prediction
```

Make sure relevant paths are defined in `config.py`.

#### Option 2 – Installed CLI Tool

```bash
spinal-run-nnunet-new-prediction \
  --path_to_convCT_nifti "/path/to/convCT.nii.gz" \
  --path_to_VMI40_nifti "/path/to/VMI40.nii.gz" \
  --path_to_output_folder "/path/to/output" \
  --path_to_nnunet_results "/path/to/nnUNet_trained_models" \
  --split True
```
#### Option 3 – Run as Python Script

```bash
python run_prediction_on_new_data.py \
  --path_to_convCT_nifti "/path/to/convCT.nii.gz" \
  --path_to_VMI40_nifti "/path/to/VMI40.nii.gz" \
  --path_to_output_folder "/path/to/output" \
  --path_to_nnunet_results "/path/to/nnUNet_trained_models" \
  --split True
```
---

#### Arguments 
- `--path_to_convCT_nifti`  
  Path to conventional CT in NIfTI format (new prediction mode)

- `--path_to_VMI40_nifti`  
  Path to VMI 40 keV NIfTI image

- `--path_to_output_folder`  
  Output directory for predictions

- `--path_to_nnunet_results`  
  Path to trained nnU-Net models

- `--split`  
  If `True`, splits data along the Z-axis to reduce memory usage

---

Notes on Multiprocessing
------------------------
- By default, the pipeline is configured for **Linux** and may use multiprocessing during nnU-Net inference.
- On **Windows**, Python multiprocessing can occasionally fail when running from a clean session.
- If inference fails on Windows, open `utils/legacy.py` and modify the function `run_nnunet_inference` to use **variant 2** (`predict_from_files_sequential`), which disables multiprocessing.
- This variant is slower but ensures safe and stable execution on Windows systems.
---

### Command-Line Inference Using nnUNet
The trained models can also be used with the default nnU-Net CLI for folder-based inference. For inference you can use the default [nnUNet inference functionalities](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md). 

To run a single model on all input images:

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER -f all
```

## Citation

If you use this repository or the associated dataset, please cite the published *Scientific Data* article and the original TCIA dataset.

**Scientific Data article:**

Nohel, M., Valek, V., Rohan, T., Stork, M., Jakubicek, R., Chmelik, J., & Dostal, M. (2026). **Spinal-Multiple-Myeloma-SEG: Segmentation of spinal multiple myeloma lesions in dual-energy CT.** *Scientific Data*. https://doi.org/10.1038/s41597-026-08061-x

```bibtex
@article{nohel2026spinalmm,
  author  = {Nohel, Michal and Valek, Vlastimil and Rohan, Tomas and Stork, Martin and Jakubicek, Roman and Chmelik, Jiri and Dostal, Marek},
  title   = {{Spinal-Multiple-Myeloma-SEG}: Segmentation of spinal multiple myeloma lesions in dual-energy {CT}},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-08061-x},
  url     = {https://doi.org/10.1038/s41597-026-08061-x}
}
```

**Original TCIA dataset:**

Nohel, M., Válek V., Rohan T., Štork M., Jakubíček R., Chmelík J., Dostál M. (2026) **Segmentation of spinal multiple myeloma lesions in dual-energy CT (Spinal-Multiple-Myeloma-SEG) (Version 1) [Dataset]**. The Cancer Imaging Archive. https://doi.org/10.7937/k4qv-hh78

```bibtex
@misc{Nohel_Valek_Rohan_Stork_Jakubicek_Chmelik_Dostál_2026,
author={Nohel, Michal and Valek, Vlastimil and Rohan, Tomas and Stork, Martin and Jakubicek, Roman and Chmelik, Jiri and Dostál, Marek},
title={Segmentation of spinal multiple myeloma lesions in dual-energy CT (Spinal-Multiple-Myeloma-SEG)},
url={https://www.cancerimagingarchive.net/collection/Spinal-Multiple-Myeloma-SEG},
DOI={10.7937/K4QV-HH78},
publisher={The Cancer Imaging Archive},
year={2026},
abstractNote={This dataset consists of dual-energy low-dose CT scans from 67 patients diagnosed with multiple myeloma, with a total of 72 scans. The dataset includes conventional CT images, virtual monoenergetic images (at 40, 80, and 120 keV), calcium-suppressed images (with suppression indices of 25, 50, 75, and 100), as well as segmentation masks of vertebrae (including vertebra type classification) and multiple myeloma lesions in the spine. In total, the dataset contains 576 image series comprising 564,464 axial slices.  In addition to image data, the dataset provides supporting non-image information, including basic demographic details of patients (mean age 66 years, range 48–85; 36% female) and their clinical data related to disease progression.  This dataset holds substantial promise for advancing and objectively evaluating Computer-Aided Detection and Diagnostic systems, particularly those based on machine learning and artificial intelligence. } }
```





