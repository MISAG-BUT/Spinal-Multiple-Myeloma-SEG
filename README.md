# Spinal-Multiple-Myeloma-SEG

## V3 benchmark prediction — Phase 4B (end-to-end spine and lesions)

V3 is independent of the existing V1 TCIA/DICOM and V2 NIfTI workflows.
**Spine-only and end-to-end benchmark lesion execution are implemented.**
This research preview has not been validated with real-model GPU inference.
`--skip_lesion_segmentation` executes the spine workflow and exits **0** only
after publishing a validated completed run. `--lesion_model 700–718` executes
spine preparation and the selected five-fold lesion model, also exiting **0**
only after the entire run has been published.
Invalid input validation exits **2**, model resolution/installation errors exit
**4**, and inference/reconstruction/publication errors exit **5**. No download is attempted
unless `--install_models` is supplied. Add `--validate_only` to validate and print
the plan without installation, inference, or output creation (exit **3**).

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

Arguments: required `--input_dir` and `--output_dir`; exactly one of
`--lesion_model 700–718` or `--skip_lesion_segmentation`; optional
`--spine_segmentation`, `--spine_model_dir`, `--cache_dir`, `--split`,
`--install_models`, and `--validate_only`. Validation-only and installation flags
cannot be combined.
`--spine_model_dir` must name an existing directory and conflicts with
`--spine_segmentation`. With model resolution requested, it must point to the
model-802 `nnUNetTrainer__nnUNetPlans__3d_fullres` directory, whose metadata and
`fold_all/checkpoint_final.pth` are structurally validated. This does not establish
the external checkpoint's cryptographic identity or actual training version.
Explicit `--split` also conflicts with a supplied segmentation. Otherwise its
default is True, with V2 parsing: case-insensitive `true` enables splitting and
every other supplied value disables it. The default cache is under LOCALAPPDATA
on Windows, `~/Library/Caches` on macOS, or XDG_CACHE_HOME/`~/.cache` on Linux.
`--cache_dir` overrides it. Validation-only runs do not create a cache.

### Explicit model installation

Add **`--install_models`** to opt into model resolution after NIfTI validation:

```bash
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --lesion_model 710 --install_models --cache_dir /data/v3-cache
```

This may download approximately 3.68 GB for automatic spine model 802 and 1.66 GB
for the selected lesion model. A supplied spine segmentation bypasses model 802
entirely; spine-only mode downloads no lesion model. Installation failure exits
with code **4**. Without the flag, missing models fail with installation guidance
and no HTTP requests. Before any predictor is constructed, all required models
are resolved and the selected lesion model's metadata is validated. Automatic
mode resolves 802 plus the selected lesion model; supplied-mask mode resolves
only the lesion model. A missing lesion model therefore cannot be discovered
only after expensive spine inference. Existing patient outputs are never overwritten.

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

`benchmark_lesion_inference.py` reuses the Phase 4A spine adapter or the supplied
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

Complete lesion runs have this structure:

```text
<output_dir>/run-<unique-id>/
  run.json
  spine/
    spine_labelled.nii.gz
    spine_binary.nii.gz
    spine_supplied.nii.gz     # supplied mode only, byte-identical source copy
  lesion/
    lesion_labelled.nii.gz
```

The entire run stays in a unique private staging directory until both stages,
reconstruction, validated NIfTI writing, and the manifest complete. A failed
spine/lesion prediction, reconstruction, or publication exposes no completed run;
owned temporary channels and prediction files are cleaned. Existing results are
preserved. Completed runs contain no cropped channels or nnU-Net working files.
Phase 4B manifests use schema version 2 and record lesion ID/name, trainer/plans/
configuration, folds/checkpoint, model verification/artifact provenance, spine
source and model-802 provenance, requested/effective split, input modalities,
ordered channels, exclusive crop bounds/padding, reference/cropped/output
geometries, relative output paths, completion time/status, and lesion completion.
No absolute patient paths or voxel data are included. Spine-only manifests retain
the Phase 4A schema version 1 and existing output contract.

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
not broadly relabelled as ordinary prediction failures. Offline validation and
source agreement do not establish exact scientific reproduction or real CUDA
inference performance.

### Spine-only execution and outputs

```bash
# Use an already installed managed model; no automatic download:
spinal-run-nnunet-benchmark-prediction \
  --input_dir /data/patient_001 --output_dir /data/results \
  --skip_lesion_segmentation --split True --cache_dir /data/v3-cache

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
`273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0`; no real checkpoints or CUDA execution
were used in that review. No global path or
nnU-Net environment configuration is changed. V3 uses the supported sequential
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
`original_affine @ voxel_translation(0, 0, z_start)`; V2's second-half origin
adjustment used only the world-Z component. No external normalization,
downsampling, reorientation, or image-space resampling is introduced.

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

Completed runs are published atomically in unique directories:

```text
<output_dir>/run-<unique-id>/
  run.json
  spine/
    spine_labelled.nii.gz
    spine_binary.nii.gz
    spine_supplied.nii.gz    # supplied mode only, byte-identical source copy
```

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

`benchmark_model_installation.ModelInstaller(cache_dir).resolve(id)` also provides
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
Archive publication requires filesystem hard-link support and fails safely if
unavailable. No checkpoints are deserialized during installation.
Complete staged model directories (including validated receipts) are published
with atomic no-clobber renames: Windows rename, Linux `RENAME_NOREPLACE`, or
macOS `RENAME_EXCL`. Existing destinations, including empty directories created
concurrently, are never replaced. Unsupported native/filesystem operations fail
closed; no fallback uses an overwriting rename. Ordinary explicit ZIP directory
entries are accepted, including DOS/Unix attributes and directories listed after
their children. Unsafe entries in unselected models are still rejected.

Inputs must be readable, nonempty, finite, real-valued 3D NIfTI images in RAS
orientation. Dimensions and orientation must match; spacing and physical voxel
alignment are checked with a **0.001 mm** tolerance (maximum Euclidean distance
between corresponding voxel-centre corners). Spatial units must be declared as
mm, meter, or micron; unknown units are rejected rather than assumed. Conflicting
coded qform/sform transforms, singular affines, and NaN/infinity are rejected.
No automatic reorientation, registration, or resampling is performed.

Supplied spine masks must have nonempty, finite, integer-valued labels and match
the required reconstruction grid. Probabilistic masks are rejected. All nonzero
labels become 1 in a separate uint8 binary mask; original labels and source files
remain unchanged. Supplied-mask model 710 still needs only VMI_40, without ConvCT.

`benchmark_nifti.py` implements in-memory binary-mask preparation, a common
five-voxel padded bounding box (clipped to boundaries, exclusive Python slice
upper bounds), and ordered model-specific channel preparation. The binary spine
mask is always last. Cropped affines are translated by the crop start so that
physical voxel locations remain correct. Spatial units and compatible qform/sform
codes are retained; a qform that cannot represent shear compatibly is disabled
while the exact sform is retained. Explicit channel writing uses deterministic
`case_0001_0000.nii.gz`, etc., in a fresh directory. Reconstruction validates the
cropped binary lesion grid before insertion into the original reference grid;
explicit reconstruction writing validates a complete temporary `.nii.gz`, then
publishes via an atomic no-clobber hard link. Existing destinations, including
concurrently created ones, remain untouched; failed temporary files are cleaned
up. This requires filesystem hard-link support. Strict lesion labels remain 0/1;
the lesion adapter verifies that convention against each installed model's
metadata before inference. No external normalization or
morphology is applied. These utilities do not perform inference.

For lesion preprocessing, reference selection is ConvCT for automatic spine mode, the first required
reconstruction for supplied-mask lesion mode, or the mask for supplied-mask
spine-only mode. `--validate_only` checks these grids without producing predictions.
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

#### Environment Setup for nnUNet Prediction (Python 3.13)
This guide explains how to set up the environment for running predictions with [nnUNet](https://github.com/MIC-DKFZ/nnUNet) on the [Spinal-Multiple-Myeloma-SEG](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG) project.
Check out the official [nnUNet installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

**Note:** This pipeline has been tested on both Linux and Windows systems with GPUs:  

- **Linux:**  
  - GPU: Nvidia Titan Xp, 12 GB GDDR5  
  - CPU: Intel Core i9-12900KF, 8+8/24 cores/threads, 2.4/3.2 GHz  
  - RAM: 64 GB DDR5  
  - Storage: 1 TB SSD (system), 4 TB HDD RAID5 (data)  
  - OS: Ubuntu 24.04  

- **Windows:**  
  - GPU: EVGA GeForce RTX 3090, 24 GB GDDR6  
  - CPU: Intel Core i9-10900KF, 10/20 cores/threads, 3.7 GHz  
  - RAM: 64 GB  
  - Storage: 2 TB SSD M.2 (system)  
  - OS: Windows 10  

> ⚠ **Important:** The script defaults to Linux-style multiprocessing. On a clean Windows setup, nnU-Net batch prediction may fail due to multiprocessing issues.  
> If segmentation fails, use **Variant 2** in the `run_nnunet_inference` function in `utils.py`, which runs predictions sequentially without multiprocessing. This is slower but safe on Windows.

---
#### Step 1: Install Python and Clone Repository
First, download and install Python 3.13.x from [python.org](https://www.python.org/downloads/). We used Python 3.13.5.  
During installation, make sure to check **"Add Python to PATH"**.  


Create a virtual environment inside the project folder. Replace <path-to-your-python-executable> with the path to Python 3.13 on your system:
```bash
<path-to-your-python-executable> -m venv venv_run_nnUNet_prediction
```

Examples:
- Windows
```bash
C:\Users\<username>\AppData\Local\Programs\Python\Python313\python.exe -m venv venv_run_nnUNet_prediction
```
- Linux/macOS
```bash
python3.13 -m venv venv_run_nnUNet_prediction
```

Activate the environment:
- Windows
```bash
venv_run_nnUNet_prediction\Scripts\activate
```
- Linux/macOS
```bash
source venv_run_nnUNet_prediction/bin/activate
```

When activated, your prompt should show (venv_run_nnUNet_prediction).

#### Step 2: Install Core Dependencies
Install required packages starting with PyTorch. For CUDA 12.6 run:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Clone the nnUNet repository, move into it, and install in editable mode:
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
cd ..
```

Clone the project repository and move into the project folder:

```bash
git clone https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG.git
cd Spinal-Multiple-Myeloma-SEG
pip install -e .
cd ..
```

#### Step 3: Additional / Optional Packages (napari / Qt)

The GUI viewer relies on `napari`, which requires a working Qt backend.  
Depending on your operating system, additional system libraries may be needed.

> 💡 **Tip:** If `napari` fails to start (Qt errors, missing plugins, or XCB issues), install the OS-specific dependencies below.

##### Linux (tested on Ubuntu)

On Linux, you may need extra Qt/XCB system libraries:

```bash
sudo apt install \
  libxcb-xinerama0 \
  libxkbcommon-x11-0 \
  libxcb-cursor0 \
  libxcb-icccm4 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-xfixes0 \
  libxcb-shape0 \
  libxcb-sync1 \
  libxcb-xkb1
```

Then install the Python packages:

```bash
# Napari viewer for GUI visualization
pip install napari

# Full napari installation with optional plugins (recommended)
pip install -U 'napari[all]'
```

##### Windows

On Windows, installing the full napari package with Qt dependencies is usually sufficient:

```bash
pip install -U 'napari[all]'
```

If you experience GUI issues, make sure your environment uses a compatible Qt backend (PyQt5 is installed automatically via `napari[all]`).

---

## Full Setup Summary
For convenience, a full setup summary:
```bash
<path-to-your-python-executable> -m venv venv_run_nnUNet_prediction
# Activate the environment
# Windows:
venv_run_nnUNet_prediction\Scripts\activate
# Linux/macOS:
source venv_run_nnUNet_prediction/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
cd ..
git clone https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG.git
cd Spinal-Multiple-Myeloma-SEG
pip install -e .
cd ..

# Optional / additional packages
pip install pydicom napari
pip install -U 'napari[all]'
```

Always replace <path-to-your-python-executable> with the correct path on your system. For different CUDA versions or CPU-only setups, check the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## Path Configuration
Before running the project, you need to edit the `config.py` file and set the correct paths to your data and output directories according to your environment.

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
- If inference fails on Windows, open `utils.py` and modify the function `run_nnunet_inference` to use **variant 2** (`predict_from_files_sequential`), which disables multiprocessing.
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





