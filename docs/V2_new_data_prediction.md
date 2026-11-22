# V2 · New data segmentation

[Back to main README](../README.md)

[Pipeline](#pipeline) · [Inputs](#input-data) · [Running](#running-the-pipeline) · [Arguments](#command-line-arguments) · [Outputs](#output-structure)

V2 applies the original dataset-preparation models to a ConvCT and VMI 40 keV pair. It starts from NIfTI and shares the original spine-guided lesion prediction with [V1](V1_TCIA_prediction.md). Use [V3](V3_benchmark_prediction.md) for automatic orientation handling, selectable spectral models or prepared lesion channels.

## Pipeline

1. Copy both inputs into a case working directory and assign nnU-Net filenames.
2. Use model 802 on ConvCT. If splitting is enabled, split the volume along Z, predict each half and merge the results.
3. Convert spine labels to a binary mask, calculate the bounding box with five voxels of padding and crop the VMI image and spine mask.
4. Apply the original 710 model to the cropped pair and restore its prediction to the full VMI 40 keV grid.

## Input data

Prepare aligned ConvCT and VMI 40 keV NIfTI images from the same examination. **Supply both images in RAS orientation on a common voxel grid.** Check shape, voxel spacing, affine and spatial alignment before running. V2 copies the input volumes and gives ConvCT a RAS filename. It performs no reorientation, registration or resampling to align the images.

Use compressed `.nii.gz` inputs. Although help advertises `.nii` as well, the implementation uses a byte-for-byte copy into a `.nii.gz` working filename. Explicitly convert uncompressed inputs to valid gzip-compressed NIfTI before use. Use a ConvCT basename such as `Myel_001_conv.nii.gz`. The output case name comes from the first two underscore-separated fields, or from the whole stem if it has only one field.

## Trained models and configuration

Use the [original Zenodo models](Model_zoo.md#original-segmentation-models), with `fold_all` for both checkpoints. The results root is the parent containing both dataset directories. The V3 lesion model 710 has the same dataset name and a different fold layout.

Defaults are read from [config.py](../config.py). Set the input, output and model paths there for no-argument use. CLI values take precedence. The configured nnU-Net environment values are set at startup, while the supplied results root determines the checkpoint location.

## Running the pipeline

```bash
uv run --locked spinal-run-nnunet-new-prediction \
  --path_to_convCT_nifti /path/to/Myel_001_conv.nii.gz \
  --path_to_VMI40_nifti /path/to/Myel_001_vmi40.nii.gz \
  --path_to_output_folder /path/to/new-output \
  --path_to_nnunet_results /path/to/original-models \
  --split True
```

To run directly, replace the launcher expression with `.venv/bin/python run_prediction_of_nnUNet_networks_on_new_data.py` or `uv run --locked python run_prediction_of_nnUNet_networks_on_new_data.py`, retaining the arguments. In VS Code, open [the V2 script](../run_prediction_of_nnUNet_networks_on_new_data.py) after selecting the project interpreter. Use verified config defaults or provide the same arguments in the integrated terminal or debugger. If GUI is installed, include `--extra gui` on synchronizing uv commands.

## Command-line arguments

| Argument | Value | Default and behavior |
|---|---|---|
| `--path_to_convCT_nifti` | ConvCT path | `config.PATH_TO_CONVCT_NIFTI`. RAS working reference for spine inference and source of the case name. |
| `--path_to_VMI40_nifti` | VMI 40 keV path | `config.PATH_TO_VMI40_NIFTI`. Same aligned RAS grid as ConvCT. |
| `--path_to_output_folder` | Directory | `config.PATH_TO_OUTPUT_FOLDER`. Parent of the case directory. |
| `--path_to_nnunet_results` | Directory | `config.PATH_TO_NNUNET_RESULTS`. Extracted original model root. |
| `--split` | `True` or `False` | `config.SPLIT_CONVCT_DEFAULT`, initially `True`. Controls spine splitting. |
| `-h`, `--help` | Flag | Display help and exit. |

## Output structure

> [!WARNING]
> V2 deletes an existing `<case>_output` directory before processing the case. Supply a fresh output parent if you need to keep prior results.

```text
output/<case>_output/
├── Conv_CT/<case>_conv_RAS_0000.nii.gz
├── Conv_CT_cropped/                 # when split
├── VMI40/<case>_monoe_40kev_0000.nii.gz
├── VMI40_cropped/                   # cropped image and binary spine channel
└── Segmentation/
    ├── Spine_segmentation_cropped/
    ├── Spine_segmentation_final/<case>_spine_segmentation.nii.gz
    ├── crop_parameters_folder/<case>.json
    ├── Lesion_segmentation_cropped/
    └── Lesion_segmentation_final/<case>_lesions_segmentation.nii.gz
```

Review overlay alignment before interpreting a result. If inference fails to read a working file, verify that both source images are compressed NIfTI. Splitting reduces the size of each spine prediction. The full image and restoration can still use substantial host memory. For absent weights or startup failures, see [installation troubleshooting](Installation.md#troubleshooting).

[Installation](Installation.md) · [TCIA DICOM workflow](V1_TCIA_prediction.md) · [Benchmark workflow](V3_benchmark_prediction.md) · [Model Zoo](Model_zoo.md)
