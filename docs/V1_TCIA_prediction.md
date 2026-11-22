# V1 · TCIA dataset segmentation

[Back to main README](../README.md)

[Pipeline](#pipeline) · [Inputs](#input-data) · [Running](#running-the-pipeline) · [Arguments](#command-line-arguments) · [Outputs](#output-structure)

V1 segments the spine and osteolytic multiple myeloma lesions in a patient from the [TCIA collection](https://doi.org/10.7937/k4qv-hh78). It uses the original dataset-preparation models. For aligned NIfTI data and the same original models, see [V2](V2_new_data_prediction.md). For the spectral benchmark model collection, see [V3](V3_benchmark_prediction.md).

## Pipeline

1. Select a patient and acquisition group, then identify conventional CT and VMI 40 keV series.
2. Convert both series to NIfTI using SimpleITK. Retain the original images and make a RAS working copy of ConvCT.
3. Predict anatomical spine labels with model 802. With splitting enabled, divide the RAS volume along Z, infer on each half and merge the masks. Return the spine segmentation to the original ConvCT orientation.
4. Binarize the spine labels. Use the mask to obtain a bounding box, pad it by five voxels and crop both the mask and original VMI 40 keV image.
5. Infer lesions with the original model 710, then place the cropped prediction in the full VMI image grid.

## Input data

Download the TCIA image data and pass the **parent of the patient directories**:

```text
dataset/
└── Myel_001/
    └── study-or-acquisition/
        ├── conventional-series/    # DICOM slices
        └── vmi40-series/           # DICOM slices
```

The series finder searches below the patient folder. The first DICOM file in each series is used for metadata. Series are grouped by `PatientName`, falling back to `PatientID`. If multiple groups are found, pass `--series_tag a` or the appropriate suffix to select a group such as `Myel_012_a`. The ID still identifies the patient folder. ConvCT must have a `SeriesDescription` ending in `_konv`, and VMI 40 keV must have exactly `MonoE 40keV[HU]`. The prefix before `_konv` supplies the output case name. The two images should describe the same examination on aligned grids for index-based cropping. This workflow does not register the images.

## Trained models and configuration

Extract the [original Zenodo model archive](https://doi.org/10.5281/zenodo.18598645). Set `PATH_TO_NNUNET_RESULTS` to the parent containing the model dataset folders, or pass `--path_to_nnunet_results`. Both original models use `nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth`. See the [original model layout](Model_zoo.md#original-segmentation-models). V3's five-fold lesion package does not provide the original `fold_all` checkpoint.

Set local defaults in [config.py](../config.py) to run without arguments. Explicit CLI values override these defaults. The installed nnU-Net package supplies the code. The legacy script also adds the configured `NNUNET_REPO_PATH` to its import search path and sets `NNUNET_RAW`, `NNUNET_PREPROCESSED` and `NNUNET_RESULTS_ENV` in the nnU-Net environment at startup. Model weights come from the explicitly selected results root.

## Running the pipeline

After [installation](Installation.md), run from the repository root. Choose a fresh case destination. The output parent is derived from the dataset path.

```bash
uv run --locked spinal-run-nnunet-TCIA \
  --path_to_DICOM_folders /path/to/dataset \
  --ID_patient Myel_001 \
  --path_to_nnunet_results /path/to/original-models \
  --split True
```

For direct Python execution, replace the command on the first line with either of these and keep the arguments:

```bash
.venv/bin/python run_prediction_of_nnUNet_networks_on_TCIA_data_final.py --help
uv run --locked python run_prediction_of_nnUNet_networks_on_TCIA_data_final.py --help
```

The commands above display help. Remove `--help` and add the prediction arguments when running a case. In VS Code, select the project interpreter and open [the V1 script](../run_prediction_of_nnUNet_networks_on_TCIA_data_final.py). Run with verified `config.py` defaults or pass the same CLI arguments through the integrated terminal or debugger. After installing GUI, retain `--extra gui` on synchronizing uv commands.

## Command-line arguments

| Argument | Value | Default and behavior |
|---|---|---|
| `--path_to_DICOM_folders` | Directory | `config.PATH_TO_DICOM_FOLDERS`. Parent of patient folders. The output root is formed by appending `_output` to this argument, so omit a trailing separator. |
| `--ID_patient` | Folder name | `config.ID_PATIENT`, initially `Myel_001`. |
| `--series_tag` | Acquisition suffix | `config.PATIENT_SERIES_TAG`, initially empty. Required when multiple groups exist. |
| `--path_to_nnunet_results` | Directory | `config.PATH_TO_NNUNET_RESULTS`. Parent of extracted model dataset folders. |
| `--split` | `True` or `False` | `config.SPLIT_CONVCT_DEFAULT`, initially `True`. Only the spine input is split. |
| `-h`, `--help` | Flag | Display help and exit. |

## Output structure

> [!WARNING]
> V1 deletes an existing case working directory before writing a new run. Keep any prior results separately or use a separate dataset-root path.

```text
dataset_output/<case>_output/
├── Conv_CT/                         # original ConvCT NIfTI
├── Conv_CT_in_RAS/                  # reoriented image and orientation reference
├── Conv_CT_in_RAS_cropped/          # split inputs when enabled
├── VMI40/                           # original VMI image
├── VMI40_cropped/                   # cropped VMI and binary spine channel
└── Segmentation/
    ├── Spine_segmentation_in_RAS_cropped/
    ├── Spine_segmentation_in_RAS/
    ├── Spine_segmentation_final/<case>_spine_segmentation.nii.gz
    ├── crop_parameters_folder/<case>.json
    ├── Lesion_segmentation_cropped/
    └── Lesion_segmentation_final/<case>_lesions_segmentation.nii.gz
```

Splitting reduces the size of each spine prediction, while DICOM conversion, working copies and the full restored masks still require disk and host memory. For a series selection error, inspect the patient group and exact metadata. For a missing checkpoint, check the trainer directory and original `fold_all` layout. The legacy predictor uses nnU-Net multiprocessing by default. See [installation troubleshooting](Installation.md#troubleshooting).

[New NIfTI data](V2_new_data_prediction.md) · [Dataset Viewer](Dataset_viewer.md) · [Model Zoo](Model_zoo.md)
