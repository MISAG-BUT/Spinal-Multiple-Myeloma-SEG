# Dataset Viewer

[Back to main README](../README.md)

The Dataset Viewer, implemented in [Database_viewer_final.py](../Database_viewer_final.py), opens spectral DICOM reconstructions and corresponding NIfTI segmentation masks from the [TCIA dataset](https://doi.org/10.7937/k4qv-hh78) in napari. It runs without model weights.

## Installation and data layout

Install the optional GUI dependencies from the repository root:

```bash
uv sync --locked --extra gui
```

Pass the two roots above the case-specific subdirectories:

```text
dataset/
└── Myel_001/
    └── study-or-acquisition/
        ├── conventional-series/
        ├── vmi40-series/
        └── ...                      # other spectral DICOM series

segmentations/
└── Myel_001/
    ├── Myel_001_spine_segmentation.nii.gz
    └── Myel_001_lesions_segmentation.nii.gz
```

Acquisitions are grouped by DICOM `PatientName` with `PatientID` as fallback. `--series_tag` selects a suffix such as `a` where several groups share the patient folder. The viewer recognizes `MonoE 40keV[HU]`, `MonoE 80keV[HU]`, `MonoE 120keV[HU]` and `Calcium Suppression 25 Index[HU*]` (likewise 50, 75 and 100). It skips the `Segmentation` DICOM series. Other series descriptions are treated as ConvCT. Use the dataset convention `<case>_konv`: the viewer drops the final five characters of that description to form the mask folder and filename prefix. This case name can differ from the patient folder, for example when the acquisition has its own suffix.

Provide all eight reconstructions and both masks. The viewer constructs all eight image layers unconditionally, so missing series can cause startup to fail. Its overlays use the loaded voxel arrays. Supply matching grids and check the displayed alignment.

## Running the viewer

```bash
uv run --locked --extra gui spinal-db-viewer \
  --path_to_DICOM_folders /path/to/dataset \
  --path_to_segmentations /path/to/segmentations \
  --ID_patient Myel_001
```

Use `uv run --locked --extra gui python Database_viewer_final.py` or `.venv/bin/python Database_viewer_final.py` with the same arguments for direct execution. In VS Code, select the GUI-enabled interpreter and open the script. Set local defaults in [config.py](../config.py) for a no-argument launch, or pass these arguments through the terminal or debugger.

| Argument | Value | Default and behavior |
|---|---|---|
| `--path_to_DICOM_folders` | Directory | `config.PATH_TO_DICOM_FOLDERS`. Parent of patient folders. |
| `--path_to_segmentations` | Directory | `config.PATH_TO_SEGMENTATIONS`. Parent of the flat, case-specific NIfTI mask directories. |
| `--ID_patient` | Folder name | `config.ID_PATIENT`, initially `Myel_001`. |
| `--series_tag` | Acquisition suffix | `config.PATIENT_SERIES_TAG`, initially empty. Required with multiple groups. |
| `-h`, `--help` | Flag | Display help and exit. |

## Using the layers

ConvCT opens as the visible grayscale background. The VMI and calcium-suppressed layers start hidden. Toggle their visibility in the layer list to compare reconstructions. Spine and lesion masks open as blue and red overlays. Select a layer to adjust its opacity and use the slice slider to move through the volume.

## Limitations and troubleshooting

- V3 publishes into nested per-run output folders. The viewer expects DICOM and the flat masks shown above and does not traverse V3 runs. For V3 outputs, open the original reference NIfTI and final masks in a NIfTI viewer that respects spatial transforms.
- For missing masks, check the name derived from the conventional series description. For missing reconstructions or the wrong acquisition, check the series descriptions and group suffix.
- For napari or Qt errors keep `--extra gui` on synchronizing uv commands. Check the selected Python interpreter, Qt system libraries and the active display. See [installation troubleshooting](Installation.md#troubleshooting).
- Interactive use was verified on the Linux environment before the module-layout refactoring. Later smoke checks covered imports, the launcher and offscreen Qt initialization.

[Installation](Installation.md) · [TCIA prediction](V1_TCIA_prediction.md) · [Benchmark prediction](V3_benchmark_prediction.md)
