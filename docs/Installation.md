# Installation

[Back to main README](../README.md)

[Requirements](#system-requirements) · [uv](#install-with-uv) · [pip](#install-with-pip) · [VS Code](#run-in-vs-code) · [Verification](#verify-the-environment) · [Troubleshooting](#troubleshooting)

## System requirements

The locked profile is Linux x86-64 with Python 3.12 (`>=3.12,<3.13`). Install Git to resolve the pinned nnU-Net source. Prediction uses an NVIDIA CUDA device. The base installation also supports command-line inspection without a GPU. The Dataset Viewer needs the optional napari and Qt dependencies and a working display.

| Component | Project pin |
|---|---|
| PyTorch and torchvision | `2.10.0+cu130`, `0.25.0+cu130` from the PyTorch CUDA 13.0 index |
| nnU-Net | `2.6.4`, Git revision `273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0` |
| DICOM and NIfTI | pydicom `3.0.2`, NiBabel `5.3.3`, SimpleITK `2.5.3` |
| Optional GUI | `napari[all]==0.6.6` |

The PyTorch distribution supplies its CUDA runtime libraries. A compatible host NVIDIA driver is still required. The system CUDA Toolkit is primarily for building CUDA software and is unnecessary for this wheel-based inference setup. See [NVIDIA's CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/).

The locked profile was verified on Ubuntu 24.04, Python 3.12.3 and an RTX 5080 (`sm_120`). GPU and host memory use depend on scan dimensions, model and splitting. Allow room for the environment, downloaded archives, extracted checkpoints, staging and outputs. These measurements do not establish universal minimum system requirements.

## Install with uv

Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/). Clone the repository and open its root:

```bash
git clone https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG.git
cd Spinal-Multiple-Myeloma-SEG
uv sync --locked
```

If you already have a checkout, run the sync command from its root. uv creates `.venv` and installs the project in editable mode. Re-run the same sync to reconcile the environment with the lockfile.

For the viewer and Qt:

```bash
uv sync --locked --extra gui
uv run --locked --extra gui spinal-db-viewer --help
```

Include `--extra gui` on later synchronizing `uv run` and `uv sync` commands to keep GUI packages installed. Direct use of the environment's Python leaves the installed extras in place.

## Install with pip

Use a fresh Python 3.12 virtual environment on Linux x86-64. Install the matching CUDA wheels first, then the remaining locked requirements and the repository:

```bash
python3.12 -m venv .venv-pip
source .venv-pip/bin/activate
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

The requirements export includes the Git-pinned nnU-Net dependency. Editable installation registers the launchers and uses the files in your checkout. Replace `-e .` with `.` for a standard non-editable installation.

For GUI in the activated pip environment, preserve the constraints on the base packages:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -c requirements.txt -e '.[gui]'
spinal-db-viewer --help
```

The uv lockfile locks the GUI dependencies. The pip export locks the base dependencies and the optional pip command resolves additional GUI packages under those constraints. Use the activated interpreter and installed launchers directly for a pip environment.

## Run in VS Code

1. Open the repository root with **File → Open Folder**.
2. Run **Python: Select Interpreter** and select `.venv/bin/python` on Linux or `.venv-pip/bin/python` for the example pip environment. On Windows the corresponding interpreter is under `Scripts\python.exe`.
3. Open the main script for the workflow. Run it from the integrated terminal with the interpreter and the workflow's arguments. V1, V2 and the viewer can read their defaults from [config.py](../config.py). V3 requires explicit inputs, selection and output arguments for prediction.
4. For debugging, use **Python Debugger: Python File**, set breakpoints and pass arguments through the debug configuration's `args` list, one token per entry. Use `${workspaceFolder}` as the working directory. For a first check, use `["--help"]`.

```bash
.venv/bin/python run_prediction_of_nnUNet_benchmark_models.py --help
uv run --locked python run_prediction_of_nnUNet_benchmark_models.py --help
```

## Linux and Windows

The legacy workflows have prior Linux and Windows usage. The current Linux lockfile, wheels and V3 validation do not establish support on Windows. A Windows deployment needs separately verified compatible dependency resolution and platform testing.

The examples in these guides use Bash. In PowerShell, use one-line commands or PowerShell line continuations, and quote paths with spaces. A Windows environment uses `.\.venv\Scripts\python.exe`; its activation script is `.\.venv\Scripts\Activate.ps1`. Linux activation is `source .venv/bin/activate`. uv launch commands work without activating the environment.

## Verify the environment

```bash
uv run --locked python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv pip check
uv run --locked spinal-run-nnunet-TCIA --help
uv run --locked spinal-run-nnunet-new-prediction --help
uv run --locked spinal-run-nnunet-benchmark-prediction --help
```

With GUI installed, pass `--extra gui` to the commands above and check the viewer:

```bash
uv run --locked --extra gui python -c "import napari; from qtpy import QtWidgets; print(napari.__version__)"
uv run --locked --extra gui spinal-db-viewer --help
```

A CUDA availability flag checks device discovery. A small operation and synchronization can also verify execution without model weights:

```bash
.venv/bin/python -c "import torch; x=torch.tensor([2., 3.], device='cuda'); y=x*x; torch.cuda.synchronize(); print(y.tolist())"
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Wrong interpreter or missing package | Check `sys.executable` and the VS Code interpreter. Deactivate an unrelated environment and sync from the repository root. |
| CUDA unavailable or incompatible driver | Check `nvidia-smi`, the CUDA wheel tag and NVIDIA driver compatibility. The locked profile uses `+cu130`. |
| VRAM or RAM exhausted | Close competing GPU workloads when appropriate. Keep spine splitting enabled for large volumes and check host RAM as well as GPU memory. |
| Weight files missing | Set up [original models](Model_zoo.md#original-segmentation-models) for V1/V2, or check the [V3 cache and offline rules](V3_benchmark_prediction.md#model-installation-and-cache). |
| GUI import or Qt platform error | Sync with `--extra gui` and confirm the interpreter and display. On Linux check the missing Qt platform plugin dependencies. `QT_QPA_PLATFORM=offscreen` is suitable for a smoke check. An interactive session needs a display. |

[TCIA workflow](V1_TCIA_prediction.md) · [New NIfTI workflow](V2_new_data_prediction.md) · [V3 benchmark workflow](V3_benchmark_prediction.md) · [Dataset Viewer](Dataset_viewer.md)
