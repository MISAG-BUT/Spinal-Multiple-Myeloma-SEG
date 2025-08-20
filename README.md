# Spinal-Multiple-Myeloma-SEG

## Overview
Welcome to the repository for the paper **"Spinal-Multiple-Myeloma-SEG"**! This repository provides the code for the implementation of the networks for segmentation within the popular [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).

## Usage
### Installation

### Environment Setup for nnUNet Prediction (Python 3.13)

This guide explains how to set up the environment for running predictions with [nnUNet](https://github.com/MIC-DKFZ/nnUNet) on the [Spinal-Multiple-Myeloma-SEG](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG) project.
Check out the official [nnUNet installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

---

First, download and install Python 3.13.x from [python.org](https://www.python.org/downloads/). We used Python 3.13.5. During installation, make sure to check "Add Python to PATH".  

Clone the project repository to your computer and move into the project folder:

```bash
git clone https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG.git
cd Spinal-Multiple-Myeloma-SEG
```

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
Install additional required packages:
```bash
pip install pydicom napari
```

For convenience, a full setup summary:
```bash
git clone https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG.git
cd Spinal-Multiple-Myeloma-SEG
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
pip install pydicom napari
```

Always replace <path-to-your-python-executable> with the correct path on your system. For different CUDA versions or CPU-only setups, check the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
.








