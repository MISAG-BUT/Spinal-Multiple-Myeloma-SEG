# Spinal-Multiple-Myeloma-SEG

## Overview
Welcome to the repository for the paper **"Spinal-Multiple-Myeloma-SEG"**! This repository provides the code for the implementation of the networks for segmentation within the popular [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).

### Read the [paper](paper): 

Authors:  
Michal Nohel, Vlatimil Valek, Tomas Rohan, Martin Stork, Roman Jakubcek, Jiri Chmelik and Marek Dostal 

Author Affiliations:  
Department of Biomedical Engineering, Faculty of Electrical Engineering and Communication, Brno University of Technology, Brno, Czech Republic
Department of Radiology and Nuclear Medicine, University Hospital Brno and Masaryk University, Brno, Czech Republic
Internal Hematology and Oncology Clinic, University Hospital Brno, Brno, Czech Republic
Department of Biophysics, Masaryk University, Brno, Czech Republic

The repository contains two main scripts:  
1. A script for visualizing the dataset, allowing you to explore the spinal and myeloma images used.  
2. A script for performing segmentation predictions on the spine and myeloma regions. These predictions were utilized to create the myeloma database used in the study.  

Together, these scripts allow researchers to reproduce the dataset preparation and segmentation workflows described in the paper.

---

## Trained Models / Zenodo
Pre-trained nnU-Net models for this project are available on Zenodo. These models were trained for segmentation of spine, vertebrae, and osteolytic spinal multiple myeloma lesions in dual-energy CT data.

- The models were created as part of the dataset preparation and annotation pipeline for **Spinal-Multiple-Myeloma-SEG**, released via [The Cancer Imaging Archive (TCIA)](https://doi.org/10.7937/k4qv-hh78).
- Spine and vertebrae segmentation models were trained on **conventional CT data** from the VerSe2020 dataset, as well as on a combined VerSe2020 + multiple myeloma dataset. These masks were used to localize the spinal region and define spatial cropping for downstream lesion segmentation.
- Lesion segmentation models were trained exclusively on **VMI 40 keV images** using an iterative semi-automatic annotation workflow.
- All models operate on **NIfTI (.nii.gz)** inputs and are intended for **research and development purposes only**.

**Links:**
- Zenodo repository with traied models is here at: [Spinal-Multiple-Myeloma-SEG_nnUNet_models](https://zenodo.org/uploads/18598645)
- Zenodo snapshot of GitHub repo: [https://zenodo.org/records/18596640](https://zenodo.org/records/18596640)
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

Clone the project repository and move into the project folder:

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

#### Step 3: Additional / Optional  Packages
Install additional required packages:
> 💡 **Tip:** Depending on your Linux setup, you may need system libraries for `napari`/Qt to run (e.g., `libxcb-xinerama0`). Install only what you need.
```bash
# DICOM support
pip install pydicom
# Napari viewer for GUI visualization
pip install napari
# Full napari installation with all optional plugins
pip install -U 'napari[all]'
```

#### Step 4: Full Setup Summary
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
# Optional / additional packages
pip install pydicom napari
pip install -U 'napari[all]'
```

Always replace <path-to-your-python-executable> with the correct path on your system. For different CUDA versions or CPU-only setups, check the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
.

## Citation

If you use this code in your research, please cite our paper:
TO DO






