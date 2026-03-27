# ==========================================================
# Imports & nnU-Net environment setup
# ==========================================================

import shutil
import argparse
from os.path import join
import sys, os
# Import config for all paths and environment setup
import config
from config import NNUNET_REPO_PATH, NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS_ENV
# Set up nnU-Net environment and sys.path
def setup_nnunet_env():
    """Set up sys.path and nnU-Net environment variables from config."""    
    if NNUNET_REPO_PATH not in sys.path:
        sys.path.append(NNUNET_REPO_PATH)
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS_ENV

setup_nnunet_env()

from utils import *

# ==========================================================
# Main pipeline
# ==========================================================
def main(path_to_DICOM_folders, ID_patient, path_to_nnunet_results, split_data=True):
    """
    Run the complete segmentation pipeline for a single patient.

    Parameters
    ----------
    path_to_DICOM_folders : str
        Path to the root directory containing patient DICOM data.
    ID_patient : str
        Name of the patient folder (e.g. 'S840').
    path_to_nnunet_results : str
        Path to trained nnU-Net models.
    split_data : bool, optional
        If True, images are split along the Z-axis to reduce memory usage.
        If False, the full volume is processed at once (requires very high RAM).
    """

    # ======================================================
    # 1. Input paths and patient-specific setup
    # ======================================================
    patient_main_file = join(path_to_DICOM_folders, ID_patient)
    path_to_output_folder = path_to_DICOM_folders + "_output"

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    base = 'F:/Example_data/DATA/'  # path to the dataset folder
    path_to_DICOM_folders = join(base, 'Spinal-Multiple-Myeloma-SEG')  #path to the DICOM folders, which are organized by patient ID and then by series description
    path_to_nnunet_results = "F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models"  #path to the folder containing trained nnU-Net models (should have subfolders for each model)
    ID_patient = "Myel_001"  
    split_data = True # If True, data are split along Z-axis to reduce memory requirements. If False, the full volume is processed at once (requires ~256 GB RAM).
    main(path_to_DICOM_folders, ID_patient, path_to_nnunet_results, split_data)
