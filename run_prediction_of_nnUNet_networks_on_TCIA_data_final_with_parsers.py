# -*- coding: utf-8 -*-
"""
Spinal Multiple Myeloma Segmentation Pipeline
=============================================

This script runs a complete nnU-Net-based segmentation pipeline for a single patient:

    1) Conversion of ConvCT and VMI40 DICOM data to NIfTI
    2) Spine segmentation from ConvCT
    3) Reorientation of spine segmentation to original image space
    4) Lesion segmentation from VMI40
    5) Final reconstruction of lesion segmentation in original space

The pipeline is designed to always start from a clean working directory
to ensure reproducibility and avoid mixing results from previous runs.

Hardware & OS Testing
--------------------
The pipeline has been tested on both Linux and Windows systems with high-end GPUs:

Linux:
    - GPU: Nvidia Titan Xp, 12 GB GDDR5
    - Motherboard: GIGABYTE Z690 GAMING X DDR5
    - CPU: Intel Core i9 12900KF (8+8 cores/threads, 2.4/3.2 GHz)
    - RAM: 64 GB DDR5
    - Storage: SSD 1 TB (SYSTEM), HDD 4 TB RAID5 (DATA)
    - OS: Ubuntu 24.04

Windows:
    - GPU: EVGA GeForce RTX 3090, 24 GB GDDR6
    - CPU: Intel Core i9-10900KF (10/20 cores/threads, 3.7 GHz)
    - RAM: 64 GB
    - Storage: SSD M.2 2TB (SYSTEM)
    - OS: Windows 10

Notes on Multiprocessing
------------------------
- By default, the pipeline is configured for Linux and may use multiprocessing
  for faster nnU-Net inference.
- On Windows, due to potential issues with Python multiprocessing, the default
  nnU-Net inference (variant 1) may fail when run from a clean session.
- In such cases, open `utils.py` and in the function `run_nnunet_inference`, 
  switch to variant 2 (`predict_from_files_sequential`), which disables multiprocessing.
  This ensures safe execution on Windows, although it may run slower.

Author: nohel
Created: Aug 13, 2025
"""

# ==========================================================
# Imports & nnU-Net environment setup
# ==========================================================
import os
import sys
import shutil
import argparse
from os.path import join

# Path to your nnU-Net repository (change this to your actual path)
NNUNET_REPO_PATH = r"F:/Code/nnUNet"  

if NNUNET_REPO_PATH not in sys.path:
    sys.path.append(NNUNET_REPO_PATH)

# nnU-Net requires these environment variables to be set.
# The exact values are not used directly here; they mainly suppress warnings.
os.environ["nnUNet_raw"] = "nnUNet_project/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "nnUNet_project/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "nnUNet_project/nnUNet_results"

from nnunetv2.paths import nnUNet_results
from utils import *

# ==========================================================
# Argument parser
# ==========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Spinal Multiple Myeloma nnU-Net segmentation pipeline"
    )

    parser.add_argument(
        "--path_to_DICOM_folders",
        type=str,
        default="F:/Example_data/DATA/MM_DICOM_Dataset",
        help=(
            "Path to the DICOM folders, which are organized by patient ID\n"
            "and then by series description. (e.g. F:/Example_data/DATA/MM_DICOM_Dataset)"
        )
    )

    parser.add_argument(
        "--ID_patient",
        type=str,
        default="S840",
        help="Patient ID folder name (e.g., S840)"
    )

    parser.add_argument(
        "--nnUNet_results",
        type=str,
        default="F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models",
        help="Path to the trained nnU-Net model folder"
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Split ConvCT volumes along Z-axis to reduce memory usage"
    )

    return parser.parse_args()


# ==========================================================
# Main pipeline
# ==========================================================
def main(path_to_DICOM_folders, ID_patient, nnUNet_results, split_data=True):
    """
    Run the complete segmentation pipeline for a single patient.

    Parameters
    ----------
    path_to_DICOM_folders : str
        Path to the root directory containing patient DICOM data.
    ID_patient : str
        Name of the patient folder (e.g. 'S840').
    nnUNet_results : str
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

    # Identify ConvCT and VMI40 DICOM folders
    patient_name, path_to_convCT_folder, path_to_VMI40_folder = find_convCT_and_VMI40_at_DICOM_folder(patient_main_file)

    # ======================================================
    # 2. Working directory preparation
    # ======================================================
    print("Creation of working folders and conversion to NIfTI - Start")

    working_folder = join(path_to_output_folder, f"{patient_name}_output")

    # Always remove the working directory if it already exists.
    # This guarantees a clean run and ensures that all results
    # are regenerated from scratch for the given patient.
    if os.path.exists(working_folder):
        print(f"Working folder already exists, removing: {working_folder}")
        shutil.rmtree(working_folder)

    # Define all working subfolders
    working_folder_conv_CT = join(working_folder, "Conv_CT")
    working_folder_conv_CT_in_RAS = join(working_folder, "Conv_CT_in_RAS")
    working_folder_conv_CT_in_RAS_cropped = join(working_folder, "Conv_CT_in_RAS_cropped")

    working_folder_VMI40 = join(working_folder, "VMI40")
    working_folder_VMI40_cropped = join(working_folder, "VMI40_cropped")

    working_folder_Segmentation = join(working_folder, "Segmentation")
    working_folder_Spine_segmentation_final = join(working_folder_Segmentation, "Spine_segmentation_final")
    working_folder_Spine_segmentation_in_RAS = join(working_folder_Segmentation, "Spine_segmentation_in_RAS")
    working_folder_Spine_segmentation_in_RAS_cropped = join(working_folder_Segmentation, "Spine_segmentation_in_RAS_cropped")

    working_folder_crop_parameters_folder = join(working_folder_Segmentation, "crop_parameters_folder")
    working_folder_Lesion_segmentation_cropped = join(working_folder_Segmentation, "Lesion_segmentation_cropped")
    working_folder_Lesion_segmentation_final = join(working_folder_Segmentation, "Lesion_segmentation_final")

    # Create folders and convert DICOM → NIfTI
    create_working_folders_and_convert_to_nifti(
        patient_name,
        working_folder,
        working_folder_conv_CT,
        working_folder_conv_CT_in_RAS,
        working_folder_conv_CT_in_RAS_cropped,
        working_folder_VMI40,
        working_folder_VMI40_cropped,
        working_folder_Segmentation,
        working_folder_Spine_segmentation_final,
        working_folder_Spine_segmentation_in_RAS,
        working_folder_Spine_segmentation_in_RAS_cropped,
        working_folder_crop_parameters_folder,
        working_folder_Lesion_segmentation_cropped,
        working_folder_Lesion_segmentation_final,
        path_to_convCT_folder,
        path_to_VMI40_folder
    )

    print("Creation of working folders and conversion to NIfTI - Done")

    # ======================================================
    # 3. Spine segmentation (ConvCT)
    # ======================================================
    print("Spine segmentation - Start")
    print("Spine segmentation - Preparation of data")

    if split_data:
        # Split ConvCT along Z-axis to reduce memory usage
        split_convCT_data(working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, patient_name)
        input_folder = working_folder_conv_CT_in_RAS_cropped
        output_folder = working_folder_Spine_segmentation_in_RAS_cropped
    else:
        input_folder = working_folder_conv_CT_in_RAS
        output_folder = working_folder_Spine_segmentation_in_RAS

    print("Spine segmentation - Prediction with nnU-Net")

    run_nnunet_inference(
        nnUNet_results,
        dataset_name="Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together",
        trainer_name="nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=("all",),
        input_folder=input_folder,
        output_folder=output_folder
    )

    print("Spine segmentation - Prediction finished")

    # ======================================================
    # 4. Spine segmentation postprocessing
    # ======================================================
    print("Spine segmentation - Reorientation to original space")

    if split_data:
        print("Spine segmentation - Merging split predictions")
        merge_data(output_folder, working_folder_Spine_segmentation_in_RAS, patient_name)

    reorient_spine_segmentation_to_original_space(
        working_folder_Spine_segmentation_final,
        working_folder_Spine_segmentation_in_RAS,
        working_folder_conv_CT_in_RAS
    )

    print("Spine segmentation - Done")

    # ======================================================
    # 5. Lesion segmentation (VMI40)
    # ======================================================
    print("Lesion segmentation - Start")
    print("Lesion segmentation - Preparation of data")

    prepare_data_for_lesion_segmentation(
        working_folder_Spine_segmentation_final,
        working_folder_crop_parameters_folder,
        working_folder_VMI40,
        working_folder_VMI40_cropped,
        patient_name
    )

    print("Lesion segmentation - Prediction with nnU-Net")

    run_nnunet_inference(
        nnUNet_results,
        dataset_name="Dataset710_MM_Lesion_seg_just_VMI_40",
        trainer_name="nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=("all",),
        input_folder=working_folder_VMI40_cropped,
        output_folder=working_folder_Lesion_segmentation_cropped
    )

    print("Lesion segmentation - Prediction finished")

    # ======================================================
    # 6. Final lesion segmentation reconstruction
    # ======================================================
    print("Lesion segmentation - Reorientation to original space")

    reorient_lesion_segmentation_to_original_space(
        working_folder_crop_parameters_folder,
        working_folder_VMI40,
        working_folder_Lesion_segmentation_cropped,
        working_folder_Lesion_segmentation_final,
        patient_name
    )

    print("Lesion segmentation - Done")


# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    args = parse_arguments()
    main(args.path_to_DICOM_folders, args.ID_patient, args.nnUNet_results, split_data=args.split)