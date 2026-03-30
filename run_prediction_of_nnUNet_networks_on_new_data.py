# -*- coding: utf-8 -*-
"""
Spinal Multiple Myeloma Segmentation Pipeline
=============================================

This script runs a nnU-Net-based segmentation pipeline for multi-energy CT
data with NIfTI input:

    1) Loads ConvCT and VMI40 volumes from NIfTI (RAS)
    2) Prepares a clean working directory
    3) Runs nnU-Net inference and produces final segmentation outputs

The pipeline always starts from a fresh working directory to ensure
reproducible results for each patient.

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
"""

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

# ----------------------------------------------------------
# Project-specific imports
# ----------------------------------------------------------
from utils import *


# ==========================================================
# Argument parser
# ==========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Spinal Multiple Myeloma nnU-Net segmentation pipeline for NIfTI input"
    )

    parser.add_argument(
        "--path_to_convCT_nifti",
        dest="path_to_convCT_nifti",
        type=str,
        default=config.PATH_TO_CONVCT_NIFTI,
        help="Path to ConvCT NIfTI file (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--path_to_VMI40_nifti",
        dest="path_to_VMI40_nifti",
        type=str,
        default=config.PATH_TO_VMI40_NIFTI,
        help="Path to VMI40 NIfTI file (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--path_to_output_folder",
        dest="path_to_output_folder",
        type=str,
        default=config.PATH_TO_OUTPUT_FOLDER,
        help="Path to output folder"
    )
    parser.add_argument(
        "--path_to_nnunet_results",
        type=str,
        default=config.PATH_TO_NNUNET_RESULTS,
        help="Path to the trained nnU-Net model folder"
    )
    parser.add_argument(
        "--split",
        type=lambda x: (str(x).lower() == 'true'),
        default=config.SPLIT_CONVCT_DEFAULT,
        help="Split ConvCT volumes along Z-axis to reduce memory usage (default: True). Use --split False to disable."
    )

    return parser.parse_args()

# ==========================================================
# Main pipeline
# ==========================================================

def main(path_to_convCT_nifti, path_to_VMI40_nifti, path_to_output_folder, path_to_nnunet_results, split_data=True):

    # ======================================================
    # 1. Input paths and patient-specific setup
    # ======================================================
    patient_name = get_patient_name(path_to_convCT_nifti)
    print(f"Patient: {patient_name}")

    # ======================================================
    # 2. Working directory preparation and nifti renaming
    # ======================================================
    print("Creation of working folders - Start")

    working_folder = join(path_to_output_folder, f"{patient_name}_output")

    # Always remove the working directory if it already exists.
    # This guarantees a clean run and ensures that all results
    # are regenerated from scratch for the given patient.
    if os.path.exists(working_folder):
        print(f"Working folder already exists, removing: {working_folder}")
        shutil.rmtree(working_folder)

    working_folder_conv_CT = join(working_folder, "Conv_CT")
    working_folder_conv_CT_cropped = join(working_folder, "Conv_CT_cropped")

    working_folder_VMI40 = join(working_folder, "VMI40")
    working_folder_VMI40_cropped = join(working_folder, "VMI40_cropped")

    working_folder_Segmentation = join(working_folder, "Segmentation")
    working_folder_Spine_segmentation_cropped = join(working_folder_Segmentation, "Spine_segmentation_cropped")
    working_folder_Spine_segmentation_final = join(working_folder_Segmentation, "Spine_segmentation_final")

    working_folder_crop_parameters_folder = join(working_folder_Segmentation, "crop_parameters_folder")
    working_folder_Lesion_segmentation_cropped = join(working_folder_Segmentation, "Lesion_segmentation_cropped")
    working_folder_Lesion_segmentation_final = join(working_folder_Segmentation, "Lesion_segmentation_final")


    maybe_mkdir_p(working_folder) 
    maybe_mkdir_p(working_folder_conv_CT) 
    maybe_mkdir_p(working_folder_conv_CT_cropped) 
    maybe_mkdir_p(working_folder_VMI40) 
    maybe_mkdir_p(working_folder_VMI40_cropped) 
    maybe_mkdir_p(working_folder_Segmentation)
    maybe_mkdir_p(working_folder_Spine_segmentation_cropped) 
    maybe_mkdir_p(working_folder_Spine_segmentation_final) 
    maybe_mkdir_p(working_folder_crop_parameters_folder) 
    maybe_mkdir_p(working_folder_Lesion_segmentation_cropped) 
    maybe_mkdir_p(working_folder_Lesion_segmentation_final) 


    convCT_dst = join(
        working_folder_conv_CT,
        patient_name + "_conv_RAS_0000.nii.gz"
    )

    shutil.copy(path_to_convCT_nifti, convCT_dst)
    print(f"Copied ConvCT: {convCT_dst}")

    vmi40_dst = join(
        working_folder_VMI40,
        patient_name + "_monoe_40kev_0000.nii.gz"
    )

    shutil.copy(path_to_VMI40_nifti, vmi40_dst)
    print(f"Copied VMI40: {vmi40_dst}")

    print("Creation of working folders  - Done")

    # ======================================================
    # 3. Spine segmentation (ConvCT)
    # ======================================================
    print("Spine segmentation - Start")
    print("Spine segmentation - Preparation of data")

    if split_data:
        # Split ConvCT along Z-axis to reduce memory usage
        split_convCT_data(working_folder_conv_CT, working_folder_conv_CT_cropped, patient_name)
        input_folder = working_folder_conv_CT_cropped
        output_folder = working_folder_Spine_segmentation_cropped
    else:
        input_folder = working_folder_conv_CT
        output_folder = working_folder_Spine_segmentation_final

    print("Spine segmentation - Prediction with nnU-Net")
    
    run_nnunet_inference(
        path_to_nnunet_results,
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
        merge_data(output_folder, working_folder_Spine_segmentation_final, patient_name)
        
    f = next(x for x in os.listdir(working_folder_Spine_segmentation_final) if x.endswith(".nii.gz"))
    os.rename(os.path.join(working_folder_Spine_segmentation_final, f), os.path.join(working_folder_Spine_segmentation_final, f[:-16] + "_spine_segmentation.nii.gz"))

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
        path_to_nnunet_results,
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
    print(f"Final spine segmentation saved at: {working_folder_Spine_segmentation_final}")
    print(f"Final lesion segmentation saved at: {working_folder_Lesion_segmentation_final}")



# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    #path_to_convCT_nifti = "F:/Example_data/DATA/New_Data/Myel_001_conv.nii.gz"
    #path_to_VMI40_nifti = "F:/Example_data/DATA/New_Data/Myel_001_monoe_40kev.nii.gz"
    #path_to_output_folder = "F:/Example_data/DATA/New_Data/Output_folder"
    #path_to_nnunet_results = "F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models"
    #split_data = True
    #main(path_to_convCT_nifti, path_to_VMI40_nifti, path_to_output_folder, path_to_nnunet_results, split_data)

    args = parse_arguments()
    main(args.path_to_convCT_nifti, args.path_to_VMI40_nifti, args.path_to_output_folder, args.path_to_nnunet_results, split_data=args.split_data)