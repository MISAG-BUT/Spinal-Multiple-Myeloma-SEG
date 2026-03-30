# -*- coding: utf-8 -*-
"""
NIfTI-based pipeline for Multi-Energy CT processing
==================================================

This script:
    1) Loads ConvCT and VMI40 volumes from NIfTI (RAS)
    2) Prepares working directory
    3) Runs further processing (nnU-Net, etc.)

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

import SimpleITK as sitk

# ==========================================================
# Helper function
# ==========================================================

def get_patient_name(path):
    """
    Returns patient name from NIfTI file.
    Uses metadata if available, otherwise extracts from filename.
    """
    if os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz")):
        # Fallback: extract patient name from filename
        filename = os.path.basename(path)

        # Remove extension
        if filename.endswith(".nii.gz"):
            filename = filename[:-7]
        elif filename.endswith(".nii"):
            filename = filename[:-4]

        parts = filename.split("_")

        # If filename contains at least two parts (e.g. Myel_001_*)
        if len(parts) >= 2:
            patient_name = parts[0] + "_" + parts[1]
        else:
            # Fallback: use full filename (e.g. pat01)
            patient_name = filename

        return patient_name

    else:
        raise ValueError(f"Unsupported input path: {path}")

# ==========================================================
# Argument parser
# ==========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Spinal Multiple Myeloma nnU-Net segmentation pipeline (NIfTI input)"
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
    '''
    run_nnunet_inference(
        path_to_nnunet_results,
        dataset_name="Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together",
        trainer_name="nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=("all",),
        input_folder=input_folder,
        output_folder=output_folder
    )
    '''
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
    main(
        args.path_to_convCT_nifti,
        args.path_to_VMI40_nifti,
        args.path_to_output_folder,
        args.path_to_nnunet_results,
        split_data=args.split_data
    )