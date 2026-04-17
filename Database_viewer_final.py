# -*- coding: utf-8 -*-
"""
Multi-Energy CT Visualization with Segmentation Overlays
=========================================================

This script:
    1) Loads multiple CT reconstructions (ConvCT, VMI, Calcium Suppression)
       from DICOM folders for a selected patient
    2) Loads corresponding spine and lesion segmentation masks (NIfTI)
    3) Visualizes everything together in napari for interactive inspection

The structure of this script mirrors the main segmentation pipeline
to ensure consistency across the project.

Author: nohel
Created: Jun 24, 2025
"""

# ==========================================================
# Imports & nnU-Net environment setup
# ==========================================================

import argparse
import os
from os.path import join
import sys
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
from utils import find_dicom_series, select_dicom_series_group, load_DICOM_data_SITK
import SimpleITK as sitk
import pydicom
import napari

# ==========================================================
# Argument parser
# ==========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Energy CT visualization with segmentation overlays"
    )

    parser.add_argument(
        "--path_to_DICOM_folders",
        type=str,
        default=config.PATH_TO_DICOM_FOLDERS,
        help=(
            "Path to the DICOM folders, which are organized by patient ID\n"
            "and then by series description. (e.g. F:/Example_data/DATA/MM_DICOM_Dataset)"
        )
    )

    parser.add_argument(
        "--path_to_segmentations",
        type=str,
        default=config.PATH_TO_SEGMENTATIONS,
        help=(
            "Path to the segmentation masks (NIfTI), which are organized by patient ID\n"
            "and then by mask type (spine or lesions) (e.g. F:/Example_data/DATA/MM_NIfTI Segmentation)."
        )
    )

    parser.add_argument(
        "--ID_patient",
        type=str,
        default=config.ID_PATIENT,
        help="Patient ID used as the folder name (e.g. S840)"
    )

    parser.add_argument(
        "--series_tag",
        type=str,
        default=config.PATIENT_SERIES_TAG,
        help=(
            "Optional patient series variant tag when multiple acquisitions exist in the same patient folder. "
            "Use 'a' or 'b' for patient names like Myel_012_a / Myel_012_b. "
            "If the patient has multiple acquisition variants, this must be specified."
        )
    )

    return parser.parse_args()

# ==========================================================
# Main
# ==========================================================

def main(path_to_DICOM_folders,path_to_segmentations,ID_patient, series_tag=None):

    print(f"Running visualization for patient {ID_patient}")
    print(f"DICOM path: {path_to_DICOM_folders}")
    print(f"Segmentation path: {path_to_segmentations}")
    if series_tag and series_tag.strip():
        print(f"Using patient series tag: {series_tag.strip()}")

    # ------------------------------------------------------
    # Locate patient folder
    # ------------------------------------------------------
    patient_main_file = join(path_to_DICOM_folders, ID_patient)

    print(f"\nLoading data for patient: {ID_patient}")
    print(f"DICOM root: {patient_main_file}")

    # ------------------------------------------------------
    # Identify and select the correct DICOM series group
    # ------------------------------------------------------
    selected_DICOM_folders = select_dicom_series_group(patient_main_file, series_tag)

    print("\nUsing the following DICOM folders for visualization:")
    for folder in selected_DICOM_folders:
        print(f"  - {folder}")

    # ======================================================
    # Load DICOM data based on SeriesDescription
    # ======================================================
    print("\nLoading DICOM volumes...")

    for DICOM_folder_path in selected_DICOM_folders:

        # Load all DICOM files except DIRFILE and Segmentation series (if present in DICOM)
        DICOM_files = [
            join(DICOM_folder_path, f)
            for f in os.listdir(DICOM_folder_path)
            if f != "DIRFILE"
        ]

        # Identify series using metadata of the first slice
        series_description = pydicom.dcmread(DICOM_files[0]).get("SeriesDescription")
        print(f"  -> {series_description}")

        # Match known series types
        if series_description == "Calcium Suppression 25 Index[HU*]":
            CaSupp25_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "Calcium Suppression 50 Index[HU*]":
            CaSupp50_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "Calcium Suppression 75 Index[HU*]":
            CaSupp75_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "Calcium Suppression 100 Index[HU*]":
            CaSupp100_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "MonoE 40keV[HU]":
            VMI40_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "MonoE 80keV[HU]":
            VMI80_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "MonoE 120keV[HU]":
            VMI120_zxy = load_DICOM_data_SITK(DICOM_folder_path)

        elif series_description == "Segmentation":
            continue  # Skip segmentation series if present in DICOM (we will load NIfTI masks instead)        

        else:
            # Fallback: assume this is conventional CT
            ConvCT_zxy = load_DICOM_data_SITK(DICOM_folder_path)

            # Extract patient name (remove _conv suffix if present)
            patient_name = series_description[:-5]

    # ======================================================
    # Load segmentation masks (NIfTI)
    # ======================================================
    print("\nLoading segmentation masks...")

    path_spine_mask = join(
        path_to_segmentations,
        patient_name,
        f"{patient_name}_spine_segmentation.nii.gz"
    )

    path_lesion_mask = join(
        path_to_segmentations,
        patient_name,
        f"{patient_name}_lesions_segmentation.nii.gz"
    )

    SegmMaskSpine = sitk.GetArrayFromImage(sitk.ReadImage(path_spine_mask))
    SegmMaskLesions = sitk.GetArrayFromImage(sitk.ReadImage(path_lesion_mask))

    print("Segmentation masks loaded.")

    # ======================================================
    # Visualization with napari
    # ======================================================
    print("\nLaunching napari viewer...")

    viewer = napari.Viewer()

    # --- Base CT ---
    viewer.add_image(ConvCT_zxy, name="ConvCT", colormap="gray", blending="additive")

    # --- Virtual Monoenergetic Images ---
    viewer.add_image(VMI40_zxy, name="VMI40", colormap="gray", blending="additive", visible=False)
    viewer.add_image(VMI80_zxy, name="VMI80", colormap="gray", blending="additive", visible=False)
    viewer.add_image(VMI120_zxy, name="VMI120", colormap="gray", blending="additive", visible=False)

    # --- Calcium Suppression Reconstructions ---
    viewer.add_image(CaSupp25_zxy, name="CaSupp25", colormap="gray", blending="additive", visible=False)
    viewer.add_image(CaSupp50_zxy, name="CaSupp50", colormap="gray", blending="additive", visible=False)
    viewer.add_image(CaSupp75_zxy, name="CaSupp75", colormap="gray", blending="additive", visible=False)
    viewer.add_image(CaSupp100_zxy, name="CaSupp100", colormap="gray", blending="additive", visible=False)

    # --- Segmentation overlays ---
    viewer.add_image(
        SegmMaskSpine,
        name="SpineMask",
        colormap="blue",
        blending="additive",
        opacity=0.5
    )

    viewer.add_image(
        SegmMaskLesions,
        name="LesionMask",
        colormap="red",
        blending="additive",
        opacity=1.0
    )

    napari.run()

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    #base = 'F:/Example_data/DATA/'  # path to the dataset folder
    #path_to_DICOM_folders = join(base, 'Spinal-Multiple-Myeloma-SEG')  #path to the DICOM folders, which are organized by patient ID and then by series description
    #path_to_segmentations = join(base, 'MM_NIfTI_Segmentation')  #path to the segmentation masks, which are organized by patient ID and then by mask type (spine or lesions)
    #ID_patient = "Myel_001"
    #series_tag= ""
    #main(path_to_DICOM_folders, path_to_segmentations, ID_patient, series_tag)  

    args = parse_arguments()
    main(
        args.path_to_DICOM_folders,
        args.path_to_segmentations,
        args.ID_patient,
        series_tag=args.series_tag
    )

    

