# config.py
# Central configuration for all paths and environment variables

# ==========================================================
# 1. TCIA MODE (DICOM → NIfTI, full pipeline)
# ----------------------------------------------------------
# Use these settings for the TCIA dataset or similarly structured DICOM data.
# This mode is used by the CLI command: spinal-run-nnunet-TCIA

# Path to the DICOM folders, organized by patient ID and then by series description
# Example: F:/Example_data/DATA/MM_DICOM_Dataset
PATH_TO_DICOM_FOLDERS = r"F:\TCIA\manifest-1774389300184\Spinal-Multiple-Myeloma-SEG"

# Path to the segmentation masks (NIfTI), organized by patient ID and then by mask type (spine or lesions)
# Example: F:/Example_data/DATA/MM_NIfTI_Segmentation
PATH_TO_SEGMENTATIONS = r"F:\TCIA\manifest-1774389300184\MM_NIfTI_Segmentation"

# Default patient ID (used as folder name)
# Example: Myel_001
ID_PATIENT = "Myel_001"

# Path to the folder containing trained nnU-Net models (should have subfolders for each model)
# Example: F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models
PATH_TO_NNUNET_RESULTS = r"F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models"

# Default for split option in prediction pipeline
# If True, ConvCT volumes are split along Z-axis to reduce memory usage (recommended for most users)
# If False, the full volume is processed at once (requires very high RAM)
SPLIT_CONVCT_DEFAULT = True

# ==========================================================
# 2. NEW DATA MODE (NIfTI input, no DICOM)
# ----------------------------------------------------------
# Use these settings for new/external data already in NIfTI format (RAS orientation).
# This mode is used by the CLI command: spinal-run-nnunet-new-prediction

# Path to ConvCT NIfTI file (for new data pipeline, RAS orientation)
# Example: F:/Example_data/DATA/New_Data/Myel_001_conv.nii.gz
PATH_TO_CONVCT_NIFTI = r"F:/Example_data/DATA/New_Data/Myel_001_conv.nii.gz"

# Path to VMI40 NIfTI file (for new data pipeline, RAS orientation)
# Example: F:/Example_data/DATA/New_Data/Myel_001_monoe_40kev.nii.gz
PATH_TO_VMI40_NIFTI = r"F:/Example_data/DATA/New_Data/Myel_001_monoe_40kev.nii.gz"

# Path to output folder (for new data pipeline)
# Example: F:/Example_data/DATA/New_Data/Output_folder
PATH_TO_OUTPUT_FOLDER = r"F:/Example_data/DATA/New_Data/Output_folder"

# ==========================================================
# COMMON SETTINGS
# ----------------------------------------------------------
# Path to your local nnU-Net repository (required for importing nnU-Net code)
# Example: F:/Code/nnUNet
NNUNET_REPO_PATH = r"F:/Code/nnUNet"

# nnU-Net environment variables (not required to change, only for suppressing nnU-Net warnings)
# You only need to set NNUNET_REPO_PATH to the location of your downloaded nnUNet repository.
# The variables below are set just to avoid warnings and are not used by the pipeline logic.
NNUNET_RAW = "nnUNet_project/nnUNet_raw"
NNUNET_PREPROCESSED = "nnUNet_project/nnUNet_preprocessed"
NNUNET_RESULTS_ENV = "nnUNet_project/nnUNet_results"