# config.py
# Central configuration for all paths and environment variables

# Path to the DICOM folders, organized by patient ID and then by series description
# Example: F:/Example_data/DATA/MM_DICOM_Dataset/S840
PATH_TO_DICOM_FOLDERS = r"F:/Example_data/DATA/MM_DICOM_Dataset"

# Path to the segmentation masks (NIfTI), organized by patient ID and then by mask type (spine or lesions)
# Example: F:/Example_data/DATA/MM_NIfTI Segmentation
PATH_TO_SEGMENTATIONS = r"F:/Example_data/DATA/MM_NIfTI Segmentation"

# Default patient ID (used as folder name)
# Example: S840
ID_PATIENT = "S840"

# Path to the folder containing trained nnU-Net models (should have subfolders for each model)
# Example: F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models/Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together
PATH_TO_NNUNET_RESULTS = r"F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models"

# Default for split option in prediction pipeline
# If True, ConvCT volumes are split along Z-axis to reduce memory usage (recommended for most users)
# If False, the full volume is processed at once (requires very high RAM)
SPLIT_CONVCT_DEFAULT = True

# Path to your local nnU-Net repository (required for importing nnU-Net code)
# Example: F:/Code/nnUNet
NNUNET_REPO_PATH = r"F:/Code/nnUNet"


# nnU-Net environment variables (not required to change, only for suppressing nnU-Net warnings)
# You only need to set NNUNET_REPO_PATH to the location of your downloaded nnUNet repository.
# The variables below are set just to avoid warnings and are not used by the pipeline logic.
NNUNET_RAW = "nnUNet_project/nnUNet_raw"
NNUNET_PREPROCESSED = "nnUNet_project/nnUNet_preprocessed"
NNUNET_RESULTS_ENV = "nnUNet_project/nnUNet_results"

