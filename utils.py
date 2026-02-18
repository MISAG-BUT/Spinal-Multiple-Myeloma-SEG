# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:30:33 2025

@author: nohel

Unified utils for DICOM loading, nnU-Net inference, reorientation, 
cropping, and segmentation preparation.
"""

# =============================================================================
# Library imports
# =============================================================================
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import os
join = os.path.join
import SimpleITK as sitk
from typing import List
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
from nibabel import io_orientation
import numpy as np
import shutil
import json
from json import JSONEncoder
import pydicom

# =============================================================================
# File & Folder Utilities
# =============================================================================
def maybe_mkdir_p(directory: str) -> None:
    """Create folder if it does not exist."""
    os.makedirs(directory, exist_ok=True)

def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    """List files in a folder optionally filtered by prefix/suffix and sorted."""
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res  



# =============================================================================
# DICOM Utilities
# =============================================================================
def find_convCT_and_VMI40_at_DICOM_folder(patient_main_file):
    """
    Search for conventional CT (_konv) and MonoE 40keV series in a patient's folder.
    Returns patient name, path to convCT, and path to VMI40 series.
    """
    DICOM_folders_all = [
        f for f in os.listdir(patient_main_file)
        if os.path.isdir(os.path.join(patient_main_file, f))
    ]
    print(DICOM_folders_all)    
    print('Searching for convCT and VMI40 data...')

    for DICOM_folder in DICOM_folders_all:
        DICOM_folder_path = join(patient_main_file, DICOM_folder)
        # Load all DICOM files, skipping DIRFILE
        DICOM_files = [os.path.join(DICOM_folder_path, f) for f in os.listdir(DICOM_folder_path) if f != 'DIRFILE']
        series_description = pydicom.dcmread(DICOM_files[0]).get('SeriesDescription')

        if series_description.endswith("_konv"):
            path_to_convCT_folder = DICOM_folder_path
            patient_name = series_description[:8]
        elif series_description == 'MonoE 40keV[HU]':
            path_to_VMI40_folder = DICOM_folder_path
        else:
            continue

    return patient_name, path_to_convCT_folder, path_to_VMI40_folder

def load_DICOM_data_SITK(path_to_series):
    """Load DICOM series into a 3D NumPy array (z, y, x)."""
    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_filenames = dicom_series_reader.GetGDCMSeriesFileNames(path_to_series)
    dicom_series_reader.SetFileNames(dicom_filenames)
    image_3d = dicom_series_reader.Execute()
    return sitk.GetArrayFromImage(image_3d)

# =============================================================================
# nnU-Net inference
# =============================================================================

def run_nnunet_inference(
    path_to_nnunet_results: str,
    dataset_name: str,
    trainer_name: str,
    use_folds,
    input_folder: str,
    output_folder: str,
):
    """
    Run nnU-Net inference directly from Python (no CLI).

    Parameters
    ----------
    path_to_nnunet_results : str
        Path to the trained nnU-Net results folder.
    dataset_name : str
        Dataset identifier used during training.
    trainer_name : str
        Name of the trainer/folder containing the checkpoint.
    use_folds : list
        List of fold indices to use for prediction.
    input_folder : str
        Folder containing input images to segment.
    output_folder : str
        Folder where output segmentations will be saved.

    Notes
    -----
    Two variants are provided:
    1) `predict_from_files` uses multiprocessing (faster for many images).
    2) `predict_from_files_sequential` disables multiprocessing (safer on Windows, slower).
    """
    # -------------------------------------------------------------------------
    # Instantiate the nnUNetPredictor
    # -------------------------------------------------------------------------
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    ) 

    # -------------------------------------------------------------------------
    # Initialize the network architecture and load the trained checkpoint
    # -------------------------------------------------------------------------
    predictor.initialize_from_trained_model_folder(
        join(path_to_nnunet_results, dataset_name, trainer_name),
        use_folds=use_folds,
        checkpoint_name='checkpoint_final.pth',
    )

    # -------------------------------------------------------------------------
    # Variant 1: Default nnU-Net batch prediction (uses multiprocessing)
    # Works best when predicting multiple images at once
    # -------------------------------------------------------------------------
    
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    

    # -------------------------------------------------------------------------
    # Variant 2: Sequential prediction (no multiprocessing)
    # Safer on Windows or problematic setups, but slower
    # -------------------------------------------------------------------------
    '''
    predictor.predict_from_files_sequential(
        input_folder,
        output_folder,
        save_probabilities=False,
        overwrite=True,
        folder_with_segs_from_prev_stage=None
    )
    '''
# =============================================================================
# Working folders & conversion
# =============================================================================
def create_working_folders_and_convert_to_nifti(
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
):
    """
    Create all working folders for a patient and convert DICOM to NIfTI (convCT and VMI40)
    with RAS-oriented copies.
    """
    maybe_mkdir_p(working_folder) 
    maybe_mkdir_p(working_folder_conv_CT) 
    maybe_mkdir_p(working_folder_conv_CT_in_RAS)
    maybe_mkdir_p(working_folder_conv_CT_in_RAS_cropped)  
    maybe_mkdir_p(working_folder_VMI40) 
    maybe_mkdir_p(working_folder_VMI40_cropped) 
    maybe_mkdir_p(working_folder_Segmentation) 
    maybe_mkdir_p(working_folder_Spine_segmentation_final) 
    maybe_mkdir_p(working_folder_Spine_segmentation_in_RAS) 
    maybe_mkdir_p(working_folder_Spine_segmentation_in_RAS_cropped) 
    maybe_mkdir_p(working_folder_crop_parameters_folder) 
    maybe_mkdir_p(working_folder_Lesion_segmentation_cropped) 
    maybe_mkdir_p(working_folder_Lesion_segmentation_final) 

    # Convert convCT
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_convCT_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    convCT_nifti = join(working_folder_conv_CT, patient_name + '_conv_0000.nii.gz')
    sitk.WriteImage(image, convCT_nifti)
    shutil.copy(convCT_nifti, join(working_folder_conv_CT_in_RAS, patient_name + '_conv_RAS_0000.nii.gz'))
    reorient_all_images_in_folder_to_ras(working_folder_conv_CT_in_RAS, num_processes=1)

    # Convert VMI40
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_VMI40_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    vmi40_nifti = join(working_folder_VMI40, patient_name + '_monoe_40kev_0000.nii.gz')
    sitk.WriteImage(image, vmi40_nifti)


# =============================================================================
# NIfTI Reorientation
# =============================================================================
def reorient_to_ras(image: str) -> None:
    """
    Reorient NIfTI image to RAS coordinates.
    Will overwrite the input image and save original affine.
    """
    assert image.endswith('.nii.gz')
    origaffine_pkl = image[:-7] + '_originalAffine.pkl'
    if not isfile(origaffine_pkl):
        img = nib.load(image)
        original_affine = img.affine
        original_axcode = nib.aff2axcodes(img.affine)
        img = img.as_reoriented(io_orientation(img.affine))
        new_axcode = nib.aff2axcodes(img.affine)
        print(image.split('/')[-1], 'original axcode', original_axcode, 'now (should be RAS)', new_axcode)
        nib.save(img, image)
        save_pickle((original_affine, original_axcode), origaffine_pkl)

def revert_reorientation(image: str) -> None:
    """Restore original orientation from saved affine."""
    assert image.endswith('.nii.gz')
    expected_pkl = image[:-7] + '_originalAffine.pkl'
    assert isfile(expected_pkl), f'Missing original affine file: {expected_pkl}'
    original_affine, original_axcode = load_pickle(expected_pkl)
    img = nib.load(image)
    img = img.as_reoriented(io_orientation(original_affine))
    restored_affine = img.affine
    assert np.all(np.isclose(original_affine, restored_affine)), 'Affine mismatch!'
    nib.save(img, image)
    os.remove(expected_pkl)

def reorient_all_images_in_folder_to_ras(folder: str, num_processes: int = 2):
    """Reorient all NIfTI images in a folder to RAS using multiprocessing."""
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(reorient_to_ras, nii_files)
    p.close()
    p.join()

def revert_orientation_on_all_images_in_folder(folder: str, num_processes: int = 1):
    """Restore original orientation for all images in a folder."""
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(revert_reorientation, nii_files)
    p.close()
    p.join()

# =============================================================================
# Data Cropping & Bounding Box
# =============================================================================
def get_3d_bounding_box_padding(array, padding=0):
    """Get 3D bounding box of non-zero voxels, optionally with padding."""
    indices = np.argwhere(array != 0)
    if len(indices) == 0:
        return None
    min_coords = np.maximum(np.min(indices, axis=0) - padding, 0)
    max_coords = np.minimum(np.max(indices, axis=0) + padding, np.array(array.shape) - 1)
    return min_coords, max_coords

class NumpyArrayEncoder(JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# =============================================================================
# Workflow: split convCT, merge, lesion/spine preparation
# =============================================================================
def split_convCT_data(working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, patient_name):
    """Split convCT image in RAS space along Z axis into upper/lower halves."""
    ras_image_path = join(working_folder_conv_CT_in_RAS, patient_name + '_conv_RAS_0000.nii.gz')
    ras_image = sitk.ReadImage(ras_image_path)
    ras_array = sitk.GetArrayFromImage(ras_image)
    mid_z = ras_array.shape[0] // 2
    upper_img = sitk.GetImageFromArray(ras_array[:mid_z])
    lower_img = sitk.GetImageFromArray(ras_array[mid_z:])
    spacing, direction, origin = ras_image.GetSpacing(), ras_image.GetDirection(), ras_image.GetOrigin()
    upper_img.SetSpacing(spacing); upper_img.SetDirection(direction); upper_img.SetOrigin(origin)
    lower_origin = list(origin); lower_origin[2] = origin[2] + mid_z * spacing[2]
    lower_img.SetSpacing(spacing); lower_img.SetDirection(direction); lower_img.SetOrigin(lower_origin)
    maybe_mkdir_p(working_folder_conv_CT_in_RAS_cropped)
    sitk.WriteImage(upper_img, join(working_folder_conv_CT_in_RAS_cropped, patient_name + '_conv_RAS_upperZ_0000.nii.gz'))
    sitk.WriteImage(lower_img, join(working_folder_conv_CT_in_RAS_cropped, patient_name + '_conv_RAS_lowerZ_0000.nii.gz'))

def merge_data(working_output_folder_for_spine_segmentation, working_folder_Spine_segmentation_in_RAS, patient_name):
    """Merge upper/lower Z segmentations into a single 3D image."""
    upper_img = sitk.ReadImage(join(working_output_folder_for_spine_segmentation, patient_name + '_conv_RAS_upperZ.nii.gz'))
    lower_img = sitk.ReadImage(join(working_output_folder_for_spine_segmentation, patient_name + '_conv_RAS_lowerZ.nii.gz'))
    merged_arr = np.concatenate([sitk.GetArrayFromImage(upper_img), sitk.GetArrayFromImage(lower_img)], axis=0)
    merged_img = sitk.GetImageFromArray(merged_arr)
    merged_img.SetSpacing(upper_img.GetSpacing())
    merged_img.SetDirection(upper_img.GetDirection())
    merged_img.SetOrigin(upper_img.GetOrigin())
    maybe_mkdir_p(working_folder_Spine_segmentation_in_RAS)
    sitk.WriteImage(merged_img, join(working_folder_Spine_segmentation_in_RAS, patient_name + '_conv_RAS.nii.gz'))

def reorient_spine_segmentation_to_original_space(
    working_folder_Spine_segmentation_final,
    working_folder_Spine_segmentation_in_RAS,
    working_folder_conv_CT_in_RAS
):
    """
    Revert reoriented spine segmentation back to original space.

    Steps:
    1. Copy all spine segmentations from RAS folder to final folder, renaming them.
    2. Copy the original affine info from conv_CT RAS folder to final folder.
    3. Revert all images to their original orientation using stored affines.
    """
    maybe_mkdir_p(working_folder_Spine_segmentation_final)

    # Copy spine segmentations and rename
    for filename in os.listdir(working_folder_Spine_segmentation_in_RAS):
        if filename.endswith(".nii.gz"):
            src_path = os.path.join(working_folder_Spine_segmentation_in_RAS, filename)
            dst_path = os.path.join(
                working_folder_Spine_segmentation_final,
                filename[:8] + '_spine_segmentation.nii.gz'
            )
            shutil.copy(src_path, dst_path)

    # Copy original affine pickle files and rename
    for filename in os.listdir(working_folder_conv_CT_in_RAS):
        if filename.endswith("originalAffine.pkl"):
            src_path = os.path.join(working_folder_conv_CT_in_RAS, filename)
            dst_path = os.path.join(
                working_folder_Spine_segmentation_final,
                filename[:8] + '_spine_segmentation_originalAffine.pkl'
            )
            shutil.copy(src_path, dst_path)

    # Revert orientation for all images in final folder
    revert_orientation_on_all_images_in_folder(working_folder_Spine_segmentation_final, 1)


def prepare_data_for_lesion_segmentation(
    working_folder_Spine_segmentation_final,
    working_folder_crop_parameters_folder,
    working_folder_VMI40,
    working_folder_VMI40_cropped,
    patient_name
):
    """
    Prepare lesion segmentation data by cropping images based on spine segmentation bounding box.

    Steps:
    1. Load the spine segmentation NIfTI to determine bounding box of the spine.
    2. Save bounding box coordinates and original size as JSON for later use.
    3. Crop the spine mask to bounding box and save.
    4. Crop the VMI40 image to the same bounding box and save.
    """
    # Find spine segmentation file
    for filename in os.listdir(working_folder_Spine_segmentation_final):
        if filename.endswith(".nii.gz"):
            spine_segmentation_path = os.path.join(working_folder_Spine_segmentation_final, filename)
            patient_name = filename[:8]  # update patient_name based on file
            break

    # Load spine segmentation and convert to binary mask
    spine_segmentation = nib.load(spine_segmentation_path)
    spine_segmentation_nii_img = spine_segmentation.get_fdata().astype(bool)

    # Compute bounding box with padding
    min_coords, max_coords = get_3d_bounding_box_padding(spine_segmentation_nii_img, padding=5)
    orig_size = spine_segmentation_nii_img.shape

    maybe_mkdir_p(working_folder_crop_parameters_folder)

    # Save crop parameters to JSON
    coordinates = {'orig_size': orig_size, 'min_coords': min_coords, 'max_coords': max_coords}
    with open(join(working_folder_crop_parameters_folder, patient_name + ".json"), "w") as f:
        json.dump(coordinates, f, cls=NumpyArrayEncoder)

    # Crop spine segmentation to bounding box
    cut_nii_img = spine_segmentation_nii_img[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ].copy()

    nib.save(
        nib.Nifti1Image(cut_nii_img, spine_segmentation.affine, spine_segmentation.header),
        join(working_folder_VMI40_cropped, patient_name + "_0001.nii.gz")
    )

    # Crop VMI40 image to same bounding box
    vmi_40kev_path = join(working_folder_VMI40, patient_name + '_monoe_40kev_0000.nii.gz')
    vmi_40kev = nib.load(vmi_40kev_path)
    vmi_40kev_nii_img = vmi_40kev.get_fdata()

    cut_nii_img = vmi_40kev_nii_img[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ].copy()

    nib.save(
        nib.Nifti1Image(cut_nii_img, spine_segmentation.affine, spine_segmentation.header),
        join(working_folder_VMI40_cropped, patient_name + "_0000.nii.gz")
    )


def reorient_lesion_segmentation_to_original_space(
    working_folder_crop_parameters_folder,
    working_folder_VMI40,
    working_folder_Lesion_segmentation_cropped,
    working_folder_Lesion_segmentation_final,
    patient_name
):
    """
    Revert cropped lesion segmentation back to full original space.

    Steps:
    1. Load JSON file to retrieve original image size and bounding box coordinates.
    2. Load predicted cropped lesion mask.
    3. Create full-size array and insert cropped mask into correct location.
    4. Save full-size lesion mask as NIfTI with original VMI40 affine/header.
    """
    # Load crop parameters
    with open(join(working_folder_crop_parameters_folder, patient_name + '.json')) as f:
        data = json.load(f)
    orig_size = data['orig_size']
    min_coords = data['min_coords']
    max_coords = data['max_coords']

    # Load original VMI40 image for affine and header
    vmi_40kev_path = join(working_folder_VMI40, patient_name + '_monoe_40kev_0000.nii.gz')
    vmi_40kev = nib.load(vmi_40kev_path)

    # Load predicted lesion mask
    predicted_file = join(working_folder_Lesion_segmentation_cropped, patient_name + '.nii.gz')
    predicted_mask = nib.load(predicted_file).get_fdata()

    # Create full-size mask and place cropped prediction into correct location
    full_size_mask = np.zeros(orig_size, dtype=bool)
    full_size_mask[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ] = predicted_mask

    # Save full-size lesion mask
    maybe_mkdir_p(working_folder_Lesion_segmentation_final)
    nib.save(
        nib.Nifti1Image(full_size_mask, vmi_40kev.affine, vmi_40kev.header),
        join(working_folder_Lesion_segmentation_final, patient_name + '_lesions_segmentation.nii.gz')
    )
