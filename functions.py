# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:30:33 2025

@author: nohel
"""

#%%
'Library import'
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

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
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

def run_nnunet_inference(nnUNet_results, dataset_name, trainer_name, use_folds, input_folder, output_folder):  
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    ) 
    
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results,dataset_name, trainer_name),
        use_folds = use_folds,
        checkpoint_name = 'checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(input_folder,
                                  output_folder,
                                  save_probabilities=False, overwrite=False,
                                  num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                  folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


def load_DICOM_data_SITK(path_to_series):
    # Load all DICOM images into a 3D volume
    dicom_series_reader = sitk.ImageSeriesReader()
    # Get the list of DICOM file names in the specified directory
    dicom_filenames = dicom_series_reader.GetGDCMSeriesFileNames(path_to_series)
    # Set the list of files to be read
    dicom_series_reader.SetFileNames(dicom_filenames)
    # Read the 3D image from the DICOM series
    image_3d = dicom_series_reader.Execute()
    # Convert the 3D image to a NumPy array (z, y, x)
    img_data = sitk.GetArrayFromImage(image_3d)
    return img_data

def create_working_folders_and_convert_to_nifti(patient_name, working_folder, working_folder_conv_CT, working_folder_conv_CT_in_RAS, working_folder_VMI40, working_folder_VMI40_cropped, working_folder_Segmentation, path_to_convCT_folder, path_to_VMI40_folder):
    maybe_mkdir_p(working_folder) 
    maybe_mkdir_p(working_folder_conv_CT) 
    maybe_mkdir_p(working_folder_conv_CT_in_RAS) 
    maybe_mkdir_p(working_folder_VMI40) 
    maybe_mkdir_p(working_folder_VMI40_cropped) 
    maybe_mkdir_p(working_folder_Segmentation) 

    # Vytvoření čtečky DICOM série
    reader = sitk.ImageSeriesReader()    
    # Získání seznamu DICOM souborů (série) ve složce
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_convCT_folder)
    reader.SetFileNames(dicom_names)    
    # Načtení 3D obrazu
    image = reader.Execute()    
    # Cesta a jméno výsledného NIfTI souboru
    output_nifti_path = join(working_folder_conv_CT, patient_name + '_conv_0000.nii.gz')    
    # Uložení do NIfTI
    sitk.WriteImage(image, output_nifti_path)
    
    shutil.copy(output_nifti_path, join(working_folder_conv_CT_in_RAS, patient_name + '_conv_RAS_0000.nii.gz'))
    reorient_all_images_in_folder_to_ras(working_folder_conv_CT_in_RAS,1)


    # Vytvoření čtečky DICOM série
    reader = sitk.ImageSeriesReader()    
    # Získání seznamu DICOM souborů (série) ve složce
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_VMI40_folder)
    reader.SetFileNames(dicom_names)    
    # Načtení 3D obrazu
    image = reader.Execute()    
    # Cesta a jméno výsledného NIfTI souboru
    output_nifti_path = join(working_folder_VMI40, patient_name + '_monoe_40kev_0000.nii.gz')    
    # Uložení do NIfTI
    sitk.WriteImage(image, output_nifti_path)


    



def reorient_all_images_in_folder_to_ras(folder: str, num_processes: int = 2):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(reorient_to_ras, nii_files)
    p.close()
    p.join()
    
def revert_orientation_on_all_images_in_folder(folder: str, num_processes: int = 1):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(revert_reorientation, nii_files)
    p.close()
    p.join()

def reorient_to_ras(image: str) -> None:
    """
    Will overwrite image!!!
    :param image:
    :return:
    """
    assert image.endswith('.nii.gz')
    origaffine_pkl = image[:-7] + '_originalAffine.pkl'
    if not isfile(origaffine_pkl):
        img = nib.load(image)
        original_affine = img.affine
        original_axcode = nib.aff2axcodes(img.affine)
        img = img.as_reoriented(io_orientation(img.affine))
        new_axcode = nib.aff2axcodes(img.affine)
        print(image.split('/')[-1], 'original axcode', original_axcode, 'now (should be ras)', new_axcode)
        nib.save(img, image)
        save_pickle((original_affine, original_axcode), origaffine_pkl)


def revert_reorientation(image: str) -> None:
    assert image.endswith('.nii.gz')
    expected_pkl = image[:-7] + '_originalAffine.pkl'
    assert isfile(expected_pkl), 'Must have a file with the original affine, as created by ' \
                                 'reorient_to_ras. Expected filename: %s' % \
                                 expected_pkl
    original_affine, original_axcode = load_pickle(image[:-7] + '_originalAffine.pkl')
    img = nib.load(image)
    before_revert = nib.aff2axcodes(img.affine)
    img = img.as_reoriented(io_orientation(original_affine))
    after_revert = nib.aff2axcodes(img.affine)
    print('before revert', before_revert, 'after revert', after_revert)
    restored_affine = img.affine
    assert np.all(np.isclose(original_affine, restored_affine)), 'restored affine does not match original affine, ' \
                                                                 'aborting!'
    nib.save(img, image)
    os.remove(expected_pkl)