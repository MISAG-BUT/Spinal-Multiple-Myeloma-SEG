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
                                  num_processes_preprocessing=4, num_processes_segmentation_export=4,
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

def create_working_folder_and_convert_to_nifti(patient_name, working_folder, path_to_convCT_folder, path_to_VMI40_folder):
    maybe_mkdir_p(working_folder) 

    # Vytvoření čtečky DICOM série
    reader = sitk.ImageSeriesReader()    
    # Získání seznamu DICOM souborů (série) ve složce
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_convCT_folder)
    reader.SetFileNames(dicom_names)    
    # Načtení 3D obrazu
    image = reader.Execute()    
    # Cesta a jméno výsledného NIfTI souboru
    output_nifti_path = join(working_folder, patient_name + '_conv_0000.nii.gz')    
    # Uložení do NIfTI
    sitk.WriteImage(image, output_nifti_path)

    # Vytvoření čtečky DICOM série
    reader = sitk.ImageSeriesReader()    
    # Získání seznamu DICOM souborů (série) ve složce
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_VMI40_folder)
    reader.SetFileNames(dicom_names)    
    # Načtení 3D obrazu
    image = reader.Execute()    
    # Cesta a jméno výsledného NIfTI souboru
    output_nifti_path = join(working_folder, patient_name + '_monoe_40kev_0000.nii.gz')    
    # Uložení do NIfTI
    sitk.WriteImage(image, output_nifti_path)