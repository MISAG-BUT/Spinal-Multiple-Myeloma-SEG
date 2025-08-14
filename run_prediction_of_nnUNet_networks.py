# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:30:33 2025

@author: nohel
"""
import sys
import os
sys.path.append(os.path.abspath('F:/Spinal-Multiple-Myeloma-SEG/nnUNet'))
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from functions import create_working_folders_and_convert_to_nifti, run_nnunet_inference
import shutil
import os
join=os.path.join
import pydicom
import SimpleITK as sitk



if __name__ == "__main__":    
    base = 'F:/Spinal-Multiple-Myeloma-SEG/DATA'

    # Find convCT and VMI40 image and convert it to nifti
    ID_patient="S840"
    patient_main_file=join(base,ID_patient)
    path_to_output_folder = base + "_output"
    
    DICOM_folders_all = []
    # Browse files in a directory and save individual folders
    for filename in os.listdir(patient_main_file):
        if filename.startswith('S20'):
            DICOM_folders_all.append(filename)
    print(DICOM_folders_all)
    
    # Load DICOM files
    for DICOM_folder in DICOM_folders_all:
        DICOM_folder_path=join(patient_main_file,DICOM_folder)
        # Loading all DICOM files from the directory, skipping the DIRFILE file
        DICOM_files = [os.path.join(DICOM_folder_path, f) for f in os.listdir(DICOM_folder_path) if f != 'DIRFILE']
        series_description = pydicom.dcmread(DICOM_files[0]).get('SeriesDescription')
        print(series_description)
        if series_description.endswith("_konv"):
            path_to_convCT_folder = DICOM_folder_path
            patient_name = series_description[:8]
        elif series_description=='MonoE 40keV[HU]':
            path_to_VMI40_folder = DICOM_folder_path
        else:
            continue

    name_of_output_folder = patient_name + "_output"
    working_folder = join(path_to_output_folder,name_of_output_folder) 
    working_folder_conv_CT = join(working_folder,'Conv_CT')
    working_folder_conv_CT_in_RAS = join(working_folder,'Conv_CT_in_RAS')
    working_folder_VMI40 = join(working_folder,'VMI40')
    working_folder_VMI40_cropped = join(working_folder,'VMI40_cropped')
    working_folder_Segmentation = join(working_folder,'Segmentation') 
    create_working_folders_and_convert_to_nifti(patient_name, working_folder, working_folder_conv_CT, working_folder_conv_CT_in_RAS, working_folder_VMI40, working_folder_VMI40_cropped, working_folder_Segmentation, path_to_convCT_folder, path_to_VMI40_folder)

    

        # %% Segmentation of spine with nnUNet        
    input_folder = working_folder_conv_CT_in_RAS 
    output_folder = working_folder_Segmentation

    dataset_name = "Dataset800_Spine_segmentation_trained_on_VerSe20"
    trainer_name = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    use_folds = ('all',)

    nnUNet_results = "F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models"

    run_nnunet_inference(nnUNet_results, dataset_name, trainer_name, use_folds, input_folder, output_folder) 











            







