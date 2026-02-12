# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:41:42 2025

@author: nohel
"""
# ------------------------------
# Add nnU-Net repository to Python path
# ------------------------------
import sys
import os
join=os.path.join

# Path to your nnU-Net repository (change this to your actual path)
nnunet_repo_path = r"F:/Code/nnUNet"  

# Add the nnU-Net path to sys.path if not already added
if nnunet_repo_path not in sys.path:
    sys.path.append(nnunet_repo_path)

# path to nnUNet folders - # these paths are not used, just for ignoring of warnings
os.environ["nnUNet_raw"] = r"nnUNet_project/nnUNet_raw"  
os.environ["nnUNet_preprocessed"] = r"nnUNet_project/nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"nnUNet_project/nnUNet_results"

sys.path.append(os.path.abspath('F:/Code/nnUNet'))
#sys.path.append(os.path.abspath('/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG')) #Linux
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed

# rest of imports
from utils import load_DICOM_data_SITK
import SimpleITK as sitk
import pydicom
import napari


if __name__ == "__main__":    
    base='F:/Example_data/DATA/' #path to the dataset folder
    path_to_DICOM_folders = join(base,'MM_DICOM_Dataset') #path to the DICOM folders, which are organized by patient ID and then by series description
    path_to_segmentations = join(base,'MM_NIfTI Segmentation') #path to the segmentation masks, which are organized by patient ID and then by mask type (spine or lesions)
    
    ID_patient="S840"
    patient_main_file=join(path_to_DICOM_folders,ID_patient)
    
    DICOM_folders_all = [
    f for f in os.listdir(patient_main_file)
    if os.path.isdir(os.path.join(patient_main_file, f))
    ]
    print(DICOM_folders_all)

    
    # Load DICOM files
    for DICOM_folder in DICOM_folders_all:
        DICOM_folder_path=join(patient_main_file,DICOM_folder)
        # Loading all DICOM files from the directory, skipping the DIRFILE file
        DICOM_files = [os.path.join(DICOM_folder_path, f) for f in os.listdir(DICOM_folder_path) if f != 'DIRFILE']
        series_description = pydicom.dcmread(DICOM_files[0]).get('SeriesDescription')
        print(series_description)
        if series_description=='Calcium Suppression 25 Index[HU*]':
            CaSupp25_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        elif series_description=='Calcium Suppression 50 Index[HU*]':
            CaSupp50_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        elif series_description=='Calcium Suppression 75 Index[HU*]':
            CaSupp75_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        elif series_description=='Calcium Suppression 100 Index[HU*]':
            CaSupp100_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        elif series_description=='MonoE 40keV[HU]':
            VMI40_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        elif series_description=='MonoE 80keV[HU]':
            VMI80_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        elif series_description=='MonoE 120keV[HU]':
            VMI120_zxy=load_DICOM_data_SITK(DICOM_folder_path)
        else:
            ConvCT_zxy=load_DICOM_data_SITK(DICOM_folder_path)
            patient_name = series_description[:8]
            
    
    #Load Masks
    path_spine_mask = join(path_to_segmentations, patient_name,patient_name + '_spine_segmentation.nii.gz')
    path_lesion_mask = join(path_to_segmentations, patient_name,patient_name + '_lesions_segmentation.nii.gz')
    
    SegmMaskSpine = sitk.GetArrayFromImage(sitk.ReadImage(path_spine_mask))
    SegmMaskLesions = sitk.GetArrayFromImage(sitk.ReadImage(path_lesion_mask))
    
    #Run napari
    v = napari.Viewer()  
    ConvCT_layer = v.add_image(ConvCT_zxy, name='ConvCT')       
    ConvCT_layer.colormap = 'gray'
    ConvCT_layer.blending = 'additive'
    
    VMI40_layer = v.add_image(VMI40_zxy, name='VMI40')       
    VMI40_layer.colormap = 'gray'
    VMI40_layer.blending = 'additive'
    VMI40_layer.visible = False
    
    VMI80_layer = v.add_image(VMI80_zxy, name='VMI80')       
    VMI80_layer.colormap = 'gray'
    VMI80_layer.blending = 'additive'
    VMI80_layer.visible = False
    
    VMI120_layer = v.add_image(VMI120_zxy, name='VMI120')       
    VMI120_layer.colormap = 'gray'
    VMI120_layer.blending = 'additive'
    VMI120_layer.visible = False
        
    CaSupp25_layer = v.add_image(CaSupp25_zxy, name='CaSupp25')       
    CaSupp25_layer.colormap = 'gray'
    CaSupp25_layer.blending = 'additive'
    CaSupp25_layer.visible = False    
    
    CaSupp50_layer = v.add_image(CaSupp50_zxy, name='CaSupp50')       
    CaSupp50_layer.colormap = 'gray'
    CaSupp50_layer.blending = 'additive'
    CaSupp50_layer.visible = False
    
    CaSupp75_layer = v.add_image(CaSupp75_zxy, name='CaSupp75')       
    CaSupp75_layer.colormap = 'gray'
    CaSupp75_layer.blending = 'additive'
    CaSupp75_layer.visible = False
    
    CaSupp100_layer = v.add_image(CaSupp100_zxy, name='CaSupp100')       
    CaSupp100_layer.colormap = 'gray'
    CaSupp100_layer.blending = 'additive'
    CaSupp100_layer.visible = False
    
    SpineMaskLayer = v.add_image(SegmMaskSpine, name='SpineMask')
    SpineMaskLayer.colormap = 'blue'
    SpineMaskLayer.blending = 'additive'
    SpineMaskLayer.opacity = 0.5
    
    LesionMaskLayer = v.add_image(SegmMaskLesions, name='LessionMask')
    LesionMaskLayer.colormap = 'red'
    LesionMaskLayer.blending = 'additive'
    LesionMaskLayer.opacity = 1
    napari.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    