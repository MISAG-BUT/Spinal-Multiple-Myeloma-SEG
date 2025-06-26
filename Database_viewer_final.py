# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:41:42 2025

@author: nohel
"""
import SimpleITK as sitk
import os
join=os.path.join
import pydicom
import napari


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
    
    
#%%
if __name__ == "__main__":
       
    #%%
    
    base='F:/Spinal-Multiple-Myeloma-SEG-Example-Data-Nohel'
    path_to_DICOM_folders = join(base,'MM_DICOM_Dataset')
    path_to_segmentations = join(base,'MM_NIfTI Segmentation')    
    
    ID_patient="S2470"
    patient_main_file=join(path_to_DICOM_folders,ID_patient)
    
    DICOM_folders_all = []
    # Browse files in a directory and save individual folders
    for filename in os.listdir(patient_main_file):
        if filename.startswith('S20'):
            DICOM_folders_all.append(filename)
    print(DICOM_folders_all)
    
    # %% Load DICOM files
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
            
    
    # %% Load Masks
    path_spine_mask = join(path_to_segmentations, patient_name,patient_name + '_spine_segmentation.nii.gz')
    path_lesion_mask = join(path_to_segmentations, patient_name,patient_name + '_lesions_segmentation.nii.gz')
    
    SegmMaskSpine = sitk.GetArrayFromImage(sitk.ReadImage(path_spine_mask))
    SegmMaskLesions = sitk.GetArrayFromImage(sitk.ReadImage(path_lesion_mask))
    
    #%% Run napari
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    