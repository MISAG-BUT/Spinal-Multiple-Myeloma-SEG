# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:30:33 2025

@author: nohel
"""
import sys
import os
#sys.path.append(os.path.abspath('F:/Spinal-Multiple-Myeloma-SEG/nnUNet'))#windows
sys.path.append(os.path.abspath('/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG')) #Linux
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from functions import create_working_folders_and_convert_to_nifti, run_nnunet_inference, maybe_mkdir_p, revert_orientation_on_all_images_in_folder, get_3d_bounding_box_padding, NumpyArrayEncoder, split_convCT_data
import shutil
import os
join=os.path.join
import pydicom
import SimpleITK as sitk
import nibabel as nib
import json
import numpy as np

def merge_data(working_output_folder_for_spine_segmentation, working_folder_Spine_segmentation_in_RAS, patient_name):
    # Cesty k dílčím obrazům
    upper_path = join(working_output_folder_for_spine_segmentation, patient_name + '_conv_RAS_upperZ.nii.gz')
    lower_path = join(working_output_folder_for_spine_segmentation, patient_name + '_conv_RAS_lowerZ.nii.gz')

    # Načtení obrazů
    upper_img = sitk.ReadImage(upper_path)
    lower_img = sitk.ReadImage(lower_path)

    # Konverze na numpy
    upper_arr = sitk.GetArrayFromImage(upper_img)  # (Z1, Y, X)
    lower_arr = sitk.GetArrayFromImage(lower_img)  # (Z2, Y, X)

    # Spojení podél osy Z
    merged_arr = np.concatenate([upper_arr, lower_arr], axis=0)

    # Nový SimpleITK obraz
    merged_img = sitk.GetImageFromArray(merged_arr)

    # Metadata ze začátku (horní blok má správný origin jako původní)
    merged_img.SetSpacing(upper_img.GetSpacing())
    merged_img.SetDirection(upper_img.GetDirection())
    merged_img.SetOrigin(upper_img.GetOrigin())

    maybe_mkdir_p(working_folder_Spine_segmentation_in_RAS)

    # Uložit výsledek
    sitk.WriteImage(merged_img, join(working_folder_Spine_segmentation_in_RAS, patient_name + '_conv_RAS.nii.gz'))
    

def reorient_data_to_original_space(working_folder_Spine_segmentation_final,working_folder_Spine_segmentation_in_RAS):
    maybe_mkdir_p(working_folder_Spine_segmentation_final)

    for filename in os.listdir(working_folder_Spine_segmentation_in_RAS):
        if filename.endswith(".nii.gz"):
            src_path = os.path.join(working_folder_Spine_segmentation_in_RAS, filename)
            dst_path = os.path.join(working_folder_Spine_segmentation_final, filename[:8] + '_spine_segmentation.nii.gz')
            shutil.copy(src_path, dst_path)

    for filename in os.listdir(working_folder_conv_CT_in_RAS):
        if filename.endswith("originalAffine.pkl"):
            src_path = os.path.join(working_folder_conv_CT_in_RAS, filename)
            dst_path = os.path.join(working_folder_Spine_segmentation_final, filename[:8] + '_spine_segmentation_originalAffine.pkl')
            shutil.copy(src_path, dst_path)

    revert_orientation_on_all_images_in_folder(working_folder_Spine_segmentation_final,1)

if __name__ == "__main__":    
    base = 'F:/Spinal-Multiple-Myeloma-SEG/DATA' #windows
    #base = '/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG/DATA' #Linux
    split_data = True

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
    working_folder_conv_CT_in_RAS_cropped = join(working_folder,'Conv_CT_in_RAS_cropped')
    working_folder_VMI40 = join(working_folder,'VMI40')
    working_folder_VMI40_cropped = join(working_folder,'VMI40_cropped')
    working_folder_Segmentation = join(working_folder,'Segmentation') 
    #create_working_folders_and_convert_to_nifti(patient_name, working_folder, working_folder_conv_CT, working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, working_folder_VMI40, working_folder_VMI40_cropped, working_folder_Segmentation, path_to_convCT_folder, path_to_VMI40_folder)


    working_folder_Spine_segmentation_final = join(working_folder_Segmentation,'Spine_segmentation_final') 
    working_folder_Spine_segmentation_in_RAS = join(working_folder_Segmentation,'Spine_segmentation_in_RAS') 
    working_folder_Spine_segmentation_in_RAS_cropped = join(working_folder_Segmentation,'Spine_segmentation_in_RAS_cropped')

    split_data = True
    if split_data:
        #split_convCT_data(working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, patient_name)
        working_input_folder_for_spine_segmentation = working_folder_conv_CT_in_RAS_cropped
        working_output_folder_for_spine_segmentation = working_folder_Spine_segmentation_in_RAS_cropped
    else:
        working_input_folder_for_spine_segmentation = working_folder_conv_CT_in_RAS
        working_output_folder_for_spine_segmentation = working_folder_Spine_segmentation_in_RAS

    
    # %% Segmentation of spine with nnUNet        
    input_folder = working_input_folder_for_spine_segmentation
    output_folder = working_output_folder_for_spine_segmentation
    dataset_name = "Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together"
    trainer_name = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    use_folds = ('all',)
    nnUNet_results = "F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models" # windows
    #nnUNet_results = "/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG_nnUNet_models" # Linux
    #run_nnunet_inference(nnUNet_results, dataset_name, trainer_name, use_folds, input_folder, output_folder) 

    merge_data(working_output_folder_for_spine_segmentation, working_folder_Spine_segmentation_in_RAS, patient_name)

    # %% reorient spine segmentation to original space
    reorient_data_to_original_space(working_folder_Spine_segmentation_final,working_folder_Spine_segmentation_in_RAS)







    for filename in os.listdir(working_folder_Spine_segmentation_final):
        if filename.endswith(".nii.gz"):
            spine_segmentation_path = os.path.join(working_folder_Spine_segmentation_final, filename)
            patient_name = filename[:8]
            break 

    spine_segmentation = nib.load(spine_segmentation_path)
    spine_segmentation_nii_img = spine_segmentation.get_fdata()
    spine_segmentation_nii_img = spine_segmentation_nii_img.astype(bool) # maska obratlu nnUNet binarne 
    
    min_coords, max_coords = get_3d_bounding_box_padding(spine_segmentation_nii_img, 5)  # nalezeni BB pro 3D     
    orig_size=spine_segmentation_nii_img.shape

    crop_parameters_folder = join(working_folder_Segmentation, "crop_parameters_folder")
    maybe_mkdir_p(crop_parameters_folder)

    coordinates={'orig_size': orig_size, 'min_coords': min_coords,'max_coords': max_coords}    # uložení JSON souboru
    with open(join(crop_parameters_folder, patient_name + ".json"), "w") as f:
        json.dump(coordinates, f, cls=NumpyArrayEncoder)

    cut_nii_img = spine_segmentation_nii_img[min_coords[0]:max_coords[0]+1,
                      min_coords[1]:max_coords[1]+1,
                      min_coords[2]:max_coords[2]+1].copy()

    pom_seg_nn_unet_binar = nib.Nifti1Image(cut_nii_img, spine_segmentation.affine, spine_segmentation.header)
    nib.save(pom_seg_nn_unet_binar, join(working_folder_VMI40_cropped, patient_name + "_0001.nii.gz")) 


    #VMI 40keV
    vmi_40kev_path=join(working_folder_VMI40, patient_name + '_monoe_40kev_0000.nii.gz')
    vmi_40kev = nib.load(vmi_40kev_path)
    vmi_40kev_nii_img=vmi_40kev.get_fdata()
    cut_nii_img = vmi_40kev_nii_img[min_coords[0]:max_coords[0]+1,
                    min_coords[1]:max_coords[1]+1,
                    min_coords[2]:max_coords[2]+1].copy()

    pom_seg_nn_unet_binar = nib.Nifti1Image(cut_nii_img, spine_segmentation.affine, spine_segmentation.header)
    nib.save(pom_seg_nn_unet_binar, join(working_folder_VMI40_cropped, patient_name + "_0000.nii.gz")) 



    # %% Segmentation of spine with nnUNet        
    input_folder = working_folder_VMI40_cropped 
    output_folder = join(working_folder_Segmentation,'Lesion_segmentation_cropped')
    dataset_name = "Dataset652_MM_lesions_seg_nnUNet_v_2_2_VMI_40"
    trainer_name = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    use_folds = ('all',)
    run_nnunet_inference(nnUNet_results, dataset_name, trainer_name, use_folds, input_folder, output_folder) 


    # %% creation of final lesion segmentation 
    current_path=join(crop_parameters_folder,patient_name)
    #open json file and get coordinates
    f=open(current_path+'.json')
    data = json.load(f)
    orig_size=data['orig_size']
    min_coords=data['min_coords']
    max_coords=data['max_coords']
    f.close()


    #VMI 40keV - load nifti info
    vmi_40kev = nib.load(vmi_40kev_path)
    vmi_40kev_nii_img=vmi_40kev.get_fdata()

    #load predicted data
    #predicted_file=path_to_predicted_segmentations+'/Predikce_myel_061_065/'+t+'.nii.gz'
    predicted_file= join(output_folder,patient_name + '.nii.gz')
    predicted_mask = nib.load(predicted_file)
    predicted_mask_data=predicted_mask.get_fdata()
    #create full size image
    full_size_mask=np.zeros(orig_size,dtype=bool)
    full_size_mask[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1]=predicted_mask_data
    #save full size image
    Lesion_segmentation_folder = join(working_folder_Segmentation,'Lesion_segmentation')
    maybe_mkdir_p(Lesion_segmentation_folder)
    pom_full_size_mask = nib.Nifti1Image(full_size_mask, vmi_40kev.affine, vmi_40kev.header)
    nib.save(pom_full_size_mask, join(Lesion_segmentation_folder, patient_name + '_lesions_segmentation.nii.gz'))






            







