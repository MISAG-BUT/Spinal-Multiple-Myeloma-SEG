# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:30:33 2025

@author: nohel
"""
import sys
import os
join=os.path.join
# path to nnUNet folders - # these paths are not used, just for ignoring of warnings
os.environ["nnUNet_raw"] = r"nnUNet_project\nnUNet_raw"  
os.environ["nnUNet_preprocessed"] = r"nnUNet_project\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"nnUNet_project\nnUNet_results"

#sys.path.append(os.path.abspath('F:/Spinal-Multiple-Myeloma-SEG/nnUNet'))#windows
sys.path.append(os.path.abspath('/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG')) #Linux
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from utils import * 

def main(base, ID_patient, nnUNet_results, split_data=True):
    patient_main_file=join(base,ID_patient)
    path_to_output_folder = base + "_output"
    

    # Find convCT and VMI40 image and convert it to nifti
    patient_name, path_to_convCT_folder, path_to_VMI40_folder = find_convCT_and_VMI40_at_DICOM_folder(patient_main_file)
    
    
    print('Creation of working folders and conversion to nifti - Start')
    name_of_output_folder = patient_name + "_output"
    working_folder = join(path_to_output_folder,name_of_output_folder) 
    working_folder_conv_CT = join(working_folder,'Conv_CT')
    working_folder_conv_CT_in_RAS = join(working_folder,'Conv_CT_in_RAS')
    working_folder_conv_CT_in_RAS_cropped = join(working_folder,'Conv_CT_in_RAS_cropped')
    working_folder_VMI40 = join(working_folder,'VMI40')
    working_folder_VMI40_cropped = join(working_folder,'VMI40_cropped')
    working_folder_Segmentation = join(working_folder,'Segmentation') 
    create_working_folders_and_convert_to_nifti(patient_name, working_folder, working_folder_conv_CT, working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, working_folder_VMI40, working_folder_VMI40_cropped, working_folder_Segmentation, path_to_convCT_folder, path_to_VMI40_folder)
    print('Creation of working folders and conversion to nifti - Done')




    print('Spine segmentation - Start')
    print('Spine segmentation - Preparation of data for segmentation')
    working_folder_Spine_segmentation_final = join(working_folder_Segmentation,'Spine_segmentation_final') 
    working_folder_Spine_segmentation_in_RAS = join(working_folder_Segmentation,'Spine_segmentation_in_RAS') 
    working_folder_Spine_segmentation_in_RAS_cropped = join(working_folder_Segmentation,'Spine_segmentation_in_RAS_cropped')

    if split_data:
        split_convCT_data(working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, patient_name)
        working_input_folder_for_spine_segmentation = working_folder_conv_CT_in_RAS_cropped
        working_output_folder_for_spine_segmentation = working_folder_Spine_segmentation_in_RAS_cropped
    else:
        working_input_folder_for_spine_segmentation = working_folder_conv_CT_in_RAS
        working_output_folder_for_spine_segmentation = working_folder_Spine_segmentation_in_RAS

    print('Spine segmentation - Start of prediction with nnUNet')
    # %% Segmentation of spine with nnUNet        
    input_folder = working_input_folder_for_spine_segmentation
    output_folder = working_output_folder_for_spine_segmentation
    dataset_name = "Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together"
    trainer_name = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    use_folds = ('all',)    
    run_nnunet_inference(nnUNet_results, dataset_name, trainer_name, use_folds, input_folder, output_folder) 
    
    print('Spine segmentation - End of prediction with nnUNet')

    print('Spine segmentation - reorientation of segmentation to original space ')
    if split_data:
        print('Spine segmentation - Merging of data ')
        merge_data(working_output_folder_for_spine_segmentation, working_folder_Spine_segmentation_in_RAS, patient_name)

    # %% reorient spine segmentation to original space
    reorient_spine_segmentation_to_original_space(working_folder_Spine_segmentation_final,working_folder_Spine_segmentation_in_RAS, working_folder_conv_CT_in_RAS)
    print('Spine segmentation - Done ')



    print('Lesion segmentation - Start ')

    print('Lesion segmentation - Preparation of data for segmentation')
    working_folder_crop_parameters_folder = join(working_folder_Segmentation, "crop_parameters_folder")
    working_folder_Lesion_segmentation_cropped = join(working_folder_Segmentation, "Lesion_segmentation_cropped")
    working_folder_Lesion_segmentation_final = join(working_folder_Segmentation, "Lesion_segmentation_final")

    prepare_data_for_lesion_segmentation(working_folder_Spine_segmentation_final, working_folder_crop_parameters_folder, working_folder_VMI40, working_folder_VMI40_cropped, patient_name)
    
    print('Lession segmentation - Start of prediction with nnUNet')

    # %% Segmentation of spine with nnUNet        
    input_folder = working_folder_VMI40_cropped 
    output_folder = working_folder_Lesion_segmentation_cropped
    dataset_name = "Dataset710_MM_Lesion_seg_just_VMI_40"
    trainer_name = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    use_folds = ('all',)
    run_nnunet_inference(nnUNet_results, dataset_name, trainer_name, use_folds, input_folder, output_folder)

    print('Lession segmentation - End of prediction with nnUNet')

    print('Lession segmentation - reorientation of segmentation to original space ')

    # %% creation of final lesion segmentation 
    reorient_lesion_segmentation_to_original_space(working_folder_crop_parameters_folder, working_folder_VMI40, working_folder_Lesion_segmentation_cropped, working_folder_Lesion_segmentation_final, patient_name)
    print('Lession segmentation - Done ')


if __name__ == "__main__":    
    # Set paths to data and models
    #base = 'F:/Spinal-Multiple-Myeloma-SEG/DATA' #windows
    base = '/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG/DATA' #Linux

    #nnUNet_results = "F:/Spinal-Multiple-Myeloma-SEG_nnUNet_models" # windows
    nnUNet_results = "/mnt/md0/nohel/Spinal-Multiple-Myeloma-SEG_nnUNet_models" # Linux

    
    ID_patient="S840"   # name of folder with all DICOM folders for one patient


   
    split_data = True   #If set to True, the original image is split into two parts along the Z-axis 
                        # to reduce computational cost, and later recombined after processing.  
                        # If set to False, prediction is performed on the entire image at once, 
                        # which requires ~256 GB of RAM.

    
    main(base, ID_patient, nnUNet_results, split_data=True)




            







