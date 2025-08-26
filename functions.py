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
import json
from json import JSONEncoder

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
        tile_step_size=1,
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
    
    # variant 1: give input and output folders
    #predictor.predict_from_files_sequential(input_folder,
    #                                        output_folder,
    #                                        save_probabilities=False, overwrite=False,
    #                                        folder_with_segs_from_prev_stage=None)


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

def create_working_folders_and_convert_to_nifti(patient_name, working_folder, working_folder_conv_CT, working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, working_folder_VMI40, working_folder_VMI40_cropped, working_folder_Segmentation, path_to_convCT_folder, path_to_VMI40_folder):
    maybe_mkdir_p(working_folder) 
    maybe_mkdir_p(working_folder_conv_CT) 
    maybe_mkdir_p(working_folder_conv_CT_in_RAS)
    maybe_mkdir_p(working_folder_conv_CT_in_RAS_cropped)  
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


def get_3d_bounding_box_padding(array, padding=0):
    indices = np.argwhere(array != 0)

    if len(indices) == 0:
        return None

    min_coords = np.min(indices, axis=0)
    max_coords = np.max(indices, axis=0)

    # Přidání paddingu a zajištění, že indexy zůstanou v platném rozsahu
    min_coords = np.maximum(min_coords - padding, 0)
    max_coords = np.minimum(max_coords + padding, np.array(array.shape) - 1)

    return min_coords, max_coords

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def split_convCT_data (working_folder_conv_CT_in_RAS, working_folder_conv_CT_in_RAS_cropped, patient_name):
    # Načíst RAS obraz
    ras_image_path = join(working_folder_conv_CT_in_RAS, patient_name + '_conv_RAS_0000.nii.gz')
    ras_image = sitk.ReadImage(ras_image_path)

    # Převést na numpy pole
    ras_array = sitk.GetArrayFromImage(ras_image)  # tvar (Z, Y, X)

    # Rozdělit podle Z (první osa)
    mid_z = ras_array.shape[0] // 2
    upper_array = ras_array[:mid_z, :, :]   # první půlka (horní část)
    lower_array = ras_array[mid_z:, :, :]   # druhá půlka (spodní část)

    # Vytvořit SITK obrazy
    upper_img = sitk.GetImageFromArray(upper_array)
    lower_img = sitk.GetImageFromArray(lower_array)

    # Zkopírovat metadata
    spacing = ras_image.GetSpacing()
    direction = ras_image.GetDirection()
    origin = ras_image.GetOrigin()

    # Nastavit metadata pro horní půlku
    upper_img.SetSpacing(spacing)
    upper_img.SetDirection(direction)
    upper_img.SetOrigin(origin)

    # Nastavit metadata pro spodní půlku
    lower_img.SetSpacing(spacing)
    lower_img.SetDirection(direction)

    # Posunout origin ve směru Z
    lower_origin = list(origin)
    lower_origin[2] = origin[2] + mid_z * spacing[2]  # posun ve směru Z
    lower_img.SetOrigin(lower_origin)

    maybe_mkdir_p(working_folder_conv_CT_in_RAS_cropped)

    # Uložit nové NIfTI soubory
    sitk.WriteImage(upper_img, join(working_folder_conv_CT_in_RAS_cropped, patient_name + '_conv_RAS_upperZ_0000.nii.gz'))
    sitk.WriteImage(lower_img, join(working_folder_conv_CT_in_RAS_cropped, patient_name + '_conv_RAS_lowerZ_0000.nii.gz'))


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
    

def reorient_spine_segmentation_to_original_space(working_folder_Spine_segmentation_final,working_folder_Spine_segmentation_in_RAS, working_folder_conv_CT_in_RAS):
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

def prepare_data_for_lesion_segmentation(working_folder_Spine_segmentation_final, working_folder_crop_parameters_folder, working_folder_VMI40, working_folder_VMI40_cropped, patient_name):
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

    maybe_mkdir_p(working_folder_crop_parameters_folder)

    coordinates={'orig_size': orig_size, 'min_coords': min_coords,'max_coords': max_coords}    # uložení JSON souboru
    with open(join(working_folder_crop_parameters_folder, patient_name + ".json"), "w") as f:
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


def reorient_lesion_segmentation_to_original_space(working_folder_crop_parameters_folder, working_folder_VMI40, working_folder_Lesion_segmentation_cropped, working_folder_Lesion_segmentation_final, patient_name):
    # %% creation of final lesion segmentation 
    current_path=join(working_folder_crop_parameters_folder,patient_name)
    #open json file and get coordinates
    f=open(current_path+'.json')
    data = json.load(f)
    orig_size=data['orig_size']
    min_coords=data['min_coords']
    max_coords=data['max_coords']
    f.close()

    #VMI 40keV - load nifti info
    vmi_40kev_path=join(working_folder_VMI40, patient_name + '_monoe_40kev_0000.nii.gz')
    vmi_40kev = nib.load(vmi_40kev_path)
    #vmi_40kev_nii_img=vmi_40kev.get_fdata()

    #load predicted data
    #predicted_file=path_to_predicted_segmentations+'/Predikce_myel_061_065/'+t+'.nii.gz'
    predicted_file= join(working_folder_Lesion_segmentation_cropped,patient_name + '.nii.gz')
    predicted_mask = nib.load(predicted_file)
    predicted_mask_data=predicted_mask.get_fdata()
    #create full size image
    full_size_mask=np.zeros(orig_size,dtype=bool)
    full_size_mask[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1]=predicted_mask_data
    #save full size image
    maybe_mkdir_p(working_folder_Lesion_segmentation_final)
    pom_full_size_mask = nib.Nifti1Image(full_size_mask, vmi_40kev.affine, vmi_40kev.header)
    nib.save(pom_full_size_mask, join(working_folder_Lesion_segmentation_final, patient_name + '_lesions_segmentation.nii.gz'))