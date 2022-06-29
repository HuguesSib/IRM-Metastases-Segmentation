import h5py
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Get the files of interest in the directory
def get_list_of_files(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    maskFiles = list()
    imgFiles = list()
    bounding_box_files = list()
    bounding_box_coord = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            maskFiles = maskFiles + get_list_of_files(fullPath)[0]
            imgFiles = imgFiles + get_list_of_files(fullPath)[1]
            bounding_box_files = bounding_box_files + get_list_of_files(fullPath)[2]
            bounding_box_coord = bounding_box_coord + get_list_of_files(fullPath)[3]
        else:
            if entry == "reg_mask_cropped_corrected.nii.gz":
                maskFiles.append(fullPath)
            if entry == "reg_img_cropped_bias_correc.nii.gz":
                imgFiles.append(fullPath)
            if entry == "reg_mask_cropped_bounding_box.nii.gz":
                bounding_box_files.append(fullPath)
            if entry == "bounding_boxes_coordinates.csv":
                bounding_box_coord.append(fullPath)


    return maskFiles, imgFiles, bounding_box_files, bounding_box_coord


# Thresholding mask
def threshold_mask(mask, thr = 0):
    # Scale mask from 0 to 255 to 0 - 1
    new_mask = mask / max(mask[np.nonzero(mask)])

    # Threshold to have a 0/1 labelization
    new_mask[new_mask > thr] = 1
    new_mask[new_mask <= thr] = 0
    return new_mask


def write_hdf5(save_path, imgFiles, maskFiles, bounding_box_files, bounding_box_coord):
    hf = h5py.File(save_path, 'a')

    for i in range(len(maskFiles)):
        #Load Segmentation
        mask = maskFiles[i]    
        curr_mask = sitk.ReadImage(mask, sitk.sitkFloat32)
        curr_mask = np.array(sitk.GetArrayViewFromImage(curr_mask))
        #curr_mask = threshold_mask(curr_mask)
        _ = hf.create_dataset(mask, data = curr_mask)
        
        #Load Image
        img = imgFiles[i]
        curr_img = sitk.ReadImage(img, sitk.sitkFloat32)
        #curr_img = N4_bias_correction(curr_img)
        curr_img = np.array(sitk.GetArrayViewFromImage(curr_img))
        _ = hf.create_dataset(img, data = curr_img)

        #Load Bounding boxes
        bb = bounding_box_files[i]
        curr_bb = sitk.ReadImage(bb, sitk.sitkFloat32)
        curr_bb = np.array(sitk.GetArrayViewFromImage(curr_bb))
        _ = hf.create_dataset(bb, data = curr_bb) 

        #Load coordinates
        bbc = bounding_box_coord[i]
        curr_bbc = pd.read_csv(bbc)
        _ = hf.create_dataset(bbc, data = curr_bbc)

        print(mask, 'and\n',img, ' LOADED ...... ', 100*(i+1)/len(maskFiles), '%\n')

    hf.close()

def read_hdf5(path):
    data = []
    group = []

    def func(name, obj):     # function to recursively store all the keys
        if isinstance(obj, h5py.Dataset):
            data.append(name)
        elif isinstance(obj, h5py.Group):
            group.append(name)

    hf = h5py.File(path, 'r')
    hf.visititems(func)

    return hf, data, group


if __name__ == "__main__":
    
    folder  = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project'
    save_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/HuguesandCris/dataset.hdf5'

    maskFiles, imgFiles, bounding_box_files, bounding_box_coord = get_list_of_files(folder)