#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:04:25 2022

@author: hsibille
"""
import os
import numpy as np
from bias_correction import N4_bias_correction



# Get the files of interest in the directory
def get_list_of_files(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    maskFiles = list()
    imgFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            maskFiles = maskFiles + get_list_of_files(fullPath)[0]
            imgFiles = imgFiles + get_list_of_files(fullPath)[1]
        else:
            if entry == "reg_mask_cropped_corrected.nii.gz":
                maskFiles.append(fullPath)
            if entry == "reg_img_cropped_bias_correc.nii.gz":
                imgFiles.append(fullPath)
    return maskFiles, imgFiles


# Thresholding mask
def threshold_mask(mask, thr = 0):
    # Scale mask from 0 to 255 to 0 - 1
    new_mask = mask / max(mask[np.nonzero(mask)])

    # Threshold to have a 0/1 labelization
    new_mask[new_mask > thr] = 1
    new_mask[new_mask <= thr] = 0
    return new_mask


