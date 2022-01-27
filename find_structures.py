from pathlib import Path
import dicom_contour.contour as dcm
import pydicom as dicom
import os
import math
import numpy as np
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt



def get_contour_file(path):
    '''
    Get contour file from the current path
    
    Input: 
    path (str) - the path that contains all the DICOM files
    
    Return: 
    contour_file (str) - the path of the contour file

    We acknowledge the use of code and guidance from Yang Zholin 
    
    '''
    # read in a patient
    DicomFiles = []
    
    for (root, dirs, files) in os.walk(path):
        for filename in files:
            if ".dcm" in filename:
                DicomFiles.append(os.path.join(root, filename))
    
    n = 0
    for FileNames in DicomFiles:
        file = dicom.read_file(FileNames)
        if 'ROIContourSequence' in dir(file): # ROIContourSequence is the specific attribute for struc file
            contour_file = FileNames
            n = n + 1
    
    if n > 1:
        warnings.warn("There are more than one contour files, returning the last one!")
    
    return contour_file


def cartesian2pixels(contour_dataset, path):
    '''
    Return image pixel array and contour label array given a contour dataset and the path 
    that contains the image files
    
    Inputs:
    contour_dataset - DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
    path (str) - the path that contains all the Dicom files
    
    Return:
    ima_array - 2d numpy array of image with pixel intensities
    contour_array - 2d numpy array of contour with labels 0 and 1
    '''
    
    contour_coord = contour_dataset.ContourData
    
    # x, y, z coordinates of the contour in mm
    x0 = contour_coord[len(contour_coord)-3]
    y0 = contour_coord[len(contour_coord)-2]
    z0 = contour_coord[len(contour_coord)-1]
    coord = []
    for i in range(0, len(contour_coord), 3):
        x = contour_coord[i]
        y = contour_coord[i+1]
        z = contour_coord[i+2]
        l = math.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        l = math.ceil(l*2) + 1 # ceil: round toward positive infinity
        for j in range(1, l+1):
            coord.append([(x-x0)*j/l+x0, (y-y0)*j/l+y0, (z-z0)*j/l+z0])
        x0 = x
        y0 = y
        z0 = z
        
    # Extract the image id corresponding to given contour
    ima_file = []
    for (root, dirs, files) in os.walk(path):
        for filename in files:
            if ".dcm" in filename:
                if not "RD" in filename:
                    if not "RS" in filename:
                        ima_file.append(filename)
                        
    correspond_ima_file = []
    for FileNames in ima_file:
        f = dicom.read_file(path + '/' + FileNames)
        if f.SOPInstanceUID == contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID:
            correspond_ima_file.append(FileNames)
            
    
    # Read that Dicom image file
    ima = dicom.read_file(path + '/' + correspond_ima_file[0])
    ima_array = ima.pixel_array
    
    # Physical distance between the center of each pixel
    x_spacing = float(ima.PixelSpacing[0])
    y_spacing = float(ima.PixelSpacing[1])
    
    # The centre of the the upper left voxel
    origin_x = ima.ImagePositionPatient[0]
    origin_y = ima.ImagePositionPatient[1]
    origin_z = ima.ImagePositionPatient[2]
    
    # mapping
    pixel_coords = [(np.round((y - origin_y) / y_spacing), np.round((x-origin_x)/x_spacing)) for x, y, _ in coord]
    
    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_array = csc_matrix((np.ones_like(rows), (rows, cols)), dtype = np.int8, shape = (ima_array.shape[0], ima_array.shape[1])).toarray()
    
    return ima_array, contour_array, correspond_ima_file

def ContourImaArray(contour_file, path, ROIContourSeq = 0):
    '''
    Return the arrays of the contour and the corresponding images given the contour file and 
    the path of the images.

    Inputs:
    contour_file (str) - the path of the contour file
    path (str) - the path that contains all the Dicom files
    ROIContourSeq (int) - shows which sequence of contouring to use, default 5 (Rectum)

    Return:
    contour_ima_arrays (list) - a list that contains pairs of image pixel array and contour label array
    '''
    contour_data = dicom.read_file(contour_file)
    Rectum = contour_data.ROIContourSequence[ROIContourSeq]
    # get contour dataset in a list
    contours = [contour for contour in Rectum.ContourSequence]
    contour_ima_arrays = [cartesian2pixels(cdata, path) for cdata in contours]
    number_of_correspond_ima = len(contours)

    return contour_ima_arrays, number_of_correspond_ima


def fill_contours(arr):
    return np.maximum.accumulate(arr,1) & \
            np.maximum.accumulate(arr[:,::-1],1)[:,::-1]

    first_contour=fill_contours(first_contour)


def get_contour_dict(contour_file, path, ROIContourSeq):
    '''
    Return a dictionary as key: image filename, value: [corresponding image array, corresponding contour array]
    
    Input:
    contour_file (str) - contour file name
    path (str) - path contains all the Dicom files
    ROIContourSeq (int) - shows which sequence of contouring to use, default 5 (Rectum)
    
    Return:
    contour_dict: dictionary with 2d numpy array
    '''
    contour_list, _ = ContourImaArray(contour_file, path, ROIContourSeq)
    
    contour_dict = {}
    for ima_arr, contour_arr, ima_id in contour_list:
        contour_dict[ima_id[0]] = [ima_arr, contour_arr]
        
    return contour_dict