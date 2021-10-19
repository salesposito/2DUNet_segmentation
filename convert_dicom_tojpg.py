import dicom
from scipy.ndimage.interpolation import zoom
import imageio
from PIL import Image
import glob
import os
import dicom
import numpy as np
import pydicom
import cv2


dicom_folder = '/Users/salvatoreesposito/Downloads/train_data/masks/' # Set the folder of your dicom files that inclued images 
jpg_folder = '/Users/salvatoreesposito/Downloads/train_data/masks/' # Set the folder of your output folder for jpg files 
# Step 1. prepare your input(.dcm) and output(.jpg) filepath 
dcm_jpg_map = {}
for dicom_f in os.listdir(dicom_folder):
    dicom_filepath = os.path.join(dicom_folder, dicom_f)
    jpg_f = dicom_f.replace('.dcm', '.png') 
    jpg_filepath = os.path.join(jpg_folder,jpg_f)
    dcm_jpg_map[dicom_filepath] = jpg_filepath


unstacked_list = []
for dicom_filepath, jpg_filepath in dcm_jpg_map.items():
  # to skip the rest of the loop
    # convert dicom file into jpg file
    dicom = pydicom.read_file(dicom_filepath)
    print("PixelData" in dicom)
    np_pixel_array = dicom.pixel_array
    unstacked_list.append(np_pixel_array)
    cv2.imwrite(jpg_filepath, np_pixel_array)
final_array = np.array(unstacked_list)



# def add_gaussian_noise(inp, expected_noise_ratio=0.05):
#         image = inp.copy()
#         if len(image.shape) == 2:
#             row,col= image.shape
#             ch = 1
#         else:
#             row,col,ch= image.shape
#         mean = 0.
#         var = 0.1
#         sigma = var**0.5
#         gauss = np.random.normal(mean,sigma,(row,col)) * expected_noise_ratio
#         gauss = gauss.reshape(row,col)
#         noisy = image + gauss
#         return noisy
# def normalize(img):
#         arr = img.copy().astype(np.float)
#         M = np.float(np.max(img))
#         if M != 0:
#             arr *= 1./M
#         return arr
# def preprocess(filename, resize_ratio=0.25):
#     img = dicom.read_file(filename).pixel_array
#     img = normalize(zoom(img, resize_ratio))
#     img = add_gaussian_noise(img)
#     return img


# for dicom_file in os.listdir('/Users/salvatoreesposito/Downloads/train_data/masks/'):
#     pp_image = preprocess(dicom_file)    
#     imageio.imwrite(dicom_file.replace("dcm","png"), pp_image, "png")