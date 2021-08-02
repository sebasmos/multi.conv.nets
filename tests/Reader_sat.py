"""
@author: Sebastian Cajas
 
Read satellite images and split into patches


!pip install geopandas
!pip install rasterio
!pip install patchify
!pip install imagecodecs
!pip install imagecodecs --force

"""

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff

def cutter(path_image, path_label,path_out):
    large_image_stack = tiff.imread(path_image)
    large_mask_stack = tiff.imread(path_label)

    large_image = large_image_stack[:,:] # Select desired number of bands
    patches_img = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite(path_out + '/patches/images/' + 'image_' + '_' + str(i)+str(j)+ ".tif", single_patch_img)


    large_mask = large_mask_stack[:,:]

    patches_mask = patchify(large_mask, (128, 128), step=128)  #Step=256 for 256 patches means no overlap


    for i in range(patches_mask.shape[0]):

        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,j,:,:]
            tiff.imwrite(path_out +'/patches/masks/' + 'mask_' + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
            single_patch_mask = single_patch_mask / 255.

def cutter_unidimensional(path_image, path_label,path_out):
    large_image_stack = tiff.imread(path_image)
    large_mask_stack = tiff.imread(path_label)

    large_image = large_image_stack[:,:,:]
    patches_img = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite(path_out + '/patches/images/' + 'image_' + '_' + str(i)+str(j)+ ".tif", single_patch_img)


    large_mask = large_mask_stack[:,:]

    patches_mask = patchify(large_mask, (128, 128), step=128)  #Step=256 for 256 patches means no overlap


    for i in range(patches_mask.shape[0]):

        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,j,:,:]
            tiff.imwrite(path_out +'/patches/masks/' + 'mask_' + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
            single_patch_mask = single_patch_mask / 255.

