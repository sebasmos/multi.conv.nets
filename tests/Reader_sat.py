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


large_image_stack = tiff.imread('/home/sebasmos/Documentos/AI india/gray/data/NN sample chapter 6/landsat_image.tif')
large_mask_stack = tiff.imread('/home/sebasmos/Documentos/AI india/gray/data/NN sample chapter 6/labels_image.tif')


large_image = large_image_stack[:,:,1]
patches_img = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap
       
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i,j,:,:]
        tiff.imwrite('../patches/images/' + 'image_' + '_' + str(i)+str(j)+ ".tif", single_patch_img)

     
large_mask = large_mask_stack[:,:]
    
patches_mask = patchify(large_mask, (128, 128), step=128)  #Step=256 for 256 patches means no overlap
    

for i in range(patches_mask.shape[0]):
    
    for j in range(patches_mask.shape[1]):
        single_patch_mask = patches_mask[i,j,:,:]
        tiff.imwrite('../patches/masks/' + 'mask_' + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
        single_patch_mask = single_patch_mask / 255.
        
        