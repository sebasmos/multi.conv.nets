"""
@author: Sebastian Cajas

Continue here https://rasterio.readthedocs.io/en/latest/topics/reproject.html 
https://sentinelhub-py.readthedocs.io/en/latest/configure.html#sentinel-hub-configuration

https://custom-scripts.sentinel-hub.com/
https://appliedsciences.nasa.gov/join-mission/training?program_area=All&languages=5&source=All
 
"""


import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show_hist
import numpy as np

###############################################################################

#test_img = rasterio.open("/home/sebasmos/Escritorio/multi.conv.nets/patches/images/image__168.tif")
# /home/sebasmos/Documentos/AI india/ai.team.datasets/Datasets /1. kaychallay


image = rasterio.open('/home/sebasmos/Documentos/AI india/ai.team.datasets/Datasets /1. kaychallay/T45QXE_2021-04-19.tif')

labels = rasterio.open("/home/sebasmos/Documentos/AI india/ai.team.datasets/Datasets /1. kaychallay/T45QXE_2021-04-19_default.ome.tiff")

##############################################################################
# Image information

# How many bands does this image have?
num_bands = image.count
print('Number of bands in image: {n}\n'.format(n=num_bands))

rows, cols = image.shape
print('Image size is: {r} rows x {c} columns\n'.format(r=rows, c=cols))

# What driver was used to open the raster?
driver = image.driver
print('Raster driver: {d}\n'.format(d=driver))

# What is the raster's projection?
proj = image.crs
print('Image projection:')
print(proj)

image = image.read()
image.shape

###############################################################################

labels = labels.read()
labels.shape
unique, counts = np.unique(labels, return_counts=True)
list(zip(unique, counts))

# img
#plt.imshow(test_img.read(1), cmap='pink')
#show_hist(test_img, bins=50, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram")

#mask
#plt.imshow(test_mask.read(1), cmap='pink')
#show_hist(test_mask, bins=50, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram")



