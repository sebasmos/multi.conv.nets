"""
@author: Sebastian Cajas
 
"""


import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show_hist


###############################################################################

test_img = rasterio.open("/home/sebasmos/Escritorio/multi.conv.nets/patches/images/image__168.tif")

test_mask = rasterio.open("/home/sebasmos/Escritorio/multi.conv.nets/patches/masks/mask__215.tif")

# img
plt.imshow(test_img.read(1), cmap='pink')
show_hist(test_img, bins=50, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram")

#mask
plt.imshow(test_mask.read(1), cmap='pink')
show_hist(test_mask, bins=50, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram")



