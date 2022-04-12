#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:25:02 2021

@author: tkpci
"""


from osgeo import gdal
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('/cis/staff/tkpci/Code/Python/myLib/')

from georegister_Images import georegisterImages


filepath = '/dirs/data/tirs/Landsat8/'
img1_filename = filepath + 'LC08_L1TP_014035_20211114_20211125_02_T1_SW_LST.TIF'

dest_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_014035_20211114_20211209_02_T1_resampled/'
img2_filename = dest_filepath + 'LC09_L1TP_014035_20211114_20211209_02_T1_SW_LST.TIF'


gdal.UseExceptions()
rd=gdal.Open(img1_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
l8 = data.ReadAsArray() 
l8 = l8 + 0.0


gdal.UseExceptions()
rd=gdal.Open(img2_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
l9 = data.ReadAsArray() 
l9 = l9 + 0.0


plt.subplot(1,3,1)
plt.imshow(l8, vmin = 278, vmax = 293, cmap = 'jet')
plt.colorbar()
plt.axis('off')
plt.title('SW LST L8')
#plt.title('B10 L8')
plt.subplot(1,3,2)
plt.imshow(l9, vmin = 278, vmax = 293, cmap = 'jet')
plt.colorbar()
plt.axis('off')
plt.title('SW LST L9')
#plt.title('B10 L9')



diff = l8-l9


plt.subplot(1,3,3)
plt.imshow(diff, vmin = -2, vmax = 1, cmap = 'jet')
plt.colorbar()
plt.axis('off')
#plt.title('DIfference: L8-L9')
plt.title('SW LST difference')



gdal.UseExceptions()
rd=gdal.Open(img3_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
l9_no_filter = data.ReadAsArray() 


plt.imshow(l9_no_filter, vmin = 270, vmax = 290, cmap = 'jet')
plt.colorbar()
plt.axis('off')
plt.title('SW LST L9 without filtering')

temp = l9_no_filter - l9
plt.imshow(temp)



#####
filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_014035_20211114_20211209_02_T1/'
sceneID_1 = 'LC09_L1TP_014035_20211114_20211209_02_T1_B10.TIF'
sceneID_2 = 'LC09_L1TP_014035_20211114_20211209_02_T1_B11.TIF'

file_1 = filepath + sceneID_1
file_2 = filepath + sceneID_2

gdal.UseExceptions()
rd=gdal.Open(file_1)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
img_1 = data.ReadAsArray() 
img_1 = img_1 + 0.0
img_1 = img_1 * 0.00038 + 0.1

gdal.UseExceptions()
rd=gdal.Open(file_2)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
img_2 = data.ReadAsArray() 
img_2 = img_2 + 0.0
img_2 = img_2 * 0.000349 + 0.1


plt.imshow(img_1 - img_2, vmin = 0.4, vmax = 0.6, cmap = 'jet')
plt.axis('off')
plt.colorbar()
plt.title('Radiance L9 difference')


## compare rad values
filepath = '/dirs/data/tirs/Landsat8/'
img2_filename = filepath + 'LC08_L1TP_014035_20211114_20211125_02_T1_emis11.TIF'

filepath = '/dirs/data/tirs/Landsat8/'
img1_filename = filepath + 'LC08_L1TP_014035_20211114_20211125_02_T1_ls_emis_data_11.TIF'

dest_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_014035_20211114_20211209_02_T1_resampled/'
img2_filename = dest_filepath + 'LC09_L1TP_014035_20211114_20211209_02_T1_emis11.TIF'

dest_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_014035_20211114_20211209_02_T1_resampled/'
img1_filename = dest_filepath + 'LC09_L1TP_014035_20211114_20211209_02_T1_ls_emis_data_11.TIF'


gdal.UseExceptions()
rd=gdal.Open(img1_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
l8 = data.ReadAsArray() 
l8 = l8 + 0.0

gdal.UseExceptions()
rd=gdal.Open(img2_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
l9 = data.ReadAsArray() 
l9 = l9 + 0.0


diff = l8-l9
# diff[diff <-0.8] = np.nan
# diff[diff == 0] = np.nan
# diff[diff > 0.8] = np.nan
# diff[diff == -1.4901161138336505e-09] = np.nan

#l8[2719:3133 ,2235:2742 ]
plt.subplot(1,3,1)
plt.imshow(l8, vmin = 0.95, vmax = 1, cmap = 'jet')
plt.colorbar()
plt.axis('off')

plt.title('ls_emis_data_11 L8')

plt.subplot(1,3,2)
plt.imshow(l9, vmin = 0.95, vmax = 1, cmap = 'jet')
plt.colorbar()
plt.axis('off')

plt.title('ls_emis_data_11 L9')

plt.subplot(1,3,3)
plt.imshow(diff, vmin = -0.01, vmax = 0.01, cmap = 'jet')
plt.colorbar()
plt.axis('off')
plt.title('DIfference: L8-L8')


######

dest_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_038034_20211216_20211216_02_T1/'
img2_filename = dest_filepath + 'LC09_L1TP_038034_20211216_20211216_02_T1_SW_LST.TIF'

dest_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_038034_20211216_20211216_02_T1/'
img1_filename = dest_filepath + 'LC09_L1TP_038034_20211216_20211216_02_T1_SW_LST_no_filter.TIF'




gdal.UseExceptions()
rd=gdal.Open(img1_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
img1 = data.ReadAsArray() 
img1 = img1 + 0.0

gdal.UseExceptions()
rd=gdal.Open(img2_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
img2 = data.ReadAsArray() 
img2 = img2 + 0.0


diff = img1-img2
# diff[diff <-0.8] = np.nan
# diff[diff == 0] = np.nan
# diff[diff > 0.8] = np.nan
# diff[diff == -1.4901161138336505e-09] = np.nan

#l8[2719:3133 ,2235:2742 ]
plt.subplot(1,3,1)
plt.imshow(img1, vmin = 270, vmax = 280, cmap = 'jet')
plt.colorbar()
plt.axis('off')

plt.title('L9 SW LST')

plt.subplot(1,3,2)
plt.imshow(img2, vmin = 270, vmax = 280, cmap = 'jet')
plt.colorbar()
plt.axis('off')

plt.title('L9 SW LST (no filter) L9')

plt.subplot(1,3,3)
plt.imshow(diff, vmin = -0.1, vmax = 0.3, cmap = 'jet')
plt.colorbar()
plt.axis('off')
plt.title('DIfference: with and without filter')

# filepath = '/dirs/data/tirs/Landsat8/'
# img1_filename = filepath + 'LC08_L1TP_014035_20211114_20211125_02_T1_B10.TIF'

# dest_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_014035_20211114_20211209_02_T1_resampled/'


# dir = os.listdir(dest_filepath)

# for scene in dir:
#     if 'B' in scene:
#         print(scene)
        
#         img2_filename = dest_filepath + scene

#         filename = img2_filename[:-4]+'_resampled.TIF'
#         georegisterImages(img1_filename, img2_filename, filename)
