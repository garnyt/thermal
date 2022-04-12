#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 08:27:20 2021

@author: tkpci
"""
from osgeo import gdal
import matplotlib.pyplot as plt
import sys
sys.path.append('/cis/staff/tkpci/Code/Python/myLib/')

from georegister_Images import georegisterImages


filepath = '/dirs/data/tirs/downloads/test/emis_analysis/LC08_L1TP_025029_20200210_20200823_02_T1_snow/'
img1_filename = filepath + 'LC08_L1TP_025029_20200210_20200823_02_T1_SW_LST.TIF'

dest_filepath = '/dirs/data/tirs/downloads/test/emis_analysis/LC08_L1TP_025029_20200617_20200823_02_T1_veg/'
img2_filename = dest_filepath + 'LC08_L1TP_025029_20200617_20200823_02_T1_B11.TIF'

filename = img2_filename[:-4]+'_resampled.TIF'


georegisterImages(img1_filename, img2_filename, filename)



gdal.UseExceptions()
rd=gdal.Open(img1_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
orig = data.ReadAsArray() 

gdal.UseExceptions()
rd=gdal.Open(filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
new = data.ReadAsArray() 


plt.subplot(1,2,1)
plt.imshow(orig)
plt.colorbar()
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(new)
plt.colorbar()
plt.axis('off')


