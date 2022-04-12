#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:00:37 2020

@author: tkpci
"""

import gdal
from dataLandsatASTER import DataLandsatASTER
import numpy as np
import matplotlib.pyplot as plt


filepath = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_P040_R034/LC80400342017089LGN00/'
filename = 'LC08_L2SP_040034_20170330_20200718_02_T1_ST_QA.TIF'
filename_cloud = 'LC08_L2SP_040034_20170330_20200718_02_T1_ST_CDIST.TIF'
filename_LST = 'LC08_L1TP_040034_20170330_20200718_02_T1_SC_LST_Kelvin.tif'
filename_B10 = 'LC08_L1TP_040034_20170330_20200718_02_T1_B10.TIF'

gdal.UseExceptions()
rd=gdal.Open(filepath + filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
img = data.ReadAsArray()

gdal.UseExceptions()
rd=gdal.Open(filepath + filename_cloud)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
cloud = data.ReadAsArray()

gdal.UseExceptions()
rd=gdal.Open(filepath + filename_LST)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
SC = data.ReadAsArray()

gdal.UseExceptions()
rd=gdal.Open(filepath + filename_B10)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
data= rd.GetRasterBand(1)
B10 = data.ReadAsArray()


plt.subplot(2,2,1)
plt.imshow(img*0.01, vmin = 0)
plt.colorbar()
plt.axis("off")
plt.title('SC QA [K]')

plt.subplot(2,2,2)
plt.imshow(cloud*0.01, vmin = 0)
plt.colorbar()
plt.axis("off")
plt.title('SC CDIST [Km]')

plt.subplot(2,2,3)
plt.imshow(SC)
plt.colorbar()
plt.axis("off")
plt.title('SC LST [K]')

plt.subplot(2,2,4)
plt.imshow(img*0.01, vmin = 0, vmax = 6)
plt.colorbar()
plt.axis("off")
plt.title('SC QA [K]')





