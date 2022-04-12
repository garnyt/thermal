#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 08:44:21 2022

@author: tkpci
"""

from osgeo import gdal
import os

import matplotlib.pyplot as plt


# change to directory - for some reason I could not get it to work when giving a full filepath
cd /cis/staff/tkpci/
# first put the ENVI file to convert to tif, and then the name where the tif files will be saved
download_call = 'gdal_translate -of GTiff 20211114_LandsatUnderflight_nano_1505.img nano.tif'

os.system(download_call)


# resample and georegister file - img1 is the gsd output you will get, img2 is the file you want to resample        
img1_filename = '/cis/staff/tkpci/nano.tif'
img2_filename = '/cis/staff/tkpci/swir.tif'
output_filename = '/cis/staff/tkpci/swir_to_nano_gsd.tif'

georegisterImages(img1_filename, img2_filename, output_filename)

gdal.UseExceptions()
rd=gdal.Open('/cis/staff/tkpci/swir_to_nano_gsd.tif')
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
# specific band number
data= rd.GetRasterBand(99)
swir = data.ReadAsArray() 


plt.imshow(swir)
plt.colorbar()

