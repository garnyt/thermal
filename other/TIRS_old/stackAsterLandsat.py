#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:58:08 2020

@author: tkpci
"""


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst

def stackImg1Img2(img1_filename, img2_filename):
    #img1 = geotif image which resolution you want to keep
    #img2 = geotif image to be resampled and matched to img1 - this will be the output image
    
    # Source
    src_filename = img2_filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    
    # We want a section of source that matches this:
    match_filename = img1_filename
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    # Output / destination
    dst_filename = '/dirs/data/tirs/downloads/testStack.tif'
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    
    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    
    del dst # Flush


def stackLandsatAster(pathrow, folder, sceneID):

#b10Name = '/dirs/data/tirs/downloads/pathrow_040036/LC08_L1TP_040036_20190912_20190913_01_RT/LC08_L1TP_040036_20190912_20190913_01_RT_B10.TIF'
aster = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_024030.tif'
# download aster stacked files

b10Name = '/dirs/data/tirs/downloads/AnkurDesai/downloads/LC08_L1TP_024030_20180504_20180516_01_T1/LC08_L1TP_024030_20180504_20180516_01_T1_B10.TIF'

asterName = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_' + pathrow + '_30m.tif' 
asterName = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_024030_30m.tif' 
b10Name = folder + '/' + sceneID + '/' + sceneID + '_B3.TIF'






gdal.UseExceptions()
rd=gdal.Open(dst_filename)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
test= rd.GetRasterBand(1)
test_data = test.ReadAsArray()

np.min(test_data)
np.max(test_data)

plt.imshow(test_data, vmin=0.85, vmax=1)
plt.colorbar()

testtruth = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_024030_30m.tif'

gdal.UseExceptions()
rd=gdal.Open(testtruth)
wkt_projection =rd.GetProjection()
geoTransform= rd.GetGeoTransform()
truth= rd.GetRasterBand(1)
truth_data = truth.ReadAsArray()

plt.imshow(truth_data, vmin=0.85, vmax=1)
plt.colorbar()

diff = truth_data-test_data

plt.imshow(diff)
plt.colorbar()

plt.hist(diff.flatten(),bins=50)
