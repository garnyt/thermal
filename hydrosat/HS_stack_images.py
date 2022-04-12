#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:58:08 2020

@author: tkpci
"""


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import os
import pdb


def stackImg1Img2(img1_filename, img2_filename, path):
    
    #img1 = geotif image which resolution you want to keep
    #img2 = geotif image to be resampled and matched to img1 - this will be the output image
    
    # Source
    src_filename = img2_filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    #src_geotrans = src.GetGeoTransform()
    
    # We want a section of source that matches this:
    match_filename = img1_filename
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    try:
        
        dst_filename = path + '/e13_e14_ndvi_registered.tif'
              
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 5, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection( match_proj)
        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
        
        del dst # Flush
        print('  Aster stacked and saved')
        
    except:
        print('  Oops, something went wrong wile stacking ASTER to Landsat - call the Queen!')


def stackLandsatAster(dataOut):

    aster = dataOut['variables']['aster_filepath'] + 'e13_e14_NDVI_' + dataOut['pathrow'] + '.tif' 
    
    files_in_folder = os.listdir(dataOut['folder'])
    
    for ele in files_in_folder:
        if 'B10' in ele:
            if 'ST' not in ele:
                landsat = dataOut['folder'] + '/' + ele
    
    path = dataOut['folder']
    
    print('  Resampling and stacking ASTER onto Landsat')
    stackImg1Img2(landsat, aster, path)







