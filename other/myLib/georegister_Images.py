#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 08:14:10 2021

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import os
import pdb


def georegisterImages(img1_filename, img2_filename, output_filename):
    
    #img1_filename = geotif image which resolution you want to keep
    #img2_filename = geotif image to be resampled and matched to img1 - this will be the output image
    
    # Source
    src_filename = img2_filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    #src.GetMetadata()
    bands = src.RasterCount
    
    
    
    # We want a section of source that matches this:
    match_filename = img1_filename
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    
    try:
        
        dst_filename = output_filename
              
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, bands, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection( match_proj)
        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
        
        del dst # Flush
        print('  File stacked and saved')
        
    except:
        print('  Oops, something went wrong wile stacking the images. Make sure input images are GEOtifs, and that they overlap on a map')
        
        
        

