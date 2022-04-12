#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:42:38 2020

@author: tkpci
"""
import osgeo.gdal as gdal

def saveGeotif(data, foldername, filename, wkt_projection, geoTransform):
    
    width = data.shape[0]
    height = data.shape[1]
    bands = 1  
    
    file = foldername + '/' + filename
    
    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(file, height, width, bands, gdal.GDT_Float32)
    ds.SetGeoTransform(geoTransform)
    ds.SetProjection(wkt_projection)
    ds.GetRasterBand(1).WriteArray(data)
    
    ds.FlushCache()
    ds = None
    
    print('File ' + filename + ' saved')