#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:43:08 2020

@author: tkpci
"""

import osgeo.gdal as gdal
import numpy as np
import subprocess
import re 
import os


def main(dataOut):
    

    files_in_folder = os.listdir(dataOut['folder'])
    for ele in files_in_folder:
        if '_B3' in ele:
            b3Name = dataOut['folder'] + '/' + ele
            if 'SR' in ele:
                dataOut['SR'] = True
            else:
                dataOut['SR'] = False
        if '_B4' in ele:
            b4Name = dataOut['folder'] + '/' + ele
        if '_B5' in ele:
            b5Name = dataOut['folder'] + '/' + ele
        if '_B6' in ele:
            b6Name = dataOut['folder'] + '/' + ele
        if '_B10' in ele:
            if 'ST_B10' not in ele:
                b10Name = dataOut['folder'] + '/' + ele
        if '_B11' in ele:
            b11Name = dataOut['folder'] + '/' + ele    
        if 'e13_e14' in ele:   
            asterName = dataOut['folder'] + '/' + ele  
            
    # open files and add to dictionary
    # open ASTER emis and NDVI and standard dev files
    gdal.UseExceptions()
    rd=gdal.Open(asterName)
    e13= rd.GetRasterBand(1)
    e13_data = e13.ReadAsArray()
    e14= rd.GetRasterBand(2)
    e14_data = e14.ReadAsArray()
    ndvi= rd.GetRasterBand(3)
    ndvi_data = ndvi.ReadAsArray()
    e13_std= rd.GetRasterBand(4)
    e13_std_data = e13_std.ReadAsArray()
    e14_std= rd.GetRasterBand(5)
    e14_std_data = e14_std.ReadAsArray()
    
    # set NaN values to zero (for future calcuations)
    where_are_NaNs = np.isnan(ndvi_data)
    ndvi_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e13_data)
    e13_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e14_data)
    e14_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e13_std_data)
    e13_std_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e14_std_data)
    e14_std_data[where_are_NaNs] = 0
    
    # open B3 data for NDSI calculation
    gdal.UseExceptions()
    rd=gdal.Open(b3Name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    b3= rd.GetRasterBand(1)
    b3_data = b3.ReadAsArray()
    where_are_NaNs = np.isnan(b3_data)
    b3_data[where_are_NaNs] = 0
    
    # open B4 data for NDVI calculation
    gdal.UseExceptions()
    rd=gdal.Open(b4Name)
    b4= rd.GetRasterBand(1)
    b4_data = b4.ReadAsArray()
    where_are_NaNs = np.isnan(b4_data)
    b4_data[where_are_NaNs] = 0
    
    # open B5 data for NDVI calculation
    gdal.UseExceptions()
    rd=gdal.Open(b5Name)
    b5= rd.GetRasterBand(1)
    b5_data = b5.ReadAsArray()
    where_are_NaNs = np.isnan(b5_data)
    b5_data[where_are_NaNs] = 0
    
    # open B6 data for NDSI calculation
    gdal.UseExceptions()
    rd=gdal.Open(b6Name)
    b6= rd.GetRasterBand(1)
    b6_data = b6.ReadAsArray()
    where_are_NaNs = np.isnan(b6_data)
    b6_data[where_are_NaNs] = 0
    
    # open B10 data
    gdal.UseExceptions()
    rd=gdal.Open(b10Name)
    b10= rd.GetRasterBand(1)
    b10_data = b10.ReadAsArray()
    where_are_NaNs = np.isnan(b10_data)
    b10_data[where_are_NaNs] = 0
    
    # open B11 data
    gdal.UseExceptions()
    rd=gdal.Open(b11Name)
    b11= rd.GetRasterBand(1)
    b11_data = b11.ReadAsArray()
    where_are_NaNs = np.isnan(b11_data)
    b11_data[where_are_NaNs] = 0

   
    dataOut['rad3'] = b3_data.astype(float)
    dataOut['rad4'] = b4_data.astype(float)
    dataOut['rad5'] = b5_data.astype(float)
    dataOut['rad6'] = b6_data.astype(float)
    dataOut['rad10'] = b10_data.astype(float)
    dataOut['rad11'] = b11_data.astype(float)
    dataOut['emis13'] = e13_data.astype(float)
    dataOut['emis14'] = e14_data.astype(float)
    dataOut['aster_ndvi'] = ndvi_data.astype(float)
    dataOut['emis13_std'] = e13_std_data.astype(float)
    dataOut['emis14_std'] = e14_std_data.astype(float)
    
    # save georegistration data for writing out SW and emis files
    dataOut['wkt_projection'] = wkt_projection
    dataOut['geoTransform'] = geoTransform
    
    print('All files successfully loaded into memory')
    
    
    return dataOut

