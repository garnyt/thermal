#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:43:08 2020

@author: tkpci
"""

import gdal
from dataLandsatASTER import DataLandsatASTER
import numpy as np
import createSiteInfo
import subprocess
import re 
import os


def dataForEmisCalc(filepathLandsat, sceneID):
    
    dataOut = DataLandsatASTER()
    
    #asterName = '/dirs/data/tirs/downloads/aster/AutoDownloads/e13_e14_ndvi_registered.tif' 
    asterName = filepathLandsat + 'e13_e14_ndvi_registered.tif' 
    b3Name = filepathLandsat + sceneID + '_B3.TIF'
    b4Name = filepathLandsat + sceneID + '_B4.TIF'
    b5Name = filepathLandsat + sceneID + '_B5.TIF'
    b6Name = filepathLandsat+ sceneID + '_B6.TIF'
    b10Name = filepathLandsat + sceneID + '_B10.TIF'
    b11Name = filepathLandsat + sceneID + '_B11.TIF'
    bqaName = filepathLandsat + sceneID + '_BQA.TIF'
    
    try:
        files_in_folder = os.listdir(filepathLandsat)
        for ele in files_in_folder:
                if 'ST_B10' in ele:
                    SC_file = ele
                    SCName = filepathLandsat + SC_file
    except:
        SCName =  'none'
        
    try:
        files_in_folder = os.listdir(filepathLandsat)
        for ele in files_in_folder:
                if 'ST_QA' in ele:
                    QA_file = ele
                    QAName = filepathLandsat + QA_file
    except:
        QAName =  'none'
        
    try:
        files_in_folder = os.listdir(filepathLandsat)
        for ele in files_in_folder:
                if 'ST_CDIST' in ele:
                    CDIST_file = ele
                    CDISTName = filepathLandsat + CDIST_file
    except:
        CDISTName =  'none'
    
    # download aster stacked files
    #pdb.set_trace()
    
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
    
    gdal.UseExceptions()
    rd=gdal.Open(b3Name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    b3= rd.GetRasterBand(1)
    b3_data = b3.ReadAsArray()
    
    where_are_NaNs = np.isnan(b3_data)
    b3_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b4Name)
    b4= rd.GetRasterBand(1)
    b4_data = b4.ReadAsArray()
    
    where_are_NaNs = np.isnan(b4_data)
    b4_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b5Name)
    b5= rd.GetRasterBand(1)
    b5_data = b5.ReadAsArray()
    
    where_are_NaNs = np.isnan(b5_data)
    b5_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b6Name)
    b6= rd.GetRasterBand(1)
    b6_data = b6.ReadAsArray()
    
    where_are_NaNs = np.isnan(b6_data)
    b6_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b10Name)
    b10= rd.GetRasterBand(1)
    b10_data = b10.ReadAsArray()
    
    where_are_NaNs = np.isnan(b10_data)
    b10_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b11Name)
    b11= rd.GetRasterBand(1)
    b11_data = b11.ReadAsArray()
    
    where_are_NaNs = np.isnan(b11_data)
    b11_data[where_are_NaNs] = 0
    
    try:
        gdal.UseExceptions()
        rd=gdal.Open(SCName)
        ST= rd.GetRasterBand(1)
        ST_data = ST.ReadAsArray()        
        where_are_NaNs = np.isnan(ST_data)
        ST_data[where_are_NaNs] = 0
    except:
        ST_data = 0
        
    try:
        gdal.UseExceptions()
        rd=gdal.Open(QAName)
        QA= rd.GetRasterBand(1)
        QA_data = QA.ReadAsArray() * 0.01   # scale factor according to initial albers data resease      
        where_are_NaNs = np.isnan(QA_data)
        QA_data[where_are_NaNs] = 0
    except:
        QA_data = 0

    try:
        gdal.UseExceptions()
        rd=gdal.Open(CDISTName)
        CDIST= rd.GetRasterBand(1)
        CDIST_data = CDIST.ReadAsArray() * 0.01   # scale factor according to initial albers data resease       
        where_are_NaNs = np.isnan(CDIST_data)
        CDIST_data[where_are_NaNs] = 0
    except:
        CDIST_data = 0        
    
    #bqaName = '/dirs/data/tirs/downloads/test/LC08_L1TP_026028_20190708_20190719_01_T1/LC08_L1TP_026028_20190708_20190719_01_T1_BQA.TIF'
    #b10Name = '/dirs/data/tirs/downloads/test/LC08_L1TP_026028_20190708_20190719_01_T1/LC08_L1TP_026028_20190708_20190719_01_T1_B10.TIF'
    
    
    try:
    # create cloud mask
        gdal.UseExceptions()
        rd=gdal.Open(bqaName)
        bqa= rd.GetRasterBand(1)
        bqa_data = bqa.ReadAsArray()
        bqa_data = bqa_data.astype(float)
    except:
        bqa_data = np.zeros([b11_data.shape[0],b11_data.shape[1]])
    
    
    
    # https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con 
    
    cloud = [2800,2804,2808,2812,6896,6900,6904,6908]
    
    # note cloud = 0, clear = 1 >> this is easier as a mask to apply
    for val in cloud:
        bqa_data[bqa_data == val] = -10
        
    bqa_data[bqa_data >= 0] = 1
    bqa_data[bqa_data < 1] = 0
    
    dataOut.setRad('rad3',b3_data.astype(float))
    dataOut.setRad('rad4',b4_data.astype(float))
    dataOut.setRad('rad5',b5_data.astype(float))
    dataOut.setRad('rad6',b6_data.astype(float))
    dataOut.setRad('rad10',b10_data.astype(float))
    dataOut.setRad('rad11',b11_data.astype(float))
    dataOut.setRad('cloud',bqa_data.astype(float))
    dataOut.setAster('emis13',e13_data.astype(float))
    dataOut.setAster('emis14',e14_data.astype(float))
    dataOut.setAster('ndvi',ndvi_data.astype(float))
    dataOut.setAster('emis13_std',e13_std_data.astype(float))
    dataOut.setAster('emis14_std',e14_std_data.astype(float))
    
    return dataOut, wkt_projection, geoTransform


