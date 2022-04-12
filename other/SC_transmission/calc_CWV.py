#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:05:55 2020

@author: tkpci

Fundction input:
    b10_DC = TIRS band10 level one digital count
    b11_DC = TIRS band11 level one digital count
    lat = point of interest
    lon = point of interest
    
    
Function output:
    CWV = calculated total column water vapor
"""

import numpy as np
import cv2
import pdb
import pandas as pd
from scipy import ndimage
import gdal
import subprocess
import os
import createSiteInfo
import re
import matplotlib.pyplot as plt
import pdb

def read_b10_b11(folder):
    
    
    for name in os.listdir(folder):
        if 'T1_B10.TIF' in name:
            b10Name = folder + name
        if 'T2_B10.TIF' in name:
            b10Name = folder + name
        if 'RT_B10.TIF' in name:
            b10Name = folder + name
        if 'B11.TIF' in name:
            b11Name = folder + name
        if 'CDIST.TIF' in name:
            cloudname = folder + name
    
    file_folder = b10Name
       
    
    gdal.UseExceptions()
    rd=gdal.Open(b10Name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    b10= rd.GetRasterBand(1)
    rad10 = b10.ReadAsArray()
    
    where_are_NaNs = np.isnan(rad10)
    rad10[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b11Name)
    b11= rd.GetRasterBand(1)
    rad11 = b11.ReadAsArray()
    
    where_are_NaNs = np.isnan(rad11)
    rad11[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(cloudname)
    cl= rd.GetRasterBand(1)
    cloud = cl.ReadAsArray()
    
    where_are_NaNs = np.isnan(rad11)
    rad11[where_are_NaNs] = 0
    
    T10 = TIRS_radiance(rad10, 10.9)
    T11 = TIRS_radiance(rad11, 12)
    
    return T10, T11, cloud, b10Name


# TIRS_radiance function converts TIRS band 10 and band 11 digital count to radiance and apparent temperature
def TIRS_radiance(counts, center_wave):
    
    radiance = counts.astype(float) * 3.3420 * 10**(-4) + 0.1
    
    wvl = center_wave * 1e-6

    L = radiance * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    Temp = (2 * h * c * c) / (L * (wvl**5))
    Temp2 = np.log(Temp+1)
    appTemp = (h * c )/ (k * wvl *Temp2)

        
    return appTemp


def calcCWV(T10,T11,cloud, lat,lon, b10Name,N):
    
    # CWV calculation based on 10x10km image size = must still ajust for large image
    # assuming pixel of interest is in center of image
    # coefficients
    c0 = 14.258
    c1 = -15.643
    c2 = 1.421
        
    cloud[cloud>0]=1
    
    
    pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + b10Name + ' ' + str(lon) + ' ' + str(lat), shell=True )
   
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    x = int(numbers[0])
    y = int(numbers[1])
    
    half = int(round(N/2))
    T10_N = T10[y-half:y+half,x-half:x+half]
    T11_N = T11[y-half:y+half,x-half:x+half]
    cloud = cloud[y-half:y+half,x-half:x+half]
    #cloud_new = ndimage.binary_erosion(cloud,structure=np.ones((15,15))).astype(np.int)
    if np.mean(cloud) == 0:
        print('Cloud in scene')
    
    T10 = np.multiply(T10_N,cloud)
    T11 = np.multiply(T11_N,cloud)
    
    T10[T10 == 0] = np.nan
    mean_b10 = np.nanmean(T10.flatten())
    
    T11[T11 == 0] = np.nan
    mean_b11 = np.nanmean(T11.flatten())
    
    ratio_top = np.nansum(np.multiply((T10-mean_b10),(T11-mean_b11)))   
    ratio_bottom = np.nansum(np.square(T10-mean_b10))
    
    ratio = np.divide(ratio_top,ratio_bottom)
    
    cwv = c0 + c1*ratio + c2*np.square(ratio)
    
    return cwv



def main():
    
    #filename = '/dirs/data/tirs/downloads/Surfrad/results/analysis_level2_trans_all.csv'
    filename = '/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_level2_trans_radiosonde.csv'
    
    out = pd.read_csv(filename)
    #scene_ID = list(out['Scene_ID'])
    scene_buoy = list(out['scene_buoy'])
    
    # cwv_L8_calc = np.zeros(len(scene_ID))-9999
    # cwv_L8_calc = list(cwv_L8_calc)
    # out['cwv_L8_calc'] = cwv_L8_calc
        
    #out.to_csv('/dirs/data/tirs/downloads/Surfrad/results/analysis_level2_trans_all.csv',index = False)
    #out.to_csv('/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_level2_trans.csv',index = False)
    
    #filepath = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_P040_R035/'
    filepath = '/dirs/data/tirs/downloads/Buoy_level2_downloads/Lake_Tahoe/'
    N = 14
    pd.options.mode.chained_assignment = None  # default='warn'
    
    for sceneID in os.listdir(filepath):
        print('Working with: ' + sceneID)
        
        folder = filepath + sceneID + '/'
        pathrow = sceneID[3:9]
        siteInfo = createSiteInfo.CreateSiteInfo()
        #line = siteInfo.pathrow.index(pathrow)
        indexes = np.where(np.array(siteInfo.pathrow) == pathrow)[0]
        
        scene = os.listdir(folder)[1]
        scene = scene[0:40]
        print(scene)
            
        
        for i in range(indexes.shape[0]):
            
            try:
                
                site_name = siteInfo.shortname[indexes[i]]
                idx = scene_buoy.index(site_name + scene[10:])
                lat = siteInfo.lat[indexes[i]]
                lon = siteInfo.lon[indexes[i]]
                T10, T11, cloud, b10Name = read_b10_b11(folder)
                   
                #plt.imshow(cloud)
                
                cwv = calcCWV(T10,T11,cloud, lat,lon, b10Name,N)
                print(cwv)
                out['cwv_L8_calc'][idx] = list(cwv)
                out.to_csv('/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_level2_trans_radiosonde2.csv',index = False)
                
            except: 
                print(sceneID + ' is not in list')
        
main()        
        
        
        
        
    
        
        
    
    
    
    
    
    
    
    
    