#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:45:38 2022

@author: tkpci
"""

import numpy as np
import pandas as pd
import os
import subprocess
import re 
import csv
import cv2
import datetime
from osgeo import gdal
import matplotlib.pyplot as plt
import pdb


data = pd.read_csv("/dirs/data/tirs/Landsat9/data/RIT_L9_BUOY_titbits_SW.csv")
data['SW_LST'] = np.zeros(len(data)) +np.nan

data['temp_diff'] = np.zeros(len(data))+np.nan
folder = '/dirs/data/tirs/Landsat9/scenes/'
dir_files = os.listdir(folder)


for rows in range(len(data)):
    #pdb.set_trace()
    buoy_date = data['date'][rows]
    buoy_date = datetime.datetime.strptime(buoy_date, '%m/%d/%Y')
    buoy_pathrow = data['scene_id'][rows][3:9]
    for files in dir_files:
        
        L9_date = files[17:25]
        L9_pathrow = files[10:16]
        L9_date = datetime.datetime.strptime(L9_date, '%Y%m%d')
        if buoy_date == L9_date and buoy_pathrow == L9_pathrow:
            
            
            # get points from files    
            lat_value = data['lat'][rows]
            lon_value = data['long'][rows]
            
            file = folder  + files + '/' 
            
            files_in_folder = os.listdir(file)
            
            for ele in files_in_folder:
                if 'SW_LST.TIF' in ele:
                    SW_file =file +  ele

            
            pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + SW_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
            
            temp = pixelsLocate.decode()
        
            numbers = re.findall('\d+',temp) 
        
            x = int(numbers[0])
            y = int(numbers[1])
            
            gdal.UseExceptions()
            rd=gdal.Open(SW_file)
            wkt_projection =rd.GetProjection()
            geoTransform= rd.GetGeoTransform()
            data_sw= rd.GetRasterBand(1)
            SW_LST = data_sw.ReadAsArray()
            
            # if L9_date.day == 24 and L9_date.month == 12:
            #     plt.imshow(SW_LST)
            #     print(SW_file)
            #     print('x = ',x)
            #     print('y = ',y)
            
            #SW_LST_pixel = np.mean(SW_LST[y-1:y+1,x-1:x+1])
            SW_LST_pixel = SW_LST[y,x]
            
            data['SW_LST'][rows] = SW_LST_pixel
            
data['temp_diff'] = data['buoy_temp']-data['SW_LST']         

data.to_csv('/dirs/data/tirs/Landsat9/data/L9_buoy_SWLST_analysis.csv')

