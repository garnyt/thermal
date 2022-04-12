#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:12:31 2022

@author: tkpci
"""

import subprocess
from osgeo import gdal
import matplotlib.pyplot as plt
import re

def locate_lat_lon_in_geotif(lat,lon, file):

    try:
        pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + file + ' ' + str(lon) + ' ' + str(lat), shell=True )
    
        temp = pixelsLocate.decode()

        numbers = re.findall('\d+',temp) 

        x = int(numbers[0]) # column    
        y = int(numbers[1]) # row
        
        row = y
        col = x
         
        return row, col
    
    except:
        print('Something went wrong. Make sure lat lon is located in scene and that lon is a minus value if west')
   
    
 
def main():
    
    filepath = '/dirs/data/tirs/L8L9_C2L1_202004_202203/data/LC90400342021364LGN01/'
    filename = 'LC09_L1GT_040034_20211230_20220121_02_T2_B10.TIF'
    file = filepath + filename
    
    lat = 37.356183
    lon = -115.962511 # note - remember to add a minus if west
    
    row, col = locate_lat_lon_in_geotif(lat,lon, file)
    
    # open file and locate pixel value
    gdal.UseExceptions()
    rd=gdal.Open(file)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    img = data.ReadAsArray() 
    
    #get point value in scene
    val = img[row,col]
    
    # plot point in scene
    plt.imshow(img, cmap = 'jet')
    plt.colorbar()
    plt.scatter(col,row, s=20,c='black')
    plt.title('Value: '+ str(val))
    
    plt.axis('off')
    
    