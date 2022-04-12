#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:30:42 2020

Create aster files for SurfRad and other single point sites
@author: tkpci

"""

from tkinter import *   
from tkinter import filedialog 
import downloadLandsat
import csv
import warnings
import os
import downloadAster
import stackImages
import gdal
import subprocess
import re 
import pdb
import saveL8AsterInClass


# enter lat lon of point to find data for
def popup_get_latlon():
    
    
    def clear_search_lat(event):
        lat.delete(0, END) 
    def clear_search_lon(event):
        lon.delete(0, END) 
    def clear_search_site(event):
        site.delete(0, END) 
    
    def get_data():
        global lat_value
        lat_value = lat.get()
        global lon_value
        lon_value = lon.get()
        global site_value
        site_value = site.get()
        root.destroy()
        
    root = Tk()
    root.title('Enter Let Lon')  
              
    # Open window 
    root.geometry('380x150+100+150') 
    lat = Entry(root, width=15)
    lat.place(x=50, y= 30)
    lat.insert(0,'Lat e.g. 43.1566')
    lat.bind("<Button-1>", clear_search_lat) 
    #lat_value = lat.get()
    lon = Entry(root, width=15)
    lon.place(x=200, y= 30)
    lon.insert(0,'Lon e.g. -77.6088')
    lon.bind("<Button-1>", clear_search_lon) 
    #lon_value = lon.get()
    site = Entry(root, width=15)
    site.place(x=50, y= 70)
    site.insert(0,'Site e.g. FortPeck')
    site.bind("<Button-1>", clear_search_site)
    
    b = Button(root, text="run", width=10, command=get_data)
    b.place(x=140, y= 110)
    
    mainloop()
 
def createAsterstackforSmallL8():
        
    popup_get_latlon()
    
    image_size_to_save = 10 #km
    latlon = [float(lat_value), float(lon_value)]
    paths,rows =downloadLandsat.findPathRow(latlon, satellite = 'L8', disp = 1)
    sceneID=0;
    files_to_download = ['B10.TIF','MTL.txt']
    num_to_download = 'one'
    folder = downloadLandsat.downloadL8data(sceneID,paths, rows, files_to_download, num_to_download)
    
    sceneID = os.listdir(folder) 
    sceneID = sceneID[0]
       
    print('  Working with: '+sceneID)
    MTL_file = folder + '/' + sceneID + '/' + sceneID +'_MTL.txt'
    
    if len(sceneID) == 21:
        pathrow = sceneID[3:9]
    else:  
        pathrow = sceneID[10:16]
     
        
    aster_already_downloaded = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_' + pathrow + '.tif'
    if not os.path.isfile(aster_already_downloaded):
    
        # get corner coordinates from MTL file
        f = open(MTL_file,"r")
        content = f.read()
        
        UL_lat = content.index('CORNER_UL_LAT_PRODUCT')
        UL_lon = content.index('CORNER_UL_LON_PRODUCT')
        LR_lat = content.index('CORNER_LR_LAT_PRODUCT')
        LR_lon = content.index('CORNER_LR_LON_PRODUCT')
        
        UR_lat = content.index('CORNER_UR_LAT_PRODUCT')
        UR_lon = content.index('CORNER_UR_LON_PRODUCT')
        LL_lat = content.index('CORNER_LL_LAT_PRODUCT')
        LL_lon = content.index('CORNER_LL_LON_PRODUCT')
    
        CORNER_UL_LAT_PRODUCT = float(content[UL_lat+24:UL_lat+24+9].rstrip())
        CORNER_UL_LON_PRODUCT = float(content[UL_lon+24:UL_lon+24+9].rstrip())
        CORNER_LR_LAT_PRODUCT = float(content[LR_lat+24:LR_lat+24+9].rstrip())
        CORNER_LR_LON_PRODUCT = float(content[LR_lon+24:LR_lon+24+9].rstrip())
        
        CORNER_UR_LAT_PRODUCT = float(content[UR_lat+24:UR_lat+24+9].rstrip())
        CORNER_UR_LON_PRODUCT = float(content[UR_lon+24:UR_lon+24+9].rstrip())
        CORNER_LL_LAT_PRODUCT = float(content[LL_lat+24:LL_lat+24+9].rstrip())
        CORNER_LL_LON_PRODUCT = float(content[LL_lon+24:LL_lon+24+9].rstrip())
        
        coord = [max(CORNER_UL_LAT_PRODUCT,CORNER_UR_LAT_PRODUCT), \
                 min(CORNER_UL_LON_PRODUCT,CORNER_LL_LON_PRODUCT),
                 min(CORNER_LR_LAT_PRODUCT,CORNER_LL_LAT_PRODUCT),
                 max(CORNER_LR_LON_PRODUCT,CORNER_UR_LON_PRODUCT)]
    
        #download and save stacked georegistered aster files
        asterCube, coordsCube = downloadAster.downLoadAster(coord)
        downloadAster.georegisterAndSaveAster(asterCube, coordsCube, pathrow, filepath = 0)
    
    
     # open landsat image and reduce size based on smaller size with lat lon in middle
    image_large = folder + '/' + sceneID + '/' + sceneID +'_B10.TIF'
    image_small = folder + '/' + sceneID + '/' + sceneID +'_B10_sml.TIF'
    
    saveL8AsterInClass.reduceImageSize(image_large, image_small, folder, sceneID, site_value, image_size_to_save)
    # open landsat data
    
    gdal.UseExceptions()
    rd=gdal.Open(image_small)
    im= rd.GetRasterBand(1)
    im_data = im.ReadAsArray()
    
    
    # stack and resample aster to tirs 
    stackImages.stackLandsatAster(pathrow, folder, sceneID, site_value)

createAsterstackforSmallL8()
