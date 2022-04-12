#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:28:47 2020

@author: tkpci
"""



import gdal
from dataLandsatASTER import DataLandsatASTER
import numpy as np
import pdb
import createSiteInfo
import subprocess
import re 
import os
from tkinter import *   
from tkinter import filedialog 
import BuoyData



# create dataOut file

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/landsat8/")
    return root.foldername
    root.destroy()

def create_dataOut():
    
    # select folder with Landsat Scenes  
    folder = popup_get_folder()
    cnt=0
        
        # get list of landst scenes
    dir_downloads = os.listdir(folder) 
    
    for sceneID in dir_downloads:
        print('  Working with scene '+ str(cnt)+ ': '+sceneID)
        cnt+=1
        
        dataOut = DataLandsatASTER()
        
        folderID = sceneID
        files_in_folder = os.listdir(folder+'/'+folderID)
        
        for ele in files_in_folder:
            if 'MTL' in ele:
                MTL_file = ele
                
        for ele in files_in_folder:
            if 'T2_B10' in ele:
                B10_file = ele
            elif 'T1_B10' in ele:
                B10_file = ele
            elif 'RT_B10' in ele:
                B10_file = ele
                
        sceneID = B10_file[0:len(B10_file)-8]
        MTL_file =  folder + '/' + folderID + '/' +  MTL_file 
        
        if len(sceneID) == 21:
            pathrow = sceneID[3:9]
        else:  
            pathrow = sceneID[10:16]
        

        SWName = folder + '/' + folderID + '/' + sceneID + '_SW_LST.tif'
        SCName = folder + '/' + folderID + '/' + sceneID + '_SC_LST_Kelvin.tif'
    
   
    
        try:
            files_in_folder = os.listdir(folder+'/'+folderID)
            for ele in files_in_folder:
                    if 'ST_CDIST' in ele:
                        CDIST_file = ele
                        CDISTName = folder + '/' + folderID + '/' + CDIST_file
        except:
            CDISTName =  'none'
        
    # download aster stacked files
   
    
        gdal.UseExceptions()
        rd=gdal.Open(SWName)
        wkt_projection =rd.GetProjection()
        geoTransform= rd.GetGeoTransform()
        SW= rd.GetRasterBand(1)
        SW_data = SW.ReadAsArray()
        
       
        gdal.UseExceptions()
        rd=gdal.Open(SCName)
        SC= rd.GetRasterBand(1)
        SC_data = SC.ReadAsArray()
        
       
        try:
            gdal.UseExceptions()
            rd=gdal.Open(CDISTName)
            CDIST= rd.GetRasterBand(1)
            CDIST_data = CDIST.ReadAsArray() * 0.01   # scale factor according to initial albers data resease       
            where_are_NaNs = np.isnan(CDIST_data)
            CDIST_data[where_are_NaNs] = 0
        except:
            CDIST_data = 0        
        

      
        dataOut.setRad('SW_LST',SW_data.astype(float))
        dataOut.setRad('SC_LST',SC_data.astype(float))
        dataOut.setRad('SC_CDIST',CDIST_data.astype(float))
        
        siteInfo = createSiteInfo.CreateSiteInfo()
                  
        data_SR = BuoyData.getBuoyValues(MTL_file, pathrow, siteInfo)
        BuoyData.writeDataToFile(data_SR, dataOut, folder, folderID, sceneID, cwv=0)

    
    



            
 