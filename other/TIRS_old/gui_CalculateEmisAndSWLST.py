#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:38:41 2019

@author: tkpci
"""

from tkinter import *   
from tkinter import filedialog 
import csv
import os
import saveL8AsterInClass
import estimate_landsat_emissivity
import calc_SW
import SaveGeotif
import matplotlib.pyplot as plt
import numpy as np
import stackImages
import createSiteInfo
import calcError
import SurfradData
import pdb
import warnings
import osgeo.gdal as gdal

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/landsat8/")
    return root.foldername
    root.destroy()
    
# def popup_scenewide():
    
#     def get_full_scene():
#         global answer
#         answer = 'full'
#         root.destroy()
    
#     def get_single_point():
#         global answer
#         answer = 'single'
#         root.destroy()
    
#     root = Tk()
#     root.title('Would you like to process full scenes or just a single point?')  
              
#     # Open window 
#     root.geometry('450x80+100+100') 
#     b1 = Button(root, text="Full Scene(s)", width=15, command=get_full_scene)
#     b1.place(x=50, y= 30)
#     b2 = Button(root, text="Single Point(s)", width=15, command=get_single_point)
#     b2.place(x=250, y= 30)
    
#     mainloop()
     

############################### RUN FUNCTIONS ######################    

def calcEmisAndSW():
# select if you want a full scene or just single point emissivity and SW LST
    warnings.simplefilter('ignore')
    #popup_scenewide() 
    
    answer = 'full'
    
    # select folder with Landsat Scenes  
    folder = popup_get_folder()
    
    if answer == 'full':
        
        # get list of landst scenes
        dir_downloads = os.listdir(folder) 
        cnt = 1
        
        for sceneID in dir_downloads:
            print('  Working with scene '+ str(cnt)+ ': '+sceneID)
            cnt+=1
            
            folderID = sceneID
            files_in_folder = os.listdir(folder+'/'+folderID)
            
            for ele in files_in_folder:
                if 'MTL.txt' in ele:
                    MTL_file = ele
                    
            for ele in files_in_folder:
                if 'T2_B10' in ele:
                    B10_file = ele
                elif 'T1_B10' in ele:
                    B10_file = ele
                elif 'RT_B10' in ele:
                    B10_file = ele
                    
            sceneID = B10_file[0:len(B10_file)-8]
             
            MTL_file = folder + '/' + folderID + '/' + MTL_file
            SW_already_calculated = folder + '/' + folderID + '/' + sceneID +'_SW_LST_new.tif'
            emis_already_calculated = folder + '/' + folderID + '/' + sceneID +'_emis10.tif'
            
            if not os.path.isfile(SW_already_calculated):
            
                if len(sceneID) == 21:
                    pathrow = sceneID[3:9]
                else:  
                    pathrow = sceneID[10:16]
                    
                aster_already_downloaded = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_' + pathrow + '.tif'
                #dataOut, wkt_projection, geoTransform = saveL8AsterInClass.dataForEmisCalc(pathrow, folder, folderID, sceneID)
    
            
                if os.path.isfile(emis_already_calculated):
                        
                    print('  ' + sceneID + '_emis already calculated')
                    
                    emis10_name = folder + '/' + folderID + '/' + sceneID +'_emis10.tif'
                    emis11_name = folder + '/' + folderID + '/' + sceneID +'_emis11.tif'
                    
                    gdal.UseExceptions()
                    rd=gdal.Open(emis10_name)
                    # wkt_projection =rd.GetProjection()
                    # geoTransform= rd.GetGeoTransform()
                    temp= rd.GetRasterBand(1)
                    emis10 = temp.ReadAsArray()
                    
                    gdal.UseExceptions()
                    rd=gdal.Open(emis11_name)
                    # wkt_projection =rd.GetProjection()
                    # geoTransform= rd.GetGeoTransform()
                    temp= rd.GetRasterBand(1)
                    emis11 = temp.ReadAsArray()

                else:

                    if not os.path.isfile(aster_already_downloaded):
                    
                        # for each scene, get corner coordinates from MTL file
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
                        
                    else:
                        print('  Aster already downloaded for path row '+ pathrow)
                
                        # stack and resample aster to tirs 
                        stackImages.stackLandsatAster(pathrow, folder, folderID, sceneID)
                    
                        # calculate emissivity - save in folder and write to class
                        
                        print('  Calculating Landsat Emissivity')
                    
                        # open all files for emissivityand SW calc into class
                        dataOut, wkt_projection, geoTransform = saveL8AsterInClass.dataForEmisCalc(pathrow, folder, folderID, sceneID)
    
                        emis10 = estimate_landsat_emissivity.estimate_landsat_emissivity(dataOut,MTL_file, sensor='TIRS10', convert='yes')
                        emis11 = estimate_landsat_emissivity.estimate_landsat_emissivity(dataOut,MTL_file,sensor='TIRS11',convert='no')
                 
                        
                try:   
                    dataOut.setRad('e10',emis10)
                    dataOut.setRad('e11',emis11)
                    
                    # calculate SW
                    print('  Calculating SW') 
                    SW_LST, SW_LST_CWV, cwv, cwv_range = calc_SW.calcSW(dataOut, emis10, emis11)
                    dataOut.setRad('SW_LST',SW_LST)
                    dataOut.setRad('SW_LST_CWV',SW_LST_CWV)
                    
                    print('  Calculating SW uncertainty map')
                    SW_error, SW_error_cov, termT,termE,termTcov,termEcov = calcError.calculateError(dataOut, emis10, emis11, cwv_range)
                    dataOut.setRad('SW_error',SW_error_cov)
                    
                    #clouds = dataOut.getRad('cloud').astype(int) 
                    
                    
                    # save georegistered files... still to be done
                    SaveGeotif.saveGeotif(emis10, folder + '/' + folderID, sceneID +'_emis10.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(emis11, folder + '/' + folderID, sceneID +'_emis11.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(SW_LST, folder + '/' + folderID, sceneID +'_SW_LST.tif', wkt_projection, geoTransform)
                    #SaveGeotif.saveGeotif(clouds, folder + '/' + folderID, sceneID +'_CloudMask.tif', wkt_projection, geoTransform)
                    #SaveGeotif.saveGeotif(SW_error, folder + '/' + folderID, sceneID +'_SW_error.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(SW_error_cov, folder + '/' + folderID, sceneID +'_SW_error_cov.tif', wkt_projection, geoTransform)
                    #SaveGeotif.saveGeotif(dataOut.getRad('SC_LST'), folder + '/' + folderID, sceneID +'_SC_LST_Kelvin.tif', wkt_projection, geoTransform)
                  
                    SaveGeotif.saveGeotif(dataOut.getRad('t10'), folder + '/' + folderID, sceneID +'_T10.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(dataOut.getRad('t11'), folder + '/' + folderID, sceneID +'T11.tif', wkt_projection, geoTransform)
                  
                    
                    #SaveGeotif.saveGeotif(termT, folder + '/' + folderID, sceneID +'_termT.tif', wkt_projection, geoTransform)
                    #SaveGeotif.saveGeotif(termE, folder + '/' + folderID, sceneID +'_termE.tif', wkt_projection, geoTransform)
                    #SaveGeotif.saveGeotif(termTcov, folder + '/' + folderID, sceneID +'_termTcov.tif', wkt_projection, geoTransform)
                    #SaveGeotif.saveGeotif(termEcov, folder + '/' + folderID, sceneID +'_termEcov.tif', wkt_projection, geoTransform)
                    
                    
                    # get Surfrad info
                    siteInfo = createSiteInfo.CreateSiteInfo()
                  
                    data_SR = SurfradData.getSurfradValues(MTL_file, pathrow, siteInfo)
                    SurfradData.writeDataToFile(data_SR, dataOut, folder, folderID, sceneID, cwv)
                
                except:
                    print('  Something went wrong with ' + sceneID)
                        
            else:
                print('  ' + sceneID + '_SW_LST already calculated')
                
    elif answer == 'single':
        
        image_size_to_save = 10   #km of output image
        
               
        # get list of landst scenes
        dir_downloads = os.listdir(folder) 
        
        for sceneID in dir_downloads:
            print('  Working with: '+sceneID)
            folderID = sceneID
            files_in_folder = os.listdir(folder+'/' + folderID) 
            
            for ele in files_in_folder:
                if 'MTL' in ele:
                    MTL_file = ele
                    
            for ele in files_in_folder:
                if 'T2_B10' in ele:
                    T2_B10_file = ele
                    
            sceneID = T2_B10_file[0:len(T2_B10_file)-8]
                    
            #sceneID = MTL_file[0:len(MTL_file)-8]
            
            
            MTL_file =  folder + '/' + folderID + '/' +  MTL_file              
            SW_already_calculated = folder + '/' + folderID + '/' + sceneID +'_SW_LST_sml.tif'
        
            if not os.path.isfile(SW_already_calculated):
            
                if len(sceneID) == 21:
                    pathrow = sceneID[3:9]
                else:  
                    pathrow = sceneID[10:16]
                
                siteInfo = createSiteInfo.CreateSiteInfo()
                
                
                try:
                # find site data based on site name
                    line = siteInfo.pathrow.index(pathrow)
                    site_name = siteInfo.sitename[line]
                    #lat_value = siteInfo.lat[line]
                    #lon_value = siteInfo.lon[line]
                    astername = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_perSite/' + str(image_size_to_save) + 'km/e13_e14_ndvi_registered_' + site_name + '.tif'
                    
                except:
                    print('  Seems like ' + sceneID + ' is not in site list. Please add with "createSiteInfo.py".')
          
                #create small landsat scene 
                    
                
                try:
                    
                    # calculate emissivity - save in folder and write to class
                          
                    print('  Calculating Landsat Emissivity')
                    

                    # open all files for emissivityand SW calc into class
                    dataOut, wkt_projection, geoTransform = saveL8AsterInClass.dataForEmisCalc_single(astername, pathrow, folder, folderID, sceneID, site_name, image_size_to_save)
                    emis10 = estimate_landsat_emissivity.estimate_landsat_emissivity(dataOut,MTL_file, sensor='TIRS10', convert='yes')
                    emis11 = estimate_landsat_emissivity.estimate_landsat_emissivity(dataOut,MTL_file,sensor='TIRS11', convert='no')
                    
                    dataOut.setRad('e10',emis10)
                    dataOut.setRad('e11',emis11)
                    
                    # calculate SW
                    print('  Calculating SW')
                    SW_LST, SW_LST_CWV, cwv, cwv_range = calc_SW.calcSW(dataOut, emis10, emis11)
                    dataOut.setRad('SW_LST',SW_LST)
                    dataOut.setRad('SW_LST_CWV',SW_LST_CWV)
                    
                    print('  Calculating SW uncertainty map')
                    SW_error, SW_error_cov = calcError.calculateError(dataOut, emis10, emis11, cwv_range)
                    dataOut.setRad('SW_error',SW_error_cov)
                    
                    clouds = dataOut.getRad('cloud').astype(int) 
                                 
                    
                    # save georegistered files... still to be done
                    SaveGeotif.saveGeotif(emis10, folder + '/' + folderID, sceneID +'_emis10_sml.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(emis11, folder + '/' + folderID, sceneID +'_emis11_sml.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(SW_LST, folder + '/' + folderID, sceneID +'_SW_LST_sml.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(SW_LST_CWV, folder + '/' + folderID, sceneID +'_SW_LST_CWV_sml.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(clouds, folder + '/' + folderID, sceneID +'_CloudMask_sml.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(SW_error, folder + '/' + folderID, sceneID +'_SW_error_sml.tif', wkt_projection, geoTransform)
                    SaveGeotif.saveGeotif(SW_error_cov, folder + '/' + folderID, sceneID +'_SW_error_cov_sml.tif', wkt_projection, geoTransform)
                   
                    data_SR = SurfradData.getSurfradValues(MTL_file, pathrow, siteInfo)
                    SurfradData.writeDataToFile(data_SR, dataOut, folder, folderID, sceneID, cwv)
                    
                except:
                    print('  Something went wrong with ' + sceneID)

                    
#calcEmisAndSW()

