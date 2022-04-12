#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:02:06 2020

@author: tkpci
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:38:41 2019

@author: tkpci
"""

from tkinter import *   
from tkinter import filedialog 
import downloadAster
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
import BuoyData
import BuoyDataNina
import pdb
import warnings
from multiprocessing import Process, Manager,Pool, cpu_count
from datetime import datetime

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/")
    return root.foldername
    root.destroy()
    

############################### RUN FUNCTIONS ######################    

def calcEmisAndSW(folder, sceneID,scene_done):

       
    try:
        print('  Working with scene: '+sceneID)
        
        
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
        #SW_already_calculated = folder + '/' + folderID + '/' + sceneID +'_SW_LST.tif'
        for scene_d in scene_done:
            SW_already_calculated_temp = folder + '/' + folderID + '/' + scene_d +'_SW_LST.tif'
            if not os.path.isfile(SW_already_calculated_temp):
                SW_already_calculated = folder + '/' + folderID + '/' + scene_d +'_SW_LST_new.tif'
            else:
                SW_already_calculated = folder + '/' + folderID + '/' + scene_d +'_SW_LST.tif'
                break
                
                
        
        
        if not os.path.isfile(SW_already_calculated):
        
            if len(sceneID) == 21:
                pathrow = sceneID[3:9]
            else:  
                pathrow = sceneID[10:16]
                
            aster_already_downloaded = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_' + pathrow + '.tif'
    
        
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
    
            
            try:
            
            # calculate emissivity - save in folder and write to class
                
                #print('  Calculating Landsat Emissivity')
                
                
                # open all files for emissivityand SW calc into class
                dataOut, wkt_projection, geoTransform = saveL8AsterInClass.dataForEmisCalc(pathrow, folder, folderID, sceneID)
                emis10 = estimate_landsat_emissivity.estimate_landsat_emissivity(dataOut,MTL_file, sensor='TIRS10', convert='yes')
                emis11 = estimate_landsat_emissivity.estimate_landsat_emissivity(dataOut,MTL_file,sensor='TIRS11',convert='no')
                
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
                # #SaveGeotif.saveGeotif(clouds, folder + '/' + folderID, sceneID +'_CloudMask.tif', wkt_projection, geoTransform)
                # #SaveGeotif.saveGeotif(SW_error, folder + '/' + folderID, sceneID +'_SW_error.tif', wkt_projection, geoTransform)
                # SaveGeotif.saveGeotif(SW_error_cov, folder + '/' + folderID, sceneID +'_SW_error_cov.tif', wkt_projection, geoTransform)
                # SaveGeotif.saveGeotif(dataOut.getRad('SC_LST'), folder + '/' + folderID, sceneID +'_SC_LST_Kelvin.tif', wkt_projection, geoTransform)
                # SaveGeotif.saveGeotif(ndvi, folder + '/' + folderID, sceneID +'_ndvi.tif', wkt_projection, geoTransform)
    
                #get Surfrad info
                # siteInfo = createSiteInfo.CreateSiteInfo()
              
                # data_SR = SurfradData.getSurfradValues(MTL_file, pathrow, siteInfo)
                # SurfradData.writeDataToFile(data_SR, dataOut, folder, folderID, sceneID, cwv)
                
                # # data_Buoy = BuoyData.getBuoyValues(MTL_file, pathrow, siteInfo)
                # # BuoyData.writeDataToFile(data_Buoy, dataOut, folder, folderID, sceneID, cwv)
                
                data_Buoy = BuoyDataNina.getBuoyValues(MTL_file, pathrow, folderID)
                BuoyDataNina.writeDataToFile(data_Buoy, dataOut, folder, folderID, sceneID, cwv)
                
            except:
                print('  Something went wrong with ' + sceneID)
                
        else:
            print('  ' + sceneID + '_SW_LST already calculated')
    except:
        print('Something went wrong')
                    
            
  


def main():
    # select folder with Landsat Scenes  
    folder = popup_get_folder()

    warnings.simplefilter('ignore')
    
    csv_file = '/dirs/data/tirs/downloads/Buoy_level2_downloads/Nina L8_C2_L1_BuoyScenes_2020-2021_Tania.csv'
    
    
    f = open(csv_file, "r")
    scene_done = f.read().split("\n")

    
    startTime = datetime.now()
    
    # get list of landst scenes
    dir_downloads = os.listdir(folder) 
    
    sceneID = 'LC80360262014325LGN01'
    
    # for sceneID in dir_downloads:
    #     print(sceneID)
    #     calcEmisAndSW(folder, sceneID)
    
    
    myProcesses = []
    
    
    for sceneID in dir_downloads:  
        print(sceneID)
        myProcesses.append(Process(target=calcEmisAndSW, args=(folder, sceneID,scene_done,)))

        
    #print(str(val+1) + " instances created")
    #counter1 = 0
    
    cores = 3
    iter = int(len(dir_downloads)/cores)
    print("Running " + str(cores) + " processes at a time")
    
    for i in range(iter+1):
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > len(dir_downloads):
            end_cnt = len(dir_downloads)
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
            
    print('\nTime elasped: ', datetime.now() - startTime)
        
#main()
