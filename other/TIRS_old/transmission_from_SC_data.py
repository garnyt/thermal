#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:43:48 2021

@author: tkpci
"""

from tkinter import *   
from tkinter import filedialog 
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import warnings
import createSiteInfo
import subprocess
from osgeo import gdal
import SurfradData
import re
import BuoyDataNina


def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/")
    return root.foldername
    root.destroy()
    
def calcTransmission(folder, sceneID):

       
    try:
        
        print('  Working with scene: '+sceneID)
        
        
        folderID = sceneID
        files_in_folder = os.listdir(folder+'/'+folderID)
        
        for ele in files_in_folder:
            if 'ATRAN.TIF' in ele:
                trans_file = ele
            if 'MTL.txt' in ele:
                MTL_file = ele
                
        trans_file = folder + '/' + folderID + '/' + trans_file        
                        
        sceneID = MTL_file[0:len(MTL_file)-8]
         
        MTL_file = folder + '/' + folderID + '/' + MTL_file
        
        if len(sceneID) == 21:
                pathrow = sceneID[3:9]
        else:  
            pathrow = sceneID[10:16]
        
        siteInfo = createSiteInfo.CreateSiteInfo()
        # data_get = SurfradData.getSurfradValues(MTL_file, pathrow, siteInfo)
        # #get points from files    
        # lat_value = data_get['SR_lat']
        # lon_value = data_get['SR_lon']
        #pdb.set_trace()
        data_get = BuoyDataNina.getBuoyValues(MTL_file, pathrow, folderID)
        
        # get points from files    
        lat_value = data_get['Buoy_lat']
        lon_value = data_get['Buoy_lon']
        lat_value = lat_value[0]
        lon_value = lon_value[0]
        
        try: 
            pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + trans_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
        except:
            print('Cant find lat lon')
      
        temp = pixelsLocate.decode()
    
        numbers = re.findall('\d+',temp) 
    
        x = int(numbers[0])
        y = int(numbers[1])
        
        gdal.UseExceptions()
        rd=gdal.Open(trans_file)
        wkt_projection =rd.GetProjection()
        geoTransform= rd.GetGeoTransform()
        SW= rd.GetRasterBand(1)
        trans = SW.ReadAsArray()
    
        transmission =  trans[y,x] *0.0001  #scale factor accoring to ST product guide
        
        #write data to file
        headers = ",".join(data_get.keys()) +  ",SceneID" + ",transmission"
        values = ",".join(str(e) for e in data_get.values()) + ',' + sceneID + ',' + str(np.round(transmission,3))
                
        # write data to txt file
        #filename_out = '/dirs/data/tirs/downloads/Surfrad/results/analysis_2020_2021_collection2_level2_transmission.csv'
        filename_out = '/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_2020_2021_collection2_level2_transmission.csv'
       
        if not os.path.isfile(filename_out):
            
            with open(filename_out, mode='w') as file_out:
                csv.excel.delimiter=';'
                file_writer = csv.writer(file_out, dialect=csv.excel)
            
                file_writer.writerow([headers])
                file_writer.writerow([values])
                
        else:
            with open(filename_out, mode='a') as file_out:
                csv.excel.delimiter=';'
                file_writer = csv.writer(file_out, dialect=csv.excel)
            
                file_writer.writerow([values])
            
    
    
    except:
        print('Oops, something went wrong')


def main():
    # select folder with Landsat Scenes  
    folder = popup_get_folder()

    warnings.simplefilter('ignore')
    
       
    # get list of landst scenes
    dir_downloads = os.listdir(folder) 

    
    for sceneID in dir_downloads:  
        print(sceneID)
        calcTransmission(folder, sceneID)
        
if __name__ == "__main__":
    main()
        