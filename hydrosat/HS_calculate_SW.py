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
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import HS_download_Aster
import HS_add_landsat_aster_to_dict_for_analysis
import HS_stack_images
import HS_calculate_landsat_emissivity
import HS_calc_LST_with_SW
import osgeo.gdal as gdal



# load all variables for SW calculations (update this json file with correct filepaths)
def load_variables_json_file(dataOut):
    
    # load variables
    # change this path to where file lies
    filepath = '/cis/staff/tkpci/Code/Python/hydrosat/'
    with open(filepath + "fixed_SW_code_variables.json", 'r') as openfile:
        # Reading from json file into dictionary
        variables = json.load(openfile)
        
    dataOut['variables'] = variables
    
    return dataOut


# create folder open popup
def popup_get_folder(dataOut):
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory(initialdir = dataOut['variables']['initialdir'])
    dataOut['folder'] = root.foldername
    return dataOut
    root.destroy()


# find sceneID, pathrow, and MTL_file 
def get_Landsat_files_and_info(dataOut):
    
    files_in_folder = os.listdir(dataOut['folder'])
    
    for ele in files_in_folder:
        if 'MTL.txt' in ele:
            MTL_file = ele
            sceneID = MTL_file[0:len(MTL_file)-8]
            MTL_file = dataOut['folder'] + '/' + MTL_file
            
    if len(sceneID) == 21:
        pathrow = sceneID[3:9]
    else:  
        pathrow = sceneID[10:16]
        
    dataOut['pathrow'] = pathrow
    dataOut['sceneID'] = sceneID
    dataOut['MTL_file'] = MTL_file
        
    return dataOut

# check if ASTER tiles has been downloaded - and if not, download based on MTL file coordinates
def check_and_download_ASTER_tiles(dataOut):
    
    # check if ASTER has already been downloaded
    aster_already_downloaded = dataOut['variables']['aster_filepath'] + 'e13_e14_NDVI_' + dataOut['pathrow'] + '.tif'
    
    # if not downloaded, get coordinates from MTL file and download ASTER
    if not os.path.isfile(aster_already_downloaded):
    
        # get corner coordinates from MTL file
        f = open(dataOut['MTL_file'],"r")
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
        
        # get corners of landsat scene
        coord = [max(CORNER_UL_LAT_PRODUCT,CORNER_UR_LAT_PRODUCT), \
                 min(CORNER_UL_LON_PRODUCT,CORNER_LL_LON_PRODUCT),
                 min(CORNER_LR_LAT_PRODUCT,CORNER_LL_LAT_PRODUCT),
                 max(CORNER_LR_LON_PRODUCT,CORNER_UR_LON_PRODUCT)]
    
        #download and save stacked georegistered aster files
        asterCube, coordsCube = HS_download_Aster.downLoadAster(dataOut, coord)
        HS_download_Aster.georegisterAndSaveAster(dataOut, asterCube, coordsCube)
        
    else:
        print('  Aster already downloaded for path row '+ dataOut['pathrow'])
    
    # resample and stack aster to tirs, and save in landsat scene directory 
    HS_stack_images.stackLandsatAster(dataOut)
    
    
def save_Geotif(dataOut, filename):
    
    data = dataOut[filename]
    width = data.shape[0]
    height = data.shape[1]
    bands = 1  
    
    file = dataOut['folder'] + '/' + dataOut['sceneID'] + '_' + filename + '.TIF'
    
    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(file, height, width, bands, gdal.GDT_Float32)
    ds.SetGeoTransform(dataOut['geoTransform'])
    ds.SetProjection(dataOut['wkt_projection'])
    ds.GetRasterBand(1).WriteArray(data)
    
    ds.FlushCache()
    ds = None
    
    print('File ' + filename + ' saved')
    


def main():
    
    # create dictionary to hold variables, and load variables needed for calculations
    dataOut = {}
    dataOut = load_variables_json_file(dataOut)
    
    # get folder for landsat scene
    dataOut = popup_get_folder(dataOut)

    # get MTL file info, pathrow, and sceneID
    dataOut = get_Landsat_files_and_info(dataOut)
    
    # check if ASTER tiles has been downloaded, and if not, download, stack, resample to tirs, and save
    check_and_download_ASTER_tiles(dataOut)

    # open all files needed for analysis and save to dictionary 
    dataOut = HS_add_landsat_aster_to_dict_for_analysis.main(dataOut)

    # caluclate landsat emissivity
    dataOut = HS_calculate_landsat_emissivity.main(dataOut)

    # calculate SW Land Surface Temperature
    dataOut = HS_calc_LST_with_SW.main(dataOut)

    # save files
    save_Geotif(dataOut, 'SW_LST')
    save_Geotif(dataOut, 'emis10')
    save_Geotif(dataOut, 'emis11')
    

if __name__=="__main__":
    main()