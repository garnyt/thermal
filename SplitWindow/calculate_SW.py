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
import download_Aster
import add_landsat_aster_to_dict_for_analysis
import stack_images
import calculate_landsat_emissivity
import calc_LST_with_SW
import SW_uncertainty
import osgeo.gdal as gdal
import matplotlib.pyplot as plt
import pdb
from multiprocessing import Process, Manager,Pool, cpu_count



# load all variables for SW calculations (update this json file with correct filepaths)
def load_variables_json_file(dataOut):
    
    # load variables
    # change this path to where file lie
    with open("SW_coefficients_ALL.json", 'r') as openfile:
        # Reading from json file into dictionary
        variables = json.load(openfile)
        
    dataOut['variables'] = variables
    
    return dataOut


# create folder open popup
def popup_get_folder(dataOut):
    
    root = Tk()
    root.withdraw() 
    #root.foldername = filedialog.askdirectory(initialdir = dataOut['variables']['initialdir'])
    root.foldername = filedialog.askdirectory(initialdir = '/dirs/data/tirs/')
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
            #pdb.set_trace()
            if len(sceneID) == 21:
                landsat = '0' + str(sceneID[2])
            else:
                landsat = sceneID[2:4]
            MTL_file = dataOut['folder'] + '/' + MTL_file
            
    if len(sceneID) == 21:
        pathrow = sceneID[3:9]
    else:  
        pathrow = sceneID[10:16]
        
    
    dataOut['landsat']  = landsat  
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
        asterCube, coordsCube = download_Aster.downLoadAster(dataOut, coord)
        download_Aster.georegisterAndSaveAster(dataOut, asterCube, coordsCube)
        
    else:
        print('  Aster already downloaded for path row '+ dataOut['pathrow'])
    
    # resample and stack aster to tirs, and save in landsat scene directory 
    stack_images.stackLandsatAster(dataOut)
    
    
def save_Geotif(dataOut, filename):
    
    data = dataOut[filename]
    width = data.shape[0]
    height = data.shape[1]
    bands = 1  
    
    if dataOut['filter'] == 'no':
        filename = filename + '_no_filter'
    
    file = dataOut['folder'] + '/' + dataOut['sceneID'] + '_' + filename + '.TIF'
    
    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(file, height, width, bands, gdal.GDT_Float32)
    ds.SetGeoTransform(dataOut['geoTransform'])
    ds.SetProjection(dataOut['wkt_projection'])
    ds.GetRasterBand(1).WriteArray(data)
    
    ds.FlushCache()
    ds = None
    
    if 'SW' in filename:
        print('File ' + dataOut['sceneID'] + '_' +filename + ' saved')
    


def main_single(smooth = 'yes', emis_water = 'no'):
    
    # create dictionary to hold variables, and load variables needed for calculations
    dataOut = {}
    dataOut = load_variables_json_file(dataOut)
    
    # get folder for landsat scene
    dataOut = popup_get_folder(dataOut)

    # get MTL file info, pathrow, and sceneID, and Landsat #
    dataOut = get_Landsat_files_and_info(dataOut)
    
  
    
    # check if ASTER tiles has been downloaded, and if not, download, stack, resample to tirs, and save
    check_and_download_ASTER_tiles(dataOut)

    # open all files needed for analysis and save to dictionary 
    dataOut = add_landsat_aster_to_dict_for_analysis.main(dataOut)

    # caluclate landsat emissivity
    dataOut = calculate_landsat_emissivity.main(dataOut)

    # calculate SW Land Surface Temperature
    dataOut['filter'] = smooth    # 'No' for no filtering
    dataOut = calc_LST_with_SW.main(dataOut)
    
    # calculate uncertainty
    dataOut = SW_uncertainty.main(dataOut)
    
    #pdb.set_trace()

    # save files
    save_Geotif(dataOut, 'SW_LST')
    save_Geotif(dataOut, 'SW_LST_qa')
    save_Geotif(dataOut, 'emis10')
    save_Geotif(dataOut, 'emis11')
    save_Geotif(dataOut, 'T10')
    save_Geotif(dataOut, 'T11')
    save_Geotif(dataOut, 'rad10')
    save_Geotif(dataOut, 'rad11')
    #save_Geotif(dataOut,'ls_emis_data_10')
    #save_Geotif(dataOut, 'ls_emis_data_11')
    
    plt.imshow(dataOut['SW_LST'], vmin = 260)
    plt.axis('off')
    plt.colorbar()
    
    
    # save_Geotif(dataOut, 'SW_LST_no_filter')

def calc_SW(dataOut, folder, smooth = 'yes'):
    
    try:
        water_emis = dataOut['water_emis']
        dataOut['folder'] = dataOut['folder_main'] + '/' + folder
        dataOut = get_Landsat_files_and_info(dataOut)
        
        #pdb.set_trace()
        
        if water_emis =='no':
            # check if ASTER tiles has been downloaded, and if not, download, stack, resample to tirs, and save
            check_and_download_ASTER_tiles(dataOut)
    
        # open all files needed for analysis and save to dictionary 
        dataOut = add_landsat_aster_to_dict_for_analysis.main(dataOut)
    
        if water_emis =='no':
            # caluclate landsat emissivity
            dataOut = calculate_landsat_emissivity.main(dataOut)
    
        # calculate SW Land Surface Temperature
        dataOut = calc_LST_with_SW.main(dataOut)
        
        if water_emis =='no':
            # calculate uncertainty
            dataOut = SW_uncertainty.main(dataOut)
            save_Geotif(dataOut, 'SW_LST_qa')
        
        #pdb.set_trace()
    
        # save files
        save_Geotif(dataOut, 'SW_LST')
        save_Geotif(dataOut, 'emis10')
        save_Geotif(dataOut, 'emis11')
        save_Geotif(dataOut, 'T10')
        save_Geotif(dataOut, 'T11')
        save_Geotif(dataOut, 'rad10')
        save_Geotif(dataOut, 'rad11')
        #save_Geotif(dataOut,'ls_emis_data_10')
        #save_Geotif(dataOut, 'ls_emis_data_11')
        

    except:
        print('Scene not working')
        print(dataOut['sceneID'])
        
        
            
def main(smooth, emis_water):
    
    dataOut = {}
    dataOut = load_variables_json_file(dataOut)
    
    # get folder where landsat scenes are 
    dataOut = popup_get_folder(dataOut)
    dataOut['folder_main'] = dataOut['folder']
    dataOut['filter'] = smooth
    dataOut['water_emis'] = emis_water

    folders = os.listdir(dataOut['folder_main'])
    
    myProcesses = []
    profiles = len(folders)
    
    for folder in folders:
        myProcesses.append(Process(target=calc_SW, args=(dataOut, folder, smooth,)))

        
    print(str(profiles) + " instances created")
    
    cores = 10
    
    iter = int(profiles/cores)
    print("Running " + str(cores) + " processes at a time")
    
    for i in range(iter+1):
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > profiles:
            end_cnt = profiles
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
            
        process.close() 

if __name__=="__main__":
    smooth = 'yes'  # or 'yes' to apply 5x5 averaing filter on SW diff term
    emis_water = 'yes' # use yes to apply only water emissivity for whole scene - correct emissivity for L8 and L9  bands will be used from the spec files
    main(smooth, emis_water)