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
import calc_LST_with_SW
import osgeo.gdal as gdal
import warnings
import pdb



# create folder open popup
def popup_get_folder(dataOut, folder):
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory()
    dataOut[folder] = root.foldername
    return dataOut
    root.destroy()


def read_data(filename):
    
    gdal.UseExceptions()
    rd=gdal.Open(filename)
    band= rd.GetRasterBand(1)
    data = band.ReadAsArray()
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0
    
    return  data.astype(float)

# find sceneID, pathrow, and MTL_file 
def add_files_to_dictionary(dataOut):
    
    files_in_folder = os.listdir(dataOut['folder'])
    
    for ele in files_in_folder:
        if '_B10_resampled' in ele:
            filename = dataOut['folder'] + '/' + ele
            dataOut['rad10'] = read_data(filename)
            gdal.UseExceptions()
            # save georegistration details of file
            rd=gdal.Open(filename)
            dataOut['wkt_projection'] =rd.GetProjection()
            dataOut['geoTransform'] = rd.GetGeoTransform()
            dataOut['sceneID'] = ele[0:40] #ele[0:len(ele)-8]
        if '_B11_resampled' in ele:
            filename = dataOut['folder'] + '/' + ele    
            dataOut['rad11'] = read_data(filename)
            
    files_in_folder = os.listdir(dataOut['folder_emis'])
    
    for ele in files_in_folder:
        
        if 'emis10_resampled' in ele:   
            #pdb.set_trace()
            filename = dataOut['folder_emis'] + '/' + ele  
            dataOut['emis10'] = read_data(filename)
        if 'emis11_resampled' in ele:   
            filename = dataOut['folder_emis'] + '/' + ele  
            dataOut['emis11'] = read_data(filename)
            
        # if 'e13_e14_ndvi_registered' in ele:   
        #     #pdb.set_trace()
        #     filename = dataOut['folder_emis'] + '/' + ele  
        #     dataOut['emis11'] = read_data(filename)
        # if 'emis11_resampled' in ele:   
        #     filename = dataOut['folder_emis'] + '/' + ele  
        #     dataOut['emis11'] = read_data(filename)

        
    return dataOut

    
def save_Geotif(dataOut, filename, savename):
    
    data = dataOut[filename]
    width = data.shape[0]
    height = data.shape[1]
    bands = 1  
    
    folder = dataOut['folder'] + '/' + dataOut['sceneID'] + '_' + savename + '.TIF'
    
    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(folder, height, width, bands, gdal.GDT_Float32)
    ds.SetGeoTransform(dataOut['geoTransform'])
    ds.SetProjection(dataOut['wkt_projection'])
    ds.GetRasterBand(1).WriteArray(data)
    
    ds.FlushCache()
    ds = None
    
    print('File ' + filename + ' saved')
    


def main():
    
    warnings.simplefilter('ignore')
    
    # create dictionary to hold variables, and load variables needed for calculations
    dataOut = {}
    
    dataOut['variables'] = {}
    dataOut['variables']['L8'] = {} 
    dataOut['variables']['L8']['SW_coefficients'] = [2.2925,0.9929,0.1545,-0.3122,3.7186,0.3502,-3.5889,0.1825]
    
    # get folder for landsat band 10 and 11
    folder = 'folder'
    dataOut = popup_get_folder(dataOut, folder)
    
    # get folder for landsat emis10 and 11
    folder = 'folder_emis'
    dataOut = popup_get_folder(dataOut, folder)

    # load thermal bands and emis files
    dataOut = add_files_to_dictionary(dataOut)
    
    # calculate SW Land Surface Temperature
    dataOut = calc_LST_with_SW.main(dataOut)

    # save files
    save_Geotif(dataOut, 'SW_LST', 'SW_LST_aster_emis')

    

if __name__=="__main__":
    main()