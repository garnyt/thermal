#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:51:46 2021

@author: tkpci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:18:15 2021

@author: tkpci
"""

from osgeo import gdal
import subprocess
import numpy as np
import pandas as pd
import os
import subprocess
import re 
import csv
import cv2
from tkinter import * 
import sys
sys.path.insert(1, '/cis/staff/tkpci/Code/Python/TIRS')
import createSiteInfo
from tkinter import filedialog 
import matplotlib.pyplot as plt

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/")
    return root.foldername
    root.destroy()


def getSurfradpixelno(LST_full_path, pathrow, siteInfo):
    
    line = siteInfo.pathrow.index(pathrow)
    site_name = siteInfo.sitename[line]
    short_name = siteInfo.shortname[line]
    lat = siteInfo.lat[line]
    lon = siteInfo.lon[line]
    
    # find pixel location based on lat lon coordinates of site
    pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + LST_full_path + ' ' + str(lon) + ' ' + str(lat), shell=True )
   
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    col = int(numbers[0])
    row = int(numbers[1])
    
    return short_name, col, row
    
    
def main():
    
 
        
    folder_level1 = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_2020_2021_collection2_level1/'
    folder_level2 = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_2020_2021_collection2_level2/'
    
    scenes = os.listdir(folder_level1)
    siteInfo = createSiteInfo.CreateSiteInfo()
    

    
    for scene in scenes:
        
        try:
        
            files_in_folder = os.listdir(folder_level1+'/'+scene)
            files_in_folder_level2 = os.listdir(folder_level2+'/'+scene)
            
            for ele in files_in_folder:
                if 'MTL.txt' in ele:
                    MTL_file = ele
                if 'SW_LST.tif' in ele:
                    LST_full_path = folder_level1+'/'+scene + '/' + ele   
                if 'B2.TIF' in ele:
                    B2_full_path = folder_level1+'/'+scene + '/' + ele
                if 'B3.TIF' in ele:
                    B3_full_path = folder_level1+'/'+scene + '/' + ele
                if 'B4.TIF' in ele:
                    B4_full_path = folder_level1+'/'+scene + '/' + ele
            
            for ele in files_in_folder_level2:
                if 'ST_CDIST.TIF' in ele:
                    CDIST_full_path = folder_level2+'/'+scene + '/' + ele 
            
            pathrow = MTL_file[10:16]
            
            short_name, col, row = getSurfradpixelno(LST_full_path, pathrow, siteInfo)
            
            short_name_l2, col_l2, row_l2 = getSurfradpixelno(CDIST_full_path, pathrow, siteInfo)
    
    
            gdal.UseExceptions()
            rd=gdal.Open(B2_full_path)
            wkt_projection =rd.GetProjection()
            geoTransform= rd.GetGeoTransform()
            img= rd.GetRasterBand(1)
            B2_data = img.ReadAsArray() #* 1.2594E-02 -62.97237  # radiance
            #B2_data = B2_data * 2.0000E-05 - 0.1  # reflectance
            B2_data = (B2_data - np.min(B2_data))/(np.max(B2_data) - np.min(B2_data))
            
            gdal.UseExceptions()
            rd=gdal.Open(B3_full_path)
            img= rd.GetRasterBand(1)
            B3_data = img.ReadAsArray()#* 1.1807E-02 -59.03316 # radiance
            #B3_data = B3_data * 2.0000E-05 - 0.1  # reflectance
            B3_data = (B3_data - np.min(B3_data))/(np.max(B3_data) - np.min(B3_data))
            
            gdal.UseExceptions()
            rd=gdal.Open(B4_full_path)
            img= rd.GetRasterBand(1)
            B4_data = img.ReadAsArray()#* 9.9560E-03 - 49.78088 # radiance
            #B4_data = B4_data * 2.0000E-05 - 0.1  # reflectance
            B4_data = (B4_data - np.min(B4_data))/(np.max(B4_data) - np.min(B4_data))
            
            gdal.UseExceptions()
            rd=gdal.Open(CDIST_full_path)
            wkt_projection =rd.GetProjection()
            geoTransform= rd.GetGeoTransform()
            img= rd.GetRasterBand(1)
            CDIST_data = img.ReadAsArray()
            
            RGB = np.zeros([B2_data.shape[0],B2_data.shape[1],3])
            RGB[:,:,0] = B4_data
            RGB[:,:,1] = B3_data
            RGB[:,:,2] = B2_data
            
            plt.subplot(1,2,1)
            plt.imshow(RGB)
            plt.axis('off')
            plt.plot(col, row, marker='v', color="red")

            plt.subplot(1,2,2)
            plt.imshow(CDIST_data, cmap = 'gray', vmin = 0, vmax = 20)
            plt.plot(col_l2, row_l2, marker='v', color="red")
            plt.axis('off')
            text = 'Distance to cloud: ',str(CDIST_data[row,col])
            plt.title(text)
            

            
            
        
    
    
    
    
    
            # data_SR['scene'] = scene    
            # data_SR['short_name'] = short_name
            # data_SR['col'] = col
            # data_SR['row'] = row
            
                
            # headers = ",".join(data_SR.keys()) 
            # values = ",".join(str(e) for e in data_SR.values())     
        
            # # write data to txt file
            # filename_out = '/dirs/data/tirs/code/matlab/CloudDistGUI/SurfRadCoordPerScene.csv'
                
            # if not os.path.isfile(filename_out):
                
            #     with open(filename_out, mode='w') as file_out:
            #         csv.excel.delimiter=';'
            #         file_writer = csv.writer(file_out, dialect=csv.excel)
                
            #         file_writer.writerow([headers])
            #         file_writer.writerow([values])
                    
            # else:
            #     with open(filename_out, mode='a') as file_out:
            #         csv.excel.delimiter=';'
            #         file_writer = csv.writer(file_out, dialect=csv.excel)
                
            #         file_writer.writerow([values])
        except:
            print(scene, ' not working')
                
        
        
        
        