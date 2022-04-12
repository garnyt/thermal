#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:22:01 2020

@author: tkpci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 08:21:34 2020

@author: tkpci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:03:37 2020

@author: tkpci
"""

import numpy as np
import pandas as pd
import os
import subprocess
import re 
import csv
import cv2
import pandas as pd
import pdb
from osgeo import gdal


def get_landsat_data(folder_c2l1,folder_c2l2,data_Buoy):
    
    # get points from files    
    lat_value = data_Buoy['Buoy_lat']
    lon_value = data_Buoy['Buoy_lon']
    
   
    sceneID = data_Buoy['sceneID']
    
    # locate surfrad site based on lat lon
    SW_file = folder_c2l1 + sceneID +'_SW_LST.TIF'
    pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + SW_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
    
      
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    x = int(numbers[0]) # column    
    y = int(numbers[1]) # row
    
    # open landsat files and get LST, app temp, radiance and emis values
    data_L8 = {}
    rad10 = open_tiff(folder_c2l1 + sceneID + '_rad10.TIF')
    rad11 = open_tiff(folder_c2l1 + sceneID + '_rad11.TIF')
    t10 = open_tiff(folder_c2l1 + sceneID + '_T10.TIF')
    t11 = open_tiff(folder_c2l1 + sceneID + '_T11.TIF')
    e10 = open_tiff(folder_c2l1 + sceneID + '_emis10.TIF')
    e11 = open_tiff(folder_c2l1 + sceneID + '_emis11.TIF')
    lst = open_tiff(folder_c2l1 + sceneID + '_SW_LST.TIF')
    
    
    # not averaging around pixel of interest as surfrad scenes does not have large uniform areas like water
    data_L8['L8_rad10'] = rad10[y,x]
    data_L8['L8_rad11'] = rad11[y,x]
    data_L8['L8_AppTemp10'] = t10[y,x]
    data_L8['L8_AppTemp11'] = t11[y,x]
    data_L8['L8_emis10'] = e10[y,x]
    data_L8['L8_emis11'] = e11[y,x]
    data_L8['L8_SW_LST'] = lst[y,x]
    data_L8['SC_LST'] = 0
    data_L8['SC_QA'] = 0
    data_L8['SC_CDIST'] = 0
    
    # get SC data and cloud dist
    files = os.listdir(folder_c2l2)
    for ele in files:
        if ele[-10:] == 'ST_B10.TIF':
            sc = open_tiff(folder_c2l2 + ele)
            # apply gain and bias
            sc = sc * 0.00341802 + 149.0 # Landsat 8-9 Collection 2 (C2) Level 2 Science Product (L2SP) Guide
            data_L8['SC_LST'] = sc[y,x]
        if ele[-12:] == 'ST_CDIST.TIF':
            cdist = open_tiff(folder_c2l2 + ele)
            # apply gain and bias
            cdist = cdist * 0.01 # scale factor according to Landsat 8-9 Collection 2 (C2) Level 2 Science Product (L2SP) Guide
            data_L8['SC_CDIST'] = cdist[y,x]
        if ele[-9:] == 'ST_QA.TIF':
            qa = open_tiff(folder_c2l2 + ele)    
            data_L8['SC_QA'] = qa[y,x] * 0.01
            
    
    return data_L8    
    
    
    
    
def open_tiff(full_filename):
    
    gdal.UseExceptions()
    rd=gdal.Open(full_filename)
    #wkt_projection =rd.GetProjection()
    #geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    img = data.ReadAsArray() 
    
    return img
    
    
def getBuoyValues(folder_c2l1,scene, csv_file):
    
    files_in_folder = os.listdir(folder_c2l1)
    
    for ele in files_in_folder:
        if 'MTL.txt' in ele:
            MTL_file = ele
            MTL_filepath = folder_c2l1  + MTL_file
            pathrow = MTL_file[10:16]
            sceneID = MTL_file[0:len(MTL_file)-8]
            break
        
    # for each scene, get time and date from MTL file
    f = open(MTL_filepath,"r")
    content = f.read()
    f.close()
    
    date_line = content.index('DATE_ACQUIRED')
    time_line = content.index('SCENE_CENTER_TIME')
    
    date_val = content[date_line+16:date_line+16+10].rstrip()
    
    time_val = content[time_line+21:time_line+21+8].rstrip()
    hours_L8 = float(time_val[0:2])
    min_L8 = float(time_val[3:5])
    time_L8 = hours_L8 + min_L8/60
    
    my_date = pd.to_datetime(date_val, format='%Y-%m-%d')
    newdate = str(my_date.month) + '/' + str(my_date.day) + '/'+ str(my_date.year)
   
    #open buoy csv
    data = pd.read_csv(csv_file)
    
    #rows = data.loc[data['scene_id_short'] == folderID_short]
    rows = data.loc[data['scene_id'] == scene]
    
    data_Buoy = dict()
    data_Buoy['Buoy_id'] = []
    data_Buoy['scene'] = []
    data_Buoy['sceneID'] = []
    data_Buoy['Buoy_lat'] = []
    data_Buoy['Buoy_lon'] = []
    data_Buoy['Buoy_temperature'] = []

    #pdb.set_trace()
    try:
    
        for index, line in rows.iterrows():
          
            
            data_Buoy['Buoy_id']= line['buoy_id']
            data_Buoy['scene'] = scene
            data_Buoy['sceneID']= sceneID            
            data_Buoy['Buoy_lat']= line['lat']
            data_Buoy['Buoy_lon'] = line['long']
            data_Buoy['Buoy_temperature'] = line['buoy_temp']

            
    except:
         print('Could not find buoy site in this scene')       
    

    return data_Buoy
    

def writeDataToFile(data_Buoy, data_L8, csv_file):
   
    # create cloud bins for analysis
    if data_L8['SC_CDIST'] > 10:
        dist_bin = '10+ km '
    elif data_L8['SC_CDIST'] > 5 and data_L8['SC_CDIST'] <= 10:
        dist_bin = '5 to 10 km'
    elif data_L8['SC_CDIST'] > 1 and data_L8['SC_CDIST'] <= 5:
        dist_bin = '1 to 5 km'
    elif data_L8['SC_CDIST'] > 0.2 and data_L8['SC_CDIST'] <= 1:
        dist_bin = '0.2 to 1 km'
    elif data_L8['SC_CDIST'] > 0 and data_L8['SC_CDIST'] <= 0.2:
        dist_bin = '0 to 0.2 km'
    else:
        dist_bin = '0 km' 

    headers = ",".join(data_Buoy.keys()) + "," + ",".join(data_L8.keys()) + \
        ",cloud_bin" + ",BuoylessL8_LST" + ",BuoylessSC_LST"
    values = ",".join(str(e) for e in data_Buoy.values()) + "," +  \
        ",".join(str(f) for f in data_L8.values()) + ',' + str(dist_bin) +  \
        ',' + str(np.round(data_Buoy['Buoy_temperature']-data_L8['L8_SW_LST'],3)) + \
        ',' + str(np.round(data_Buoy['Buoy_temperature']-data_L8['SC_LST'],3))
        

        
        
    # write data to txt file
    filename_out = csv_file[:-4]+ '_results.csv'
    
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
                


def main(csv_file,filepath_C2L1,filepath_C2L2):
    
    files = os.listdir(filepath_C2L1)

    for folder_c2l1 in files:
       print(folder_c2l1)
       
       try:
           folder_c2l2 = filepath_C2L2 + folder_c2l1 + '/'
       
           # get surfrad data for specific landsat scene
           data_Buoy = getBuoyValues(filepath_C2L1 + folder_c2l1 + '/', folder_c2l1, csv_file)
           
           # get associated pixel values for surfrad point
           data_L8 = get_landsat_data(filepath_C2L1 + folder_c2l1 + '/',folder_c2l2, data_Buoy)
           
           # write data to file
           writeDataToFile(data_Buoy, data_L8,csv_file)

       except:
           print('scene not working')
    
    
    
if __name__ in "main":
    csv_file = '/dirs/data/tirs/Landsat9/RIT_L9_BUOY_2022_03_01.csv'
    filepath_C2L1 = '/dirs/data/tirs/Landsat9/L9_C2L1_2021_202203/'
    filepath_C2L2 = '/dirs/data/tirs/Landsat9/L9_C2L2_2021_202203/'
    main(csv_file,filepath_C2L1,filepath_C2L2)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    