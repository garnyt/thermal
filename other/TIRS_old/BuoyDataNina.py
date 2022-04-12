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


def getBuoyValues(MTL_file, pathrow, folderID):
    
    
    # for each scene, get time and date from MTL file
    f = open(MTL_file,"r")
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
    
   
    folderID_short = folderID[:-5]
   
    #open buoy csv
    data = pd.read_csv("/dirs/data/tirs/downloads/Buoy_level2_downloads/Nina L8_C2_L1_BuoyScenes_2020-2021_Tania.csv")
    
    #pdb.set_trace()
    #rows = data.loc[data['scene_id_short'] == folderID_short]
    rows = data.loc[data['scene_id'] == folderID]
    
    data_Buoy = dict()
    data_Buoy['Buoy_id'] = []
    data_Buoy['Buoy_date']= []
    data_Buoy['Buoy_lat'] = []
    data_Buoy['Buoy_lon'] = []
    data_Buoy['Buoy_temperature'] = []

    
    try:
    
        for index, line in rows.iterrows():
          
            
            data_Buoy['Buoy_id'].append(line['buoy_id'])
            data_Buoy['Buoy_date'].append(line['date'])            
            data_Buoy['Buoy_lat'].append(line['lat'])
            data_Buoy['Buoy_lon'].append(line['long'])
            data_Buoy['Buoy_temperature'].append(line['buoy_temp _C2'])

            
    except:
         print('Could not find buoy site in this scene')       
    

    return data_Buoy
    

def writeDataToFile(data_Buoy, dataOut, folder, folderID, sceneID,cwv=-9999):

 
    buoy_temp = data_Buoy['Buoy_lat']
       
    for rows in range(len(buoy_temp)):
        # get points from files    
        lat_value = data_Buoy['Buoy_lat'][rows]
        lon_value = data_Buoy['Buoy_lon'][rows]
        
        SW_file = folder + '/' + folderID + '/' + sceneID +'_SW_LST.tif'
        pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + SW_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
        
          
        temp = pixelsLocate.decode()
    
        numbers = re.findall('\d+',temp) 
    
        x = int(numbers[0])
        y = int(numbers[1])
        
        data_L8 = {}
        data_L8['Scene_ID'] = sceneID
        data_L8['L8_rad10'] = np.round(dataOut.getRad('rad10')[y,x],3)
        data_L8['L8_rad11'] = np.round(dataOut.getRad('rad11')[y,x],3)
        data_L8['L8_AppTemp10'] = np.round(dataOut.getRad('t10')[y,x],3)
        data_L8['L8_AppTemp11'] = np.round(dataOut.getRad('t11')[y,x],3)
        data_L8['L8_emis10'] = np.round(dataOut.getRad('e10')[y,x],3)
        data_L8['L8_emis11'] = np.round(dataOut.getRad('e11')[y,x],3)
        data_L8['L8_SW_LST'] = np.round(dataOut.getRad('SW_LST')[y,x],3)
        try:
            data_L8['L8_SW_LST_CWV'] = np.round(dataOut.getRad('SW_LST_CWV')[y,x],3)
        except:
            data_L8['L8_SW_LST_CWV'] = 0
            
        try:
            data_L8['L8_SW_uncertainty'] = np.round(dataOut.getRad('SW_error')[y,x],3)
        except:
            data_L8['L8_SW_uncertainty'] = 0 
            
        try:
            data_L8['SC_QA'] = np.round(dataOut.getRad('SC_QA')[y,x],3)
        except:
            data_L8['SC_QA'] = 0 
            
        try:
            data_L8['SC_LST'] = np.round(dataOut.getRad('SC_LST')[y,x],3)
        except:
            data_L8['SC_LST'] = 0 
            
        try:
            data_L8['SC_CDIST'] = np.round(dataOut.getRad('SC_CDIST')[y,x],3)
        except:
            data_L8['SC_CDIST'] = 0 
          
         
        dist, dist_bin = distToNearestCloud(dataOut,x,y)
        
        if dist == 'N/A':     
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
            ",dist_to_cloud" + ",cloud_bin" + ",cwv" + ",BuoylessL8_LST" + ",BuoylessSC_LST"
        values = ",".join(str(e[rows]) for e in data_Buoy.values()) + "," +  \
            ",".join(str(f) for f in data_L8.values()) + ',' + str(dist) + ',' + str(dist_bin) +  \
            ','  + str(cwv)+ ',' + str(np.round(data_Buoy['Buoy_temperature'][rows]-data_L8['L8_SW_LST'],3)) + ',' + str(np.round(data_Buoy['Buoy_temperature'][0]-data_L8['SC_LST'],3))
        
        
        
        # write data to txt file
        filename_out = '/dirs/data/tirs/downloads/Buoy_results/buoyNina_analysis_2020_2021.csv'
        
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
                
    
def distToNearestCloud(dataOut,x,y):
    
    try:
    
        # cloud = 0, nocloud  = 1
        cloud = dataOut.getRad('cloud')
        
        ind = np.argwhere(cloud == 0)
        
        if ind.shape[0] == 0:
            dist = np.sqrt(((cloud.shape[0]/2)**2)*2) * 30 / 1000
            
            
        else:
            distances = np.sqrt((ind[:,0] - y) ** 2 + (ind[:,1] - x) ** 2)
            nearest_index = np.argmin(distances)
         
            # distance in km
            dist = distances[nearest_index] * 30 / 1000   # 30 meter pixels to km
            
        if cloud.shape[0] > 1000:
            dist = 'N/A'
        
        # distance bin (0, 0-0.2, 0.2-1, 1-5, 5-10, >10)
        if dist > 10:
            dist_bin = '10+ km '
        elif dist > 5 and dist <= 10:
            dist_bin = '5 to 10 km'
        elif dist > 1 and dist <= 5:
            dist_bin = '1 to 5 km'
        elif dist > 0.2 and dist <= 1:
            dist_bin = '0.2 to 1 km'
        elif dist > 0 and dist <= 0.2:
            dist_bin = '0 to 0.2 km'
        else:
            dist_bin = '0 km' 
    except:
         dist = 'N/A' 
         dist_bin = 'N/A'
    
    return dist, dist_bin

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    