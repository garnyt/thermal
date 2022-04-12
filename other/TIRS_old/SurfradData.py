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


def getSurfradValues(MTL_file, pathrow, siteInfo):
    
    line = siteInfo.pathrow.index(pathrow)
    site_name = siteInfo.sitename[line]
    short_name = siteInfo.shortname[line]
    lat = siteInfo.lat[line]
    lon = siteInfo.lon[line]
    
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
    new_year_day = pd.Timestamp(year=my_date.year, month=1, day=1)
    num_of_days = (my_date - new_year_day).days + 1
    year = str(my_date.year)
    
    # create 3 character day of year
    if num_of_days < 10:
        num_of_days_str = '00' + str(num_of_days)
    elif num_of_days < 100:
        num_of_days_str = '0' + str(num_of_days)
    else:
        num_of_days_str = str(num_of_days)

    
    #create download name
    ftp_path = 'ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/' + site_name + '/' + str(year) + '/'  
    ftp_name =  short_name + year[2:4] + num_of_days_str + '.dat'
    ftp_fullname = ftp_path + ftp_name
    
    ftp_dest = '/dirs/data/tirs/downloads/test/' + ftp_name
        
    import urllib 
    from contextlib import closing
    import shutil

    with closing(urllib.request.urlopen(ftp_fullname))as r:
        with open(ftp_dest, 'wb') as f:
            shutil.copyfileobj(r, f)       
            
    data = np.loadtxt(ftp_dest, skiprows=2)
    
    # find closest time to surfrad data  
    time_SR = data[:,6]
    index_closest = np.abs(time_SR-time_L8).argmin()
    
    data[index_closest,0:7]
    
    data_SR = {}
    data_SR['SR_sitename'] = site_name
    data_SR['SR_time'] = data[index_closest,6]
    data_SR['SR_lat'] = lat
    data_SR['SR_lon'] = lon
    data_SR['SR_solar_zen'] = data[index_closest,7]
    data_SR['SR_dw_ir'] = data[index_closest,16]
    data_SR['SR_uw_ir'] = data[index_closest,22]
    data_SR['SR_airtemp'] = data[index_closest,38]
    data_SR['SR_rh'] = data[index_closest,40]
    data_SR['SR_windspd'] = data[index_closest,42]
    data_SR['SR_winddir'] = data[index_closest,44]
    data_SR['SR_pressure'] = data[index_closest,46]
    data_SR['L8_time'] = time_L8
    data_SR['L8_date'] = date_val
    
    os.remove(ftp_dest)

    return data_SR
    

def writeDataToFile(data_SR, dataOut, folder, folderID, sceneID,cwv=-9999):

    # get points from files    
    lat_value = data_SR['SR_lat']
    lon_value = data_SR['SR_lon']
    
    try: 
        SW_file = folder + '/' + folderID + '/' + sceneID +'_SW_LST_sml.tif'
        pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + SW_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
    except:
        SW_file = folder + '/' + folderID + '/' + sceneID +'_SW_LST.tif'
        pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + SW_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
    
      
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    x = int(numbers[0]) # column    
    y = int(numbers[1]) # row
    
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
    
    # calculate SURFRAD LST
    SR_emis = (data_L8['L8_emis10'] + data_L8['L8_emis11'])/2
    stepbol = 5.670367*10**(-8)
    SR_LST = ((data_SR['SR_uw_ir'] -(1-SR_emis)*data_SR['SR_dw_ir'] )/(SR_emis*stepbol))**(1/4)

     
    dist, dist_bin = distToNearestCloud(dataOut,x,y)
    
    if dist == 'N/A':     
        if data_L8['SC_CDIST'] > 10:
            dist_bin = '10+ km '
        elif data_L8['SC_CDIST'] > 5 and data_L8['SC_CDIST'] <= 10:
            dist_bin = '5 to 10 km'
        elif data_L8['SC_CDIST'] > 1 and data_L8['SC_CDIST'] <= 5:
            dist_bin = '1 to 5 km'
        elif data_L8['SC_CDIST'] > 0 and data_L8['SC_CDIST'] <= 1:
            dist_bin = '0 to 1 km'
        else:
            dist_bin = '0 km' 

    
    headers = ",".join(data_SR.keys()) + "," + "Surfrad_LST" + "," + ",".join(data_L8.keys()) + \
        ",dist_to_cloud" + ",cloud_bin" + ",cwv" + ",SRlessL8_LST" + ",SRlessSC_LST"
    values = ",".join(str(e) for e in data_SR.values()) + "," + str(SR_LST) + "," +  \
        ",".join(str(f) for f in data_L8.values()) + ',' + str(dist) + ',' + str(dist_bin) +  \
        ','  + str(cwv)+ ',' + str(np.round(SR_LST-data_L8['L8_SW_LST'],3)) + ',' + str(np.round(SR_LST-data_L8['SC_LST'],3))
    
    
    
    # write data to txt file
    filename_out = '/dirs/data/tirs/downloads/Surfrad/results/analysis_2020_2021_collection2_level2.csv'
    
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
        elif dist > 0 and dist <= 1:
            dist_bin = '0 to 1 km'
        else:
            dist_bin = '0 km' 
    except:
         dist = 'N/A' 
         dist_bin = 'N/A'
    
    return dist, dist_bin

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    