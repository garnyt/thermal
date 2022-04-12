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
import createSiteInfo
import pdb
from osgeo import gdal


def getSurfradValues(folder_c2l1, siteInfo):
    
    files_in_folder = os.listdir(folder_c2l1)
    
    for ele in files_in_folder:
        if 'MTL.txt' in ele:
            MTL_file = ele
            MTL_filepath = folder_c2l1  + MTL_file
            pathrow = MTL_file[10:16]
            sceneID = MTL_file[0:len(MTL_file)-8]
            break
            
    
    line = siteInfo.pathrow.index(pathrow)
    site_name = siteInfo.sitename[line]
    short_name = siteInfo.shortname[line]
    lat = siteInfo.lat[line]
    lon = siteInfo.lon[line]
    
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
    data_SR['sceneID'] = sceneID
    
    os.remove(ftp_dest)

    return data_SR


def get_landsat_data(folder_c2l1,folder_c2l2,data_SR):
    
    # get points from files    
    lat_value = data_SR['SR_lat']
    lon_value = data_SR['SR_lon']
    
    
    sceneID = data_SR['sceneID']
    
    # locate surfrad site based on lat lon
    SW_file = folder_c2l1 + data_SR['sceneID'] +'_SW_LST.TIF'
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
    
    

def writeDataToFile(data_SR, data_L8):

    
    # calculate SURFRAD LST
    SR_emis = (data_L8['L8_emis10'] + data_L8['L8_emis11'])/2
    stepbol = 5.670367*10**(-8)
    SR_LST = ((data_SR['SR_uw_ir'] -(1-SR_emis)*data_SR['SR_dw_ir'] )/(SR_emis*stepbol))**(1/4)

    # create cloud bins for analysis
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
        ",cloud_bin" + ",SRlessL8_LST" + ",SRlessSC_LST"
    values = ",".join(str(e) for e in data_SR.values()) + "," + str(SR_LST) + "," +  \
        ",".join(str(f) for f in data_L8.values()) + ',' + str(dist_bin) +  \
        ',' + str(np.round(SR_LST-data_L8['L8_SW_LST'],3)) + ',' + str(np.round(SR_LST-data_L8['SC_LST'],3))
    
    
    
    # write data to txt file
    filename_out = '/dirs/data/tirs/L8L9_C2L1_202004_202203/surfrad_analysis_202004_202203_collection2.csv'
    
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
                
    
# def distToNearestCloud(dataOut,x,y):
    
#     try:
    
#         # cloud = 0, nocloud  = 1
#         cloud = dataOut.getRad('cloud')
        
#         ind = np.argwhere(cloud == 0)
        
#         if ind.shape[0] == 0:
#             dist = np.sqrt(((cloud.shape[0]/2)**2)*2) * 30 / 1000
            
            
#         else:
#             distances = np.sqrt((ind[:,0] - y) ** 2 + (ind[:,1] - x) ** 2)
#             nearest_index = np.argmin(distances)
         
#             # distance in km
#             dist = distances[nearest_index] * 30 / 1000   # 30 meter pixels to km
            
#         if cloud.shape[0] > 1000:
#             dist = 'N/A'
        
#         # distance bin (0, 0-0.2, 0.2-1, 1-5, 5-10, >10)
#         if dist > 10:
#             dist_bin = '10+ km '
#         elif dist > 5 and dist <= 10:
#             dist_bin = '5 to 10 km'
#         elif dist > 1 and dist <= 5:
#             dist_bin = '1 to 5 km'
#         elif dist > 0 and dist <= 1:
#             dist_bin = '0 to 1 km'
#         else:
#             dist_bin = '0 km' 
#     except:
#          dist = 'N/A' 
#          dist_bin = 'N/A'
    
#     return dist, dist_bin

def main():
    
    # get Surfrad site info
    siteInfo = createSiteInfo.CreateSiteInfo()
    
    filepath = '/dirs/data/tirs/L8L9_C2L1_202004_202203/data/'
    files = os.listdir(filepath)
    filepath_c2l2 = '/dirs/data/tirs/L8L9_C2L2_202004_202203/data/'
    
    
    for folder_c2l1 in files:
        print(folder_c2l1)
        
        try:
            folder_c2l2 = filepath_c2l2 + folder_c2l1 + '/'
        
            # get surfrad data for specific landsat scene
            data_SR = getSurfradValues(filepath + folder_c2l1 + '/', siteInfo)
            
            # get associated pixel values for surfrad point
            data_L8 = get_landsat_data(filepath + folder_c2l1 + '/',folder_c2l2,data_SR)
            
            # write data to file
            writeDataToFile(data_SR, data_L8)

        except:
            print('scene not working')

    
if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    