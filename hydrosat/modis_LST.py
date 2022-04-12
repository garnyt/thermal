#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:49:09 2022

@author: tkpci
"""

# calculate MODIS ST and compare to Surfrad

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/cis/staff2/tkpci/Code/Python/TIRS')
import createSiteInfo
from osgeo import gdal
import subprocess
import re 
import urllib 
from contextlib import closing
import shutil
import csv
import math

# download scenes

filepath = '/dirs/data/tirs/hydrosat/data/'

txt_files = os.listdir(filepath)

for txt_file in txt_files:
    
    file = filepath + txt_file
    
    file1 = open(file, 'r')
    Lines = file1.readlines()
    
    for line in Lines:
        
        download_call = 'wget --user tk1783_rit.edu --password 9656Yolo! -P '+filepath+'MODIS/ '+ line[:-1]

        os.system(download_call)
        

filepath = '/dirs/data/tirs/hydrosat/data/MODIS/'
#file = 'MOD21A1D.A2020094.h08v05.061.2020331203124.hdf'

files = os.listdir(filepath)

files_downloaded = os.listdir('/dirs/data/tirs/hydrosat/data/MODIS/LST/')

for file in files:
    
    filename = file[:-4]
    if file[-1] != '1' and os.path.isfile(filepath+file) and filename not in files_downloaded:
        
        try:
            
            
  
            in_file = f"/dirs/data/tirs/hydrosat/data/MODIS/"+file # raw MODIS HDF in sinusoid projection
            out_file = f"/dirs/data/tirs/hydrosat/data/MODIS/LST/"+file[:-3]+"tif"
            out_file_time = f"/dirs/data/tirs/hydrosat/data/MODIS/LST/"+file[:-4]+"time.tif"
            out_file_e29 = f"/dirs/data/tirs/hydrosat/data/MODIS/LST/"+file[:-4]+"e29.tif"
            out_file_e31 = f"/dirs/data/tirs/hydrosat/data/MODIS/LST/"+file[:-4]+"e31.tif"
            out_file_e32 = f"/dirs/data/tirs/hydrosat/data/MODIS/LST/"+file[:-4]+"e32.tif"
                
            # open dataset
            dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
            subdataset =  gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
            subdataset2 =  gdal.Open(dataset.GetSubDatasets()[3][0], gdal.GA_ReadOnly)
            subdataset3 =  gdal.Open(dataset.GetSubDatasets()[4][0], gdal.GA_ReadOnly)
            subdataset4 =  gdal.Open(dataset.GetSubDatasets()[5][0], gdal.GA_ReadOnly)
            subdataset5 =  gdal.Open(dataset.GetSubDatasets()[6][0], gdal.GA_ReadOnly)
            
            
            # gdalwarp
            kwargs = {'format': 'GTiff', 'dstSRS': 'EPSG:4326'}
            ds = gdal.Warp(destNameOrDestDS=out_file,srcDSOrSrcDSTab=subdataset, **kwargs)
            ds = gdal.Warp(destNameOrDestDS=out_file_time,srcDSOrSrcDSTab=subdataset2, **kwargs)
            ds = gdal.Warp(destNameOrDestDS=out_file_e29,srcDSOrSrcDSTab=subdataset3, **kwargs)
            ds = gdal.Warp(destNameOrDestDS=out_file_e31,srcDSOrSrcDSTab=subdataset4, **kwargs)
            ds = gdal.Warp(destNameOrDestDS=out_file_e32,srcDSOrSrcDSTab=subdataset5, **kwargs)
            del ds
            
        except:

            print('Error: ', file)
            


filepath = '/dirs/data/tirs/hydrosat/data/MODIS/LST/'
#scene = 'MOD21A1D.A2020232.h11v04.061.2020342152125time.tif'

files = os.listdir(filepath)


def getSurfradValues():
    
    # get Surfrad info
    siteInfo = createSiteInfo.CreateSiteInfo()
    site_name = ['Fort_Peck_MT', 'Desert_Rock_NV']
    short_name = ['fpk', 'dra']
    lat = [48.30783,36.62373]
    lon = [-105.1017,-116.01947]
    GMT_loc = [-7,-8]
    cnt = 1
    
    for scene in files:
        
        if scene[-8:-4] == 'time':
        
            
        
            gdal.UseExceptions()
            rd=gdal.Open(filepath+scene)
            # wkt_projection =rd.GetProjection()
            # geoTransform= rd.GetGeoTransform()
            data= rd.GetRasterBand(1)
            img_time = data.ReadAsArray() 
            
            #gdal.UseExceptions()
            rd=gdal.Open(filepath+scene[:-8]+'.tif')
            data= rd.GetRasterBand(1)
            img = data.ReadAsArray() 
            
            rd=gdal.Open(filepath+scene[:-8]+'e29.tif')
            data= rd.GetRasterBand(1)
            e29 = data.ReadAsArray() 
            
            rd=gdal.Open(filepath+scene[:-8]+'e31.tif')
            data= rd.GetRasterBand(1)
            e31 = data.ReadAsArray() 
            
            rd=gdal.Open(filepath+scene[:-8]+'e32.tif')
            data= rd.GetRasterBand(1)
            e32 = data.ReadAsArray() 
            
            year = scene[10:14]
            num_of_days_str = scene[14:17]
                
            if scene[18:24] == 'h10v04' or scene[18:24] == 'h11v04':
                cnt = 0
            elif scene[18:24] == 'h08v05':
                cnt = 1
            
            lat_value = lat[cnt]
            lon_value = lon[cnt]

            MODIS_file = filepath+scene[:-8]+'.tif'
            pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + MODIS_file + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
            
              
            temp = pixelsLocate.decode()

            numbers = re.findall('\d+',temp) 

            x = int(numbers[0]) # column    
            y = int(numbers[1]) # row
            
            #data with scaling factor and offset according to MOD21 user manual
            MOD_LST = img[y,x]*0.02
            time_L8 = img_time[y,x]*0.1
            
            # https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
            # convert local solar time to GMT
            L_st = GMT_loc[cnt]*15
            
            B = (int(num_of_days_str)-81)*(360/365)
            E = 9.87 * np.sin(np.deg2rad(2*B)) - 7.53 * np.cos(np.deg2rad(B)) - 1.5 * np.sin(np.deg2rad(B)) 
            #E = 9.87 * np.sin(np.deg2rad(2*B)) - 7.53 * np.cos(np.deg2rad(B)) - 1.5 * np.sin(np.deg2rad(B))
            
            TC = 4*(lon[cnt]-L_st) + E
            
            time_MOD_GMT = time_L8 - TC/60- GMT_loc[cnt]
            
            emis29 = e29[y,x] * 0.002 + 0.49
            emis31 = e31[y,x] * 0.002 + 0.49
            emis32 = e32[y,x] * 0.002 + 0.49 
            
            if time_L8 < 25:

                #create download name
                ftp_path = 'ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/' + site_name[cnt] + '/' + str(year) + '/'  
                ftp_name =  short_name[cnt] + year[2:4] + num_of_days_str + '.dat'
                ftp_fullname = ftp_path + ftp_name
                
                ftp_dest = '/dirs/data/tirs/downloads/test/' + ftp_name
                    
                
            
                with closing(urllib.request.urlopen(ftp_fullname))as r:
                    with open(ftp_dest, 'wb') as f:
                        shutil.copyfileobj(r, f)       
                
                data_surfrad = np.loadtxt(ftp_dest, skiprows=2)
        
                # find closest time to surfrad data  
                time_SR = data_surfrad[:,6]
                index_closest = np.abs(time_SR-time_MOD_GMT).argmin()
                
                #data_surfrad[index_closest,0:7]
                
                data_SR = {}
                data_SR['Scene'] = scene[:-8]
                data_SR['SR_sitename'] = site_name[cnt]
                data_SR['SR_time'] = data_surfrad[index_closest,6]
                data_SR['SR_lat'] = lat[cnt]
                data_SR['SR_lon'] = lon[cnt]
                #data_SR['SR_solar_zen'] = data[index_closest,7]
                data_SR['SR_dw_ir'] = data_surfrad[index_closest,16]
                data_SR['SR_uw_ir'] = data_surfrad[index_closest,22]
                #data_SR['SR_airtemp'] = data[index_closest,38]
                #data_SR['SR_rh'] = data[index_closest,40]
                #data_SR['SR_windspd'] = data[index_closest,42]
                #data_SR['SR_winddir'] = data[index_closest,44]
                #data_SR['SR_pressure'] = data[index_closest,46]
                data_SR['MODIS_time_solar'] = time_L8
                data_SR['MODIS_time_GMT'] = time_MOD_GMT
                data_SR['L8_year'] = year
                data_SR['L8_doy'] = num_of_days_str
                
                
                os.remove(ftp_dest)
                
                # calculate SURFRAD LST - Wang 2005 - e = 0.2122e29 + 0.3859 e31 + 0.4029e32
                SR_emis = 0.2122*emis29 + 0.3859*emis31 + 0.4029*emis32
                stepbol = 5.670367*10**(-8)
                SR_LST = ((data_SR['SR_uw_ir'] -(1-SR_emis)*data_SR['SR_dw_ir'] )/(SR_emis*stepbol))**(1/4)
    
    
    
                headers = ",".join(data_SR.keys()) + "," + "Surfrad_LST" + ","+"MODIS_LST"
                values = ",".join(str(e) for e in data_SR.values()) + "," + str(SR_LST) + ","+str(MOD_LST)
                
                
                
                # write data to txt file
                filename_out = '/dirs/data/tirs/hydrosat/analysis/SR_MODIS_analysis.csv'
                
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


    
    
    
                



from pyhdf.SD import SD, SDC

file_name = '/dirs/data/tirs/hydrosat/data/MODIS/MOD21A1D.A2020094.h08v05.061.2020331203124.hdf'
file = SD(file_name, SDC.READ)

print(file.info())

datasets_dic = file.datasets()

for idx,sds in enumerate(datasets_dic.keys()):
    print (idx,sds)
    
sds_obj = file.select('View_Time')
    
sds_obj = file.select('LST_1KM') # select sds

data = sds_obj.get() # get sds data

attr = sds_obj.attributes()

data = np.asarray(data) * attr['scale_factor'] + attr['add_offset']

plt.imshow(data)
plt.colorbar()
plt.axis("off")

