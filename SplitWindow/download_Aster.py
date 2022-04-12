# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:16:18 2019

@author: tkpci

Code to download all ASTER tiles that covers a specific lat lon range
Input: coord = [lat_NW, lon_NW, lat_SE, lon_SE]
Example: 
CORNER_UL_LAT_PRODUCT = 39.95276
CORNER_UL_LON_PRODUCT = -124.63895
CORNER_LR_LAT_PRODUCT = 37.83207
CORNER_LR_LON_PRODUCT = -121.95225

coord = [CORNER_UL_LAT_PRODUCT,CORNER_UL_LON_PRODUCT,CORNER_LR_LAT_PRODUCT,CORNER_LR_LON_PRODUCT]
"""

import requests
import math
import numpy as np
import os
import h5py
import osgeo.gdal as gdal
from osgeo import osr
import pdb
import matplotlib.pyplot as plt



def downLoadAster(dataOut, coord):

    folder = dataOut['variables']['autodownloads_Aster']
    files = os.listdir(folder)
    
    # remove all previously downloaded files
    for f in files:
        os.remove(folder + f)
    
    lat_NW = int(math.ceil(coord[0]))
    lon_NW = int(math.ceil(coord[1]))-1
    
    if len(coord) == 2:
        lat_tiles = 1
        lon_tiles = 1
    else:
        lat_tiles = int(math.ceil(coord[0]-math.floor(coord[2])))
        lon_tiles = int(math.ceil(coord[3]-math.floor(coord[1])))
        
    
    # create stitched files - note that last row/column of aster overlap with first
    asterCube = np.zeros([1000*lat_tiles,1000*lon_tiles,5])
    
    
    username = 'Cisthermal'
    password = 'C15Th3rmal'
    
    # 
    #r = requests.get(url, allow_redirects=True)
    
    with requests.Session() as session:
     
        session.auth = (username, password)
        
        
    
        for i in range(lat_tiles):
            for j in range(lon_tiles):
                
                #pdb.set_trace()
                
                lat_sign = np.sign(lat_NW-i)
                if (np.abs(lat_NW)-i) < 10:
                    if lat_NW-i == 0:
                        lat = '00'
                    elif lat_sign == -1:
                        lat = '-0'+str(abs(lat_NW-i))
                    
                    else:
                        lat = '0'+str(abs(lat_NW-i))
                else:
                    lat = str(lat_NW-i)
                
                lon_sign = np.sign(lon_NW+j)
                if (abs(lon_NW+j)) < 10:
                    if lon_sign == -1:
                        lon = '-00'+str(abs(lon_NW+j))
                    else:
                        lon = '00'+str(abs(lon_NW+j))
                elif (abs(lon_NW+j)) < 100:
                    if lon_sign == -1:
                        lon = '-0'+str(abs(lon_NW+j))
                    else:
                        lon = '0'+str(abs(lon_NW+j))
                else:
                    lon = str(lon_NW+j)    
                 
                
                if len(lat) == 1:
                    lat = '0'+ lat
                    
                
                
                url = 'https://e4ftl01.cr.usgs.gov/ASTER/ASTT/AG100.003/2000.01.01/AG100.v003.'+lat+'.'+lon+'.0001.h5'
              
                
                file = url[url.rfind('/')+1:]
                filename = folder + file 
            
            
                r1 = session.request('get', url)
                r = session.get(r1.url, auth=(username, password))
            
                if r.status_code == 200:
                    print('Downloading ASTER tile ' + file)
                    open(filename, 'wb').write(r.content)
                elif r.status_code == 404:
                    print('File Not Found.')
                else:
                    print('File not downloading - check server')
                
                #pdb.set_trace()
                h5 = h5py.File(filename,'r')      #list(f)   #f['Emissivity']['Mean'].shape
                
                # apply gain and bias adjustments to data
                a13 = np.divide(np.array((h5['Emissivity']['Mean'][3,:,:]),dtype='float'),1000)
                a13 = np.where(a13 < 0,0,a13)
                a13 = np.where(a13 > 1,1,a13)
                a14 = np.divide(np.array((h5['Emissivity']['Mean'][4,:,:]),dtype='float'),1000)
                a14 = np.where(a14 < 0,0,a14)
                a14 = np.where(a14 > 1,1,a14)
                ndvi = np.divide(np.array((h5['NDVI']['Mean']),dtype='float'),100)
                ndvi = np.where(ndvi < 0,0,ndvi)
                                                
                a13_std = np.divide(np.array((h5['Emissivity']['SDev'][3,:,:]),dtype='float'),10000)
                a14_std = np.divide(np.array((h5['Emissivity']['SDev'][4,:,:]),dtype='float'),10000)
                
                asterCube[i*1000:i*1000+1000,j*1000:j*1000+1000,0] = a13
                asterCube[i*1000:i*1000+1000,j*1000:j*1000+1000,1] = a14
                asterCube[i*1000:i*1000+1000,j*1000:j*1000+1000,2] = ndvi
                asterCube[i*1000:i*1000+1000,j*1000:j*1000+1000,3] = a13_std
                asterCube[i*1000:i*1000+1000,j*1000:j*1000+1000,4] = a14_std
            
    coordsCube = [lat_NW, lon_NW, lat_NW-lat_tiles, lon_NW+lon_tiles]
    
    return asterCube, coordsCube
 
#asterCube, coordCube = downLoadAster(coord)
       
def georegisterAndSaveAster(dataOut, asterCube, coordsCube):

    filepath = dataOut['variables']['aster_filepath']
    
    filename = filepath + 'e13_e14_NDVI_' + dataOut['pathrow'] + '.tif'

    width = asterCube.shape[0]
    height = asterCube.shape[1]
    bands = asterCube.shape[2]

    upper_left_x = coordsCube[0]
    upper_left_y = coordsCube[1]
    
    lower_right_x = coordsCube[2]
    lower_right_y = coordsCube[3]

    x_resolution = -(upper_left_x-lower_right_x)/width
    y_resolution = -(upper_left_y-lower_right_y)/height


    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(filename, height, width, bands, gdal.GDT_Float32)
    
    ds.SetGeoTransform([upper_left_y, y_resolution, 0, upper_left_x, 0, x_resolution])
    # Define target SRS
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) 
    srs.SetWellKnownGeogCS("WGS84")
    ds.SetProjection(srs.ExportToWkt())

    
    
    ds.GetRasterBand(1).WriteArray(asterCube[:,:,0])
    ds.GetRasterBand(2).WriteArray(asterCube[:,:,1])
    ds.GetRasterBand(3).WriteArray(asterCube[:,:,2])
    ds.GetRasterBand(4).WriteArray(asterCube[:,:,3])
    ds.GetRasterBand(5).WriteArray(asterCube[:,:,4])
    ds.FlushCache()
    ds = None
    
    print('Aster emis and NDVI saved to file')
   

 



