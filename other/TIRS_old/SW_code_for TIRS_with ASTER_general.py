#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Wed Sept 23, 2020

@author: Tania Kleynhans, Rochester Institute of Technology

Fundction input:
    b10_DC = TIRS band10 collection two digital count (if collection 1 data - see note above TIRS_radiance function)
    b11_DC = TIRS band11 collection two digital count (if collection 1 data - see note above TIRS_radiance function)
    ls_emis_final_b10 = emissivity for TIRS band 10
    ls_emis_final_b11 = emissivity for TIRS band 11
    a13_std = aster standard emissivity deviation band 13
    a14_std = aster standard emissivity deviation band 14

Function output:
    splitwindowLST = split window land surface temperature with gain and bias adjustment and averaging filter
    splitwindowQA = split window land surface temperature uncertainty product
"""

import numpy as np
import cv2
from landsatxplore.earthexplorer import EarthExplorer
import gdal
import os
import shutil
import gzip
import tarfile
import matplotlib.pyplot as plt
import requests
import math
import h5py
from osgeo import osr
from osgeo import gdal, gdalconst
import pdb


# please enter your earth explorer username and password
# example: sceneID = 'LC81161192020215LGN00'
# example: filepath = '/dirs/data/tirs/downloads/test/'

# this function will download the landsat scene to a folder using a scene_ID
# then scene will be untarred and unzipped into own folder
def download_landsat(filepath, scene_ID):
    
    username = 'cisthermal'
    password = 'C15Th3rmal'
    
    ee = EarthExplorer(username, password)
    
    ee.download(scene_id=scene_ID, output_dir=filepath)
    
    ee.logout()
    
    for file in os.listdir(filepath):
        if file.endswith(".gz"):
            targzfile = os.path.join(filepath, file)
    
    tarfile = ungzip(targzfile)
    os.remove(targzfile)
    scene_filename = untar(tarfile, filepath, scene_ID)
    os.remove(tarfile)
   
    

# appTemp function converts TOA radiance of band 10 and band 11 to apparent temperature using coefficients as specified in MTL file
def appTemp(radiance, band='b10'):

    # K1 and K2 constants to be found in MTL.txt file for each band
    if band == 'b10':
        K2 = 1321.0789
        K1 = 774.8853
    elif band == 'b11':
        K2 = 1201.1442
        K1 = 480.8883
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')
        return

    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)))

    return temperature

# TIRS_radiance function converts TIRS band 10 and band 11 digital count to radiance with/without the
# additional gain bias adjustment as per calval decisions - this should be included in the collection 2 data (along with some geometric updates)
# if input is collection 2 data, then GB='no', else "yes"
def TIRS_radiance(counts, band='b10', GB='no'):

    radiance = counts.astype(float) * 3.3420 * 10**(-4) + 0.1

    if (band == 'b10' and GB =='yes'):
        radiance = radiance * 1.0151 - 0.14774
    elif (band == 'b11' and GB =='yes'):
        radiance = radiance * 1.06644-0.46326

    return radiance


def SW_coeff():
    
    # split window coefficients as per journal article: Towards an Operational, Split Window-Derived Surface Temperature Product
    #for the Thermal Infrared Sensors Onboard Landsat 8 and 9, A.Gerace, T.Kleynhans, R.Eon, M. Montanaro
    
    coeff = []
    coeff.append(2.2925)  #b0
    coeff.append(0.9929)  #b1
    coeff.append(0.1545)  #b2
    coeff.append(-0.3122)  #b3
    coeff.append(3.7186)  #b4
    coeff.append(0.3502)  #b5
    coeff.append(-3.5889)  #b6
    coeff.append(0.1825)  #b7
    # regression error
    coeff.append(0.73) 
    
    return coeff

# calculating SW 
def calcSW(b10_DC, b11_DC, ls_emis_final_b10, ls_emis_final_b11):

    # convert DC to radiance with("yes")/witout("no") addtional gain bias adjustment
    # if collection 2 data is used, then use "no". If collection 1 data is used, then "yes"
    radiance_b10_GB_adjust = TIRS_radiance(b10_DC, band='b10', GB='no')
    radiance_b11_GB_adjust = TIRS_radiance(b11_DC, band='b11', GB='no')

    # convert TOA radiance to apparent temperature
    appTemp_b10 = appTemp(radiance_b10_GB_adjust, band='b10')
    appTemp_b11 = appTemp(radiance_b11_GB_adjust, band='b11')

    # calculate the 5x5 averaged app temp for SW algorithm difference terms
    kernel = np.ones((5,5),np.float32)/25
    appTemp_b10_ave = cv2.filter2D(appTemp_b10,-1,kernel)
    appTemp_b11_ave = cv2.filter2D(appTemp_b11,-1,kernel)

    # calculate terms for split window algorithm
    T_diff = (appTemp_b10_ave - appTemp_b11_ave)/2
    T_plus =  (appTemp_b10 + appTemp_b11)/2
    e_mean = (ls_emis_final_b10 + ls_emis_final_b11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (ls_emis_final_b10-ls_emis_final_b11)/(e_mean**2)
    quad = (appTemp_b10_ave-appTemp_b11_ave)**2

    # split window coefficients
    coeff = SW_coeff()
    

    # calculate split window LST
    splitwindowLST = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad

    return splitwindowLST, appTemp_b10, appTemp_b11


def ungzip(filepath):
    """ un-gzip a file (equivalent `gzip -d filepath`) """
    
    new_filepath = filepath.replace('.gz', '')
    print('Unzipping download (.gzip)')

    with open(new_filepath, 'wb') as f_out, gzip.open(filepath, 'rb') as f_in:
        try:
           shutil.copyfileobj(f_in, f_out)
        except:
            print('File does not unzip: '+filepath)


    return new_filepath


def untar(filepath, directory, scene_id):
    """ extract all files from a tar archive (equivalent `tar -xvf filepath directory`)"""
    
    print('Unzipping download (.tar)')
    os.mkdir(directory + scene_id)
    
    scene_filename = ''
    
    with tarfile.open(filepath, 'r') as tf:
        scene_filename = tf.getmembers()[0].name[:(tf.getmembers()[0].name).rfind('.')]
        tf.extractall(directory+scene_id)

    return scene_filename


def saveGeotif(data, foldername, filename, wkt_projection, geoTransform):
    
    width = data.shape[0]
    height = data.shape[1]
    bands = 1  
    
    file = foldername + '/' + filename
    
    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(file, height, width, bands, gdal.GDT_Float32)
    ds.SetGeoTransform(geoTransform)
    ds.SetProjection(wkt_projection)
    ds.GetRasterBand(1).WriteArray(data)
    
    ds.FlushCache()
    ds = None
    
    print('File ' + filename + ' saved')
    
    
def downloadASTERemis(MTL_file, sceneID, filepathASTER, filepathLandsat, B10_file):
     
    
    # find path row of scene
    if len(sceneID) == 21:
        pathrow = sceneID[3:9]
    else:  
        pathrow = sceneID[10:16]
        
    # check if files have been donwloaded before
    aster_already_downloaded = filepathASTER + '/e13_e14_NDVI_' + pathrow + '.tif'
            
    if not os.path.isfile(aster_already_downloaded):

    
        # find corner coordinates from MTL to use for ASTER tiles
        f = open(MTL_file,"r")
        content = f.read()
        
        # read corner coordinates from MTL file
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
        
        coord = [max(CORNER_UL_LAT_PRODUCT,CORNER_UR_LAT_PRODUCT), \
                 min(CORNER_UL_LON_PRODUCT,CORNER_LL_LON_PRODUCT),
                 min(CORNER_LR_LAT_PRODUCT,CORNER_LL_LAT_PRODUCT),
                 max(CORNER_LR_LON_PRODUCT,CORNER_UR_LON_PRODUCT)]
    
        #download and save stacked georegistered aster files
        asterCube, coordsCube = downLoadAster(coord, filepathASTER)
        georegisterAndSaveAster(asterCube, coordsCube, pathrow, filepathASTER)
        
    else:
        print('Aster already downloaded for this path row')    
    

    # stack and resample aster to tirs 
    stackLandsatAster(pathrow, filepathASTER, filepathLandsat, B10_file)


def downLoadAster(coord, folder):

    
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

    # earth data download details
    username = 'Cisthermal'
    password = 'C15Th3rmal'
    
    
    try:
    
        with requests.Session() as session:
         
            session.auth = (username, password)
        
            for i in range(lat_tiles):
                for j in range(lon_tiles):
                    
                    lat_sign = np.sign(lat_NW-i)
                    if (np.abs(lat_NW)+i) < 10:
                        if lat_sign == -1:
                            lat = '-0'+str(np.abs(lat_NW)-i)
                        else:
                            lat = '-0'+str(np.abs(lat_NW)-i)
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
                
                    url = 'https://e4ftl01.cr.usgs.gov/ASTER_B/ASTT/AG100.003/2000.01.01/AG100.v003.'+lat+'.'+lon+'.0001.h5'
                  
                    
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
                    
                    
                    h5 = h5py.File(filename,'r')      #list(f)   #f['Emissivity']['Mean'].shape
                    
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
        
    except:
        print('ASTER not downloaded. Make sure this path row has associated ASTER tile (i.e. not over water or antarctic')
        
    return asterCube, coordsCube

       
def georegisterAndSaveAster(asterCube, coordsCube, pathrow, filepath):

    
    filename = filepath + 'e13_e14_NDVI_' + pathrow + '.tif'

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
    
    for file in os.listdir(filepath):
        if file.endswith('.h5'):
            os.remove(filepath + file)
    
    print('Aster emis and NDVI saved to file')  
    

def stackImg1Img2(img1_filename, img2_filename, path):
    #img1 = geotif image which resolution you want to keep
    #img2 = geotif image to be resampled and matched to img1 - this will be the output image

    # Source
    src_filename = img2_filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    #src_geotrans = src.GetGeoTransform()
    
    # We want a section of source that matches this:
    match_filename = img1_filename
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    try:
        dst_filename = path + 'e13_e14_ndvi_registered.tif'
              
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 5, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection( match_proj)
        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
        
        del dst # Flush
        print('  Aster stacked and saved')
        
    except:
        print('  Oops, something went wrong wile stacking ASTER to Landsat - call the Queen!')


def stackLandsatAster(pathrow, filepathASTER, filepathLandsat, B10_file):

    aster = filepathASTER + '/e13_e14_NDVI_' + pathrow + '.tif' 
    landsat = B10_file
    
    path = filepathLandsat
    
        
    print('  Resampling and stacking ASTER onto Landsat')
    
    stackImg1Img2(landsat, aster, path)    

    
def main():
    
    
    # select folder for download
    filepath = '/dirs/data/tirs/CRISP/'
    
    # create folder for ASTER downloaded data 
    filepathASTER = '/dirs/data/tirs/downloads/DRS/ASTER/'
    
    # add scene ID to download - other format does not work
    scene_ID = 'LC80390372020299LGN00'
    scene_ID = 'LC08_L1TP_043033_20210517_20210517_02_RT'
    
    # download, unzip, and save raw landsat file
    download_landsat(filepath, scene_ID)
    
    
    # find downloaded filepath
    for file in os.listdir(filepath):
        if 'LC8' or 'LC08' in file:
            filepathLandsat = filepath + file + '/'
            
    # find scene_name, MTL file, and B10/B11 files
    for file in os.listdir(filepathLandsat):
        if 'MTL' in file:
            MTL_file = filepathLandsat + file 
        if 'B10.TIF' in file:
            B10_file = filepathLandsat + file 
        if 'B11.TIF' in file:
            B11_file = filepathLandsat + file 
    
    
        
    
    # check if ASTER emis files have been downloaded based on path row, and if not, download files
    downloadASTERemis(MTL_file, scene_ID, filepathASTER, filepathLandsat, B10_file)    

    
    
    # note hardcoded emissivity values for snow based on UCSB emissivity database spectrally sampled with TIRS RSR's
    # ls_emis_final_b10=0.9904
    # ls_emis_final_b11=0.9724
    
    
    
    #downloadASTERemis(MTL_file, sceneID, pathrow, filepath)
    
    
    # open landsat8 TIRS data in digital count
    # find downloaded filename
    for file in os.listdir(filepath + scene_ID):
        if file.endswith("B10.TIF"):
            b10Name = filepath + scene_ID + '/' + file
        if file.endswith("B11.TIF"):
            b11Name = filepath + scene_ID + '/' + file
    
       
    gdal.UseExceptions()
    rd=gdal.Open(b10Name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    b10= rd.GetRasterBand(1)
    b10_DC = b10.ReadAsArray()
    
    where_are_NaNs = np.isnan(b10_DC)
    b10_DC[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b11Name)
    b11= rd.GetRasterBand(1)
    b11_DC = b11.ReadAsArray()
    
    where_are_NaNs = np.isnan(b11_DC)
    b11_DC[where_are_NaNs] = 0
    
    
    
    # calculate SW
    splitwindowLST, appTemp_b10, appTemp_b11 = calcSW(b10_DC, b11_DC, ls_emis_final_b10, ls_emis_final_b11)
    
    # save temperature data to georegistered tif file
    saveGeotif(splitwindowLST, filepath + scene_ID, file[:-8] +'_SW_LST.TIF', wkt_projection, geoTransform)
                   
    
    return splitwindowLST    
    

    
#if __name__ == '__main__':
#    splitwindowLST = main()
    



  