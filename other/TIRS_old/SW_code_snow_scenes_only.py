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

# please enter your earth explorer username and password
# example: sceneID = 'LC81161192020215LGN00'
# example: filepath = '/dirs/data/tirs/downloads/test/'

# this function will download the landsat scene to a folder using a scene_ID
# then scene will be untarred and unzipped into own folder
def download_landsat(filepath, scene_ID):
    username = 'xxxxxx'
    password = 'xxxxxx' 
    
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
def TIRS_radiance(counts, band='b10', GB='yes'):

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
    radiance_b10_GB_adjust = TIRS_radiance(b10_DC, band='b10', GB='yes')
    radiance_b11_GB_adjust = TIRS_radiance(b11_DC, band='b11', GB='yes')

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
    
    
def main():
    
    
    # change folder and scene ID as needed
    filepath = '/dirs/data/tirs/downloads/test/'
    scene_ID = 'LC81001212020215LGN00'
    
    download_landsat(filepath, scene_ID)
    
    # find downloaded filename
    for file in os.listdir(filepath):
        if file.endswith(".tar.gz"):
            filename = file
    
    # note hardcoded emissivity values for snow based on UCSB emissivity database spectrally sampled with TIRS RSR's
    ls_emis_final_b10=0.9904
    ls_emis_final_b11=0.9724
    
    
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
    

    
if __name__ == '__main__':
    splitwindowLST = main()
    



  