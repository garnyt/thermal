#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:34:23 2022

@author: tkpci

open landsat and mODIS images and stack
"""

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
from tkinter import *   
from tkinter import filedialog 
from osgeo import gdal_array
import spectral.io.envi as envi
from pyhdf.SD import SD, SDC
    
def popup_openfile(initDir):
    
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.filename = filedialog.askopenfilename(initialdir = initDir)
    return root.filename
    root.destroy()

def popup_get_folder(initDir):
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askopenfilename(initialdir = initDir)
    return root.foldername
    root.destroy()
    
def save_landsat_with_gain_bias_adjustment(folder):
    
    files = os.listdir(folder)
    
    for file in files:
        if 'MTL.txt' in file:
            MTL_name = folder + file
        elif 'B10.tif' in file:
            b10_name = folder + file
        elif 'B11.tif' in file:
            b11_name = folder + file
            
    # this reads the MTL file and get the gain and bias values for B10 and B11
    f = open(MTL_name,"r")
    content = f.read()

    rad10_mult_index = content.index('RADIANCE_MULT_BAND_10')
    rad11_mult_index = content.index('RADIANCE_MULT_BAND_11')
    rad10_add_index = content.index('RADIANCE_ADD_BAND_10')
    rad11_add_index = content.index('RADIANCE_ADD_BAND_11')
    
    s = content[rad10_mult_index:rad10_mult_index+35]
    rad10_mult = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    s = content[rad11_mult_index:rad11_mult_index+35]
    rad11_mult = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    s = content[rad10_add_index:rad10_add_index+35]
    rad10_add = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    s = content[rad11_add_index:rad11_add_index+35]
    rad11_add = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    
    
    
    # read band 10 and get georeferenced data
    gdal.UseExceptions()
    rd=gdal.Open(b10_name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    b10 = data.ReadAsArray()
    # apply gain and bias to digital count to get radiance values
    b10 = b10*rad10_mult + rad10_add
    
    # read band 11
    gdal.UseExceptions()
    rd=gdal.Open(b11_name)
    data= rd.GetRasterBand(1)
    b11 = data.ReadAsArray()
    # apply gain and bias to digital count to get radiance values
    b11 = b10*rad11_mult + rad11_add
    
    return b10, b11, wkt_projection, geoTransform


def save_Geotif(data, filename, wkt_projection, geoTransform):
    
    width = data.shape[0]
    height = data.shape[1]
    bands = 1  
    
    file = dataOut['folder'] + '/' + dataOut['sceneID'] + '_' + filename + '.TIF'
    
    driv = gdal.GetDriverByName("GTiff")
    ds = driv.Create(file, height, width, bands, gdal.GDT_Float32)
    ds.SetGeoTransform(geoTransform)
    ds.SetProjection(wkt_projection)
    ds.GetRasterBand(1).WriteArray(data)
    
    ds.FlushCache()
    ds = None
    
    print('File ' + filename + ' saved')

def open_MODIS(filename):
    
    file = SD(file_name, SDC.READ)
    
    datasets_dic = file.datasets()
    
    for idx,sds in enumerate(datasets_dic.keys()):
        print(idx,sds)
        
    sds_object = file.select('Latitude')
    lat = sds_object.get()
    
    sds_object = file.select('Longitude')
    lon = sds_object.get()
    
    sds_object = file.select('EV_1KM_Emissive')
    #print(sds_object.info())
    
    emissive = sds_object.get()
    B31 = (emissive[10,:,:]-sds_object.attributes()['radiance_offsets'][10])*sds_object.attributes()['radiance_scales'][10]
    B32 = (emissive[11,:,:]-sds_object.attributes()['radiance_offsets'][11])*sds_object.attributes()['radiance_scales'][11]
    
    
    
    plt.imshow(lon)
    
def main():

    #select landsat folder (where MTL file and band 10 and 11 lies)
    initDir = '/dirs/data/tirs/' # set initial directory to make opening files faster
    
    # select Landsat Folder
    folder = popup_get_folder(initDir)  
    
    b10, b11, wkt_projection, geoTransform = open_landsat(folder)
    
    filename + '.TIF'
    
    # select MODIS hdf file
    file_name = popup_openfile(initDir)
    
    
    
    
    
    
    
    
    
    
    