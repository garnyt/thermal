#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:18:58 2022

@author: tkpci
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
import pdb
 

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askopenfilename(initialdir = "/dirs/data/tirs/downloads/test/")
    return root.foldername
    root.destroy()
    
def popup_openfile(initDir):
    
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.filename = filedialog.askopenfilename(initialdir = initDir,title = "Select image file to opn (hdr, tif, jpg, png only)")
    return root.filename
    root.destroy()

def readENVI(initDir):
    
    full_filename = popup_openfile(initDir)
    #pdb.set_trace()
    
    # test is file ends with .hdr
    if not full_filename.find('.hdr'):
        full_filename = full_filename + '.hdr'
    
    # read in ENVI data file    
    temp = envi.open(full_filename)
    img = temp.open_memmap(writeable = True)
       
    
    return img  


def main():
        

    BL = readENVI("/dirs/data/tirs/SW_analysis_2022/bilinear/edclpdsftp.cr.usgs.gov/downloads/provisional/land_surface_temperature/february_2022/resample_methods/")
    BL = BL *0.00341802 + 149.0
    BL = BL/10000
    
    CC = readENVI("/dirs/data/tirs/SW_analysis_2022/bilinear/edclpdsftp.cr.usgs.gov/downloads/provisional/land_surface_temperature/february_2022/resample_methods/")
    CC = CC *0.00341802 + 149.0  
    CC = CC/10000
    
    plt.subplot(2,3,1)    
    plt.imshow(BL, vmin = 0.95,vmax = 0.99, cmap = 'jet')
    plt.axis('off')
    plt.colorbar()
    plt.title('emis Bilinear [K]')
    
    plt.subplot(2,3,2)
    plt.imshow(CC, vmin = 0.95,vmax = 0.99,cmap = 'jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('emis Cubic [K]')
    
    plt.subplot(2,3,3)
    plt.imshow(BL-CC, vmin = -0.002,vmax = 0 ,cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Difference [K] LC08_L1TP_172070_20180109_20200902_02_T1')
    
    plt.subplot(2,3,4)    
    plt.imshow(BL[4219:4549,3654:3977], vmin = 0.95, vmax = 0.99,cmap = 'jet')
    plt.axis('off')
    plt.colorbar()
    plt.title('emis Bilinear [K]')
    
    plt.subplot(2,3,5)
    plt.imshow(CC[4219:4549,3654:3977], vmin = 0.95, vmax = 0.99,cmap = 'jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('emis Cubic [K]')
    
    plt.subplot(2,3,6)
    plt.imshow(BL[4219:4549,3654:3977]-CC[4219:4549,3654:3977], vmin = -0.002,vmax = 0 ,cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Difference [K]') 
    

