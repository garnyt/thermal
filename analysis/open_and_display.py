#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 19:05:19 2022

@author: tkpci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:10:27 2021

@author: tkpci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:00:37 2020

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
 

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askopenfilename(initialdir = "/dirs/data/tirs/downloads/test/")
    return root.foldername
    root.destroy()
    
def popup_openfile(initDir = '/dirs/data/tirs/'):
    
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.filename = filedialog.askopenfilename(initialdir = initDir,title = "Select image file to opn (hdr, tif, jpg, png only)")
    return root.filename
    root.destroy()

def read_data(initDir = '/dirs/data/tirs/'):
    
    # read data
    full_filename = popup_openfile(initialdir)
    filetype = full_filename[full_filename.find('.')+1:]
    
    filetypeOptions = ['jpg','JPG','jpeg','JPEG','png','PNG']
    filedng = ['dng','DNG']
  
    if filetype in filetypeOptions:
        img = readTIFF(full_filename)
        wavelengths = []
    elif filetype in ['tif', 'TIF']:    
        gdal.UseExceptions()
        rd=gdal.Open(full_filename)
        #wkt_projection =rd.GetProjection()
        #geoTransform= rd.GetGeoTransform()
        data= rd.GetRasterBand(1)
        img = data.ReadAsArray() 
        wavelengths = []
    elif filetype == 'hdr':
        img, wavelengths = readENVI(full_filename)
    elif filetype in filedng:
        img = readDNG(full_filename)
        wavelengths = []
    else:
        mb.showerror("File type", "Can only open hdr or tif files")
        img = []
        wavelengths = []
            
            
    return img, wavelengths  

def readENVI(full_filename):
    
    # test is file ends with .hdr
    if not full_filename.find('.hdr'):
        full_filename = full_filename + '.hdr'
    
    # read in ENVI data file    
    temp = envi.open(full_filename)
    img = temp.open_memmap(writeable = True)
    
    # read in header info
    # header_dict = envi_header.read_hdr_file(full_filename)
    # # Extract wavelengths, splitting into a list
    # try:
    #     wavelengths = header_dict['wavelength'].split(',')   
        
    #     # Convert from string to float (optional)    
    #     wave = [float(l) for l in wavelengths] 
    #     wavelengths = np.array(wave) 
    # except:
    #     try:
    #         wavelengths = header_dict['original wavelength'].split(',')   
            
    #         # Convert from string to float (optional)    
    #         wave = [float(l) for l in wavelengths] 
    #         wavelengths = np.array(wave) 
    #     except:
    #         wavelengths = []      
    wavelengths = []  
    
    return img, wavelengths  


def main(filepath):
        
    
    filename = popup_openfile(filepath)
    
    gdal.UseExceptions()
    rd=gdal.Open(filename)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    img = data.ReadAsArray()
    
    plt.imshow(img)
    plt.colorbar()
 
    
filepath = '/dirs/home/staff/mxmpci/Aaron/rit_sw/90m/I411818/' 
main(filepath)
    
    
    
    
    

