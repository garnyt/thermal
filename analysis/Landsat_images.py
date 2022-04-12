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
    
def popup_openfile(initDir):
    
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.filename = filedialog.askopenfilename(initialdir = initDir,title = "Select image file to opn (hdr, tif, jpg, png only)")
    return root.filename
    root.destroy()

def read_data(initialdir= "/dirs/data/tirs/SW_analysis_2022/bilinear/edclpdsftp.cr.usgs.gov/downloads/provisional/land_surface_temperature/february_2022/resample_methods/"):
    
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

    
    

def main():
        

    BL, wavelengths  = read_data()
    BL = BL *0.00341802 + 149.0
    
    img_camel = np.squeeze(img_camel)
    
    #SW_Ray = img *0.00341802 + 149.0
    
    initDir = '/dirs/data/tirs/'
    filename = popup_openfile(initDir)
    
    gdal.UseExceptions()
    rd=gdal.Open(filename)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    BL = data.ReadAsArray()#*0.00341802 + 149.0
    
    # filepath ='/dirs/data/tirs/Landsat9/scenes/LC09_L1TP_030035_20211114_20220119_02_T1/'
    # sceneID = 'LC09_L1TP_030035_20211114_20220119_02_T1_B11.TIF'
    
    initDir = '/dirs/data/tirs/SW_analysis_2022/bilinear/edclpdsftp.cr.usgs.gov/downloads/provisional/land_surface_temperature/february_2022/resample_methods/cubic_convolution/'
    filename = popup_openfile(initDir)
    
    gdal.UseExceptions()
    rd=gdal.Open(filename)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    CC = data.ReadAsArray()*0.00341802 + 149.0
    
    plt.subplot(1,3,1)    
    plt.imshow(BL,vmax=300, vmin = 270, cmap = 'jet')
    plt.axis('off')
    plt.colorbar()
    plt.title('Bilinear')
    
    plt.subplot(1,3,2)
    plt.imshow(CC, cmap = 'jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Cubic')
    
    plt.subplot(1,3,3)
    plt.imshow(BL-CC, cmap='jet')#, vmin = 0.95,vmax = 1 ,cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Difference')
    
    
    
    
    plt.imshow(B10[3841:3895,3904:3961]-B11[3841:3895,3904:3961], cmap='jet')#, vmin = 0.95,vmax = 1 ,cmap='jet')
    plt.title('Difference Image - L9 SW cubic vs bilinear')
    plt.colorbar()
    plt.axis('off')
    
    plt.imshow(B10-B11,vmin = 65000, cmap='jet')#, vmin = 0.95,vmax = 1 ,cmap='jet')
    plt.title('Difference Image')
    plt.colorbar()
    plt.axis('off')
    
    plt.imshow(B11)
    
    filepath = '/dirs/data/tirs/downloads/test/LC08_L1TP_026027_20140608_20200911_02_T1/'
    #sceneID = 'LC08_L1TP_195054_20201115_20210315_02_T1'
    #sceneID = 'LC08_L1TP_118062_20150819_20200908_02_T1'
    sceneID = 'LC08_L1TP_026027_20140608_20200911_02_T1'
    filename_1 = sceneID + '_SW_LST_qa.TIF'
    
    #filename_1 = 'e13_e14_ndvi_registered.tif'
    
    gdal.UseExceptions()
    rd=gdal.Open(filepath + filename_1)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    tania = data.ReadAsArray() 
    
    filepath = '/dirs/data/tirs/downloads/test/LC80360262014325LGN01_ray/'
    filename_2 = 'internal_landsat_ndvi.tif'
    
    gdal.UseExceptions()
    rd=gdal.Open(filepath + filename_2)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    img = data.ReadAsArray() 
    
    
    #[6975:7189,3914:4173]
    #[7100:7200,4000:4112]
    
    img1 = img
    img2 = tania
    
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap = 'gray', vmin = 0.9, vmax = 0.99)
    plt.axis('off')
    plt.title('SW_Ray')
    
    
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap = 'gray', vmin=0.9, vmax = 0.99)
    plt.axis('off')
    plt.title('SW_RIT')
    #plt.colorbar()
    
    
    diff = img2.astype(float)-img1.astype(float)
    diff[img == 0] = 0
    diff[diff< -1000] = 0
    diff[diff> 1000] = 0
    diff[np.isnan(diff)] = 0
    #plt.subplot(1,3,3)
    plt.imshow(diff, cmap = 'gray', vmin = -0.01, vmax = 0.01)
    plt.axis('off')
    plt.colorbar()
    plt.title('Difference')
    
    plt.hist(diff.flatten(), bins=50, range=[-0.1, 0.1])
    plt.title(sceneID)
    plt.xlabel('Difference in Kelvin')    

    
    dataOut = {}
    dataOut['wkt_projection'] = wkt_projection
    dataOut['geoTransform'] = geoTransform
    dataOut['folder'] = filepath
    dataOut['sceneID'] = sceneID
    dataOut['SR_B5'] = img
    
    save_Geotif(dataOut, 'SR_B5')
    
    
    #####################
    
    snow_emis, wavelengtsh = read_data()
    veg_emis, wavelengtsh = read_data()
    dry_emis, wavelengtsh = read_data()
    aster, wavelengtsh = read_data()
    
    plt.subplot(1,3,1)
    plt.imshow(snow_emis, cmap = 'gray', vmin = 0.95, vmax = 1)
    plt.axis('off')
    plt.title('B10 snow emis')
    plt.subplot(1,3,2)
    plt.imshow(veg_emis, cmap = 'gray', vmin = 0.95, vmax = 1)
    plt.axis('off')
    plt.title('B10 veg emis')
    plt.subplot(1,3,3)
    plt.imshow(dry_emis, cmap = 'gray', vmin = 0.95, vmax = 1)
    plt.axis('off')
    plt.title('B10 dry emis')
    

    img, wavelengtsh = read_data()
    img_camel, wavelengtsh = read_data()
    img = img/10000
    img_camel = img_camel/10000
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap = 'jet', vmin = 0.96, vmax = 1)
    plt.axis('off')
    plt.colorbar()
    plt.title('Emis band 10 (aster)')
    plt.subplot(1,3,2)
    plt.imshow(img_camel, cmap = 'jet',vmin = 0.96, vmax = 1)
    plt.axis('off')
    plt.colorbar()
    plt.title('Emis band 10 (camel)')
    plt.subplot(1,3,3)
    plt.imshow(img-img_camel, cmap = 'jet',vmin = -0.01, vmax = 0.01)
    plt.axis('off')
    plt.colorbar()
    plt.title('Diff: aster and camel emis')

    plt.subplot(1,3,1)
    plt.imshow(img, cmap = 'jet', vmin = 270, vmax = 280)
    plt.axis('off')
    plt.colorbar()
    plt.title('SW LST [K] (aster)')
    plt.subplot(1,3,2)
    plt.imshow(img_camel, cmap = 'jet',vmin = 270, vmax = 280)
    plt.axis('off')
    plt.colorbar()
    plt.title('SW LST [K] (camel)')
    plt.subplot(1,3,3)
    plt.imshow(img-img_camel, cmap = 'jet',vmin = -0.5, vmax = 1.5)
    plt.axis('off')
    plt.colorbar()
    plt.title('Diff: aster and camel LST [K]')
    
    plt.imshow(img-img_camel)

    

