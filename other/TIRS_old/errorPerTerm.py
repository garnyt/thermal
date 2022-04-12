#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:59:06 2020

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
from tkinter import *   
from tkinter import filedialog 
import os


root = Tk()
root.withdraw()
root.folder = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/test/")
folder = root.folder
root.destroy()


files_in_folder = os.listdir(folder+'/')
            
for ele in files_in_folder:
    if 'termT.tif' in ele:
        termT = ele
    elif 'termE.tif' in ele:
        termE = ele
    elif 'termTcov' in ele:
        termTcov = ele
    elif 'termEcov' in ele:
        termEcov = ele
    elif 'SW_error.tif' in ele:
        SW = ele
    elif 'SW_error_cov' in ele:
        SW_cov = ele
                    
filenameT = folder + '/' + termT
filenameE = folder + '/' + termE
filenameTcov = folder + '/' + termTcov
filenameEcov = folder + '/' + termEcov
filenameSW = folder + '/' + SW
filenameSW_cov = folder + '/' + SW_cov

gdal.UseExceptions()
rd=gdal.Open(filenameT)
truth= rd.GetRasterBand(1)
data = truth.ReadAsArray()


gdal.UseExceptions()
rd=gdal.Open(filenameTcov)
truth= rd.GetRasterBand(1)
data = np.dstack([data, truth.ReadAsArray()])

data = np.dstack([data, data[:,:,0]+data[:,:,1]])

gdal.UseExceptions()
rd=gdal.Open(filenameE)
truth= rd.GetRasterBand(1)
data = np.dstack([data, truth.ReadAsArray()])

gdal.UseExceptions()
rd=gdal.Open(filenameEcov)
truth= rd.GetRasterBand(1)
data = np.dstack([data, truth.ReadAsArray()])

data = np.dstack([data, data[:,:,3]+data[:,:,4]])

gdal.UseExceptions()
rd=gdal.Open(filenameSW)
truth= rd.GetRasterBand(1)
data = np.dstack([data, truth.ReadAsArray()])

gdal.UseExceptions()
rd=gdal.Open(filenameSW_cov)
truth= rd.GetRasterBand(1)
data = np.dstack([data, truth.ReadAsArray()])


names = ['T_uncertainty^2','App Temp cov','Total App Temp Uncertainty','Emis_uncertainty^2','Emis cov','Total Emis uncertainty']
plt.close()
plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(data[:,:,i],vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(names[i])
    plt.axis('off')

 # calculate individual uncertainty per term as % of total uncertainty
termT = np.divide(LST_error_cov**2,((T10diff*T10_error)**2 + (T11diff*T11_error)**2)) 
termE = np.divide(LST_error_cov**2,((E10diff*e10_error_cov)**2 + (E11diff*e11_error_cov)**2))
termTcov = np.divide(LST_error_cov**2,(2*corr_appTemp*T10diff*T11diff*T10_error*T11_error))
termEcov = np.divide(LST_error_cov**2,(2*corr_emis*E10diff*E11diff*e10_error_cov*e11_error_cov))



folder =  "/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/e13_e14_NDVI_017030.tif"


gdal.UseExceptions()
rd=gdal.Open(folder)
truth= rd.GetRasterBand(1)
data = truth.ReadAsArray()


plt.close()

plt.imshow(data, vmin=0.01, vmax=0.06)
plt.colorbar()
plt.axis('off')