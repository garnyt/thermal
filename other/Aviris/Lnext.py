#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:24:52 2021

@author: tkpci
"""
import gzip
import os
import shutil
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import spectral.io.envi as envi
from tkinter import *   
from tkinter import filedialog 
import envi_header
from tkinter import messagebox
from multiprocessing import Process, Manager,Pool, cpu_count



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


def untar(filepath, filepath_new):
    """ extract all files from a tar archive (equivalent `tar -xvf filepath directory`)"""
    
    print('Unzipping download (.tar)')
    #Dont create new directory - this code does it by itself
    
    scene_filename = ''
    
    with tarfile.open(filepath_new, 'r') as tf:
        scene_filename = tf.getmembers()[0].name[:(tf.getmembers()[0].name).rfind('.')]
        tf.extractall(filepath_new[:-4])

    return scene_filename


# create RSR
def create_RSR(center, FWHM, wavelengths):
    
    #wave = np.arange(center-2,center+2,0.01)
    wave = np.arange(350,2500,1)
    rsr = norm.pdf(wave,center,FWHM/2.3548)
    rsr_resampled = np.interp(wavelengths,  wave, rsr)
    rsr_resampled = rsr_resampled/max(rsr_resampled)
    #plt.plot(wave,rsr)
     
    return rsr_resampled, wave

def open_envi(filename):

    # read in ENVI data file    
    temp = envi.open(filename)
    img = temp.open_memmap(writeable = True)
    
    header_dict = envi_header.read_hdr_file(filename)
    wavelengths = header_dict['wavelength'].split(',')   
        
    # Convert from string to float (optional)    
    wave = [float(l) for l in wavelengths] 
    wavelengths = np.array(wave) 
        

    return img, wavelengths, header_dict

def writeENVI(save_filename, img, wavelengths, header_dict=0):
    
    if header_dict == 0:
        messagebox.showinfo("Choose .hdr file", "Select a sample .hdr file. Wavelengths, sample, lines and bands will be updated.")
        filename = popup_get_file(save_filename)
        header_dict = envi_header.read_hdr_file(filename)
        
    if save_filename[-3:] != 'hdr':
        save_filename = save_filename + '.hdr'
    
    
    header_dict['lines'] = str(img.shape[0])
    header_dict['samples'] = str(img.shape[1])
    header_dict['bands'] = str(img.shape[2])
    header_dict['wavelength'] = '{'+ ', '.join(str(np.round(x,0)) for x in wavelengths)+'}'
    header_dict['fwhm'] = [] 
    
    envi.save_image(save_filename, img, metadata = header_dict)

def popup_get_file():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askopenfilename(initialdir = "/dirs/data/tirs/downloads/")
    return root.foldername
    root.destroy()


def main():
    
    bands = np.arange(1,21,1)
    center_wave = [410,443,490,560,620,650,665,705,740,842,865,945,985,1035,1090,1375,1610,2100,2210,2260]
    FWHMs = [20,20,65,35,20,20,30,15,15,15,20,20,20,20,30,30,90,30,40,40]
    
    # filepath = '/dirs/data/tirs/aviris/data/'
    # filename = 'f070804t01p00r04rdn_c.tar'
    # #filepath_new = ungzip(filepath+filename)
    # filepath_new = filepath + filename
    # scene_filename = untar(filepath, filepath_new)
    
    filename = popup_get_file()
    
    img, wavelengths, header_dict = open_envi(filename)
    img = np.array(img.astype(float))
    # add gain and convert to W/m2/sr/um
    # gain as per the gain file in the aviris data folder
    img[:,:,0:110] = np.divide(img[:,:,0:110],300)
    img[:,:,110:160] = np.divide(img[:,:,110:160],600)
    img[:,:,160:225] = np.divide(img[:,:,160:225],1200)
    # in readme file in aviris data folder - units after gain is applied is uW/cm2.sr.nm - so convert to W/m2/sr/um
    img = img*10
    
    
    
    img_reshaped = np.reshape(img, [img.shape[0]*img.shape[1],img.shape[2]])
    
    # reshape to 2D data - 1st - spatial, 2nd = spectral
    img_resampled = np.empty([img_reshaped.shape[0],len(center_wave)])

    # create new image for each MSI band
    for i in range(len(center_wave)):
        print('Resampling band ',center_wave[i], ' nm')
        rsr_resampled,wavelength = create_RSR(center_wave[i], FWHMs[i], wavelengths)
        
        
        rsr_all = np.tile(rsr_resampled, [img_reshaped.shape[0],1])
        temp = np.multiply(rsr_all,img_reshaped)
        img_resampled[:,i] = np.trapz(list(temp),x=list(wavelengths),axis=1)/np.trapz(list(rsr_all),x=list(wavelengths), axis=1)
        plt.plot(wavelengths, rsr_resampled)
        
    img_MSI = np.reshape(img_resampled, [img.shape[0],img.shape[1],len(center_wave)]) 
    
    save_filename = filename[:-4] + '_MSI_Lnext'
    writeENVI(save_filename, img_MSI, center_wave, header_dict)
        
    plt.figure()
    plt.imshow(img[:,:,30])
    plt.figure()
    plt.imshow(img_MSI[:,:,6])
    
    
    
    
    
    
    