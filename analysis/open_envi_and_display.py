#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 08:35:41 2020

@author: tkpci
"""

from tkinter import *   
from tkinter import filedialog 
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
import re
import pdb
import envi_header



def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.foldername = filedialog.askopenfilename(initialdir = "/dirs/data/tirs/downloads/")
    return root.foldername
    root.destroy()

def open_envi_and_display():
    
    filename = popup_get_folder()
    
    # read in ENVI data file    
    temp = envi.open(filename)
    img = temp.open_memmap(writeable = True)
    
      
    plt.imshow(img[:,:,0]/np.max(img[:,:,0])) #, cmap = "gray")
    plt.axis("off")
    


def get_envi_header_dict(filename):
    
    filename
    
    # f = open(filename, "r")
    # print(f.read()) 
    
    infile = open(filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()   
    
    wavelengths = {}
    word = 'wavelength'
    
    for i in range(0,len(lines)):
        
        if "bands" in lines[i]:
            
            num = [int(s) for s in lines[i].split() if s.isdigit()]
            wavelengths = np.zeros(num[0],)
        
        if word in lines[i]:
            #pdb.set_trace()
            k=0
            test = True
            wave = lines[i+1]
            while test:
                wavelengths[k] = float(wave[2:11])
                k +=1
                wave = lines[i+k+1]
                if k == num[0]:
                    test = False

    return wavelengths
    
#open_envi_and_display()
    
# wavelengths = get_envi_header_dict(filename)    
    
header_dict = envi_header.read_hdr_file(full_filename)    
    
    