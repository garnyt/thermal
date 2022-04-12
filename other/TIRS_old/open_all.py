"""
Created on Tue Nov 19 11:03:16 2019

@author: Tania Kleynhans

Read in any data with or without prompt/filename

"""

from tkinter import filedialog
from tkinter import Tk
import csv
import numpy as np

import spectral.io.envi as envi

def open_all(filename = 0, filetype = 'img'):
    
    if filename == 0:
        root = Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1) 
        root.filename = filedialog.askopenfilename()
        filename = root.filename
    
    if filetype == 'img':
        temp = envi.open(filename + '.hdr')
        data = temp.open_memmap(writeable = True)
        
    if filetype == 'csv':
        temp = open(filename)
        csv_reader = csv.reader(temp)
        data = list(csv_reader)
        data = np.asarray(data)
        #data = data.astype(int)-1
        #bands = flatten(np.transpose(data))  
        #trainX_out = trainX[:,bands]
    
    
    
    return data

