#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 08:18:55 2021

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt

def load_RSR(sensor='TIRS10'):
    
    # load RSR of specified sensor
    
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    if sensor == 'TIRS10':
        filename = filepath + 'TIRS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'TIRS11':
        filename = filepath + 'TIRS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,2]
    elif sensor == 'TIRS2_10':
        filename = filepath + 'TIRS2.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'TIRS2_11':
        filename = filepath + 'TIRS2.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,2]
        
    return wave, rsr


wave, rsr = load_RSR(sensor='TIRS2_11')
plt.plot(wave, rsr, color='r')

plt.plot(wave, rsr,'--', color='k')

plt.xlabel('Wavelength [$\mu$m]', fontsize=15)
plt.ylabel('Relative spectral response', fontsize=15)
plt.legend(['TIRS-1','TIRS-2'], fontsize=15)
