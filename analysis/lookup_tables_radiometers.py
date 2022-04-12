#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 07:08:37 2020

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

def read_RSR():
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    filename = filepath + 'thermopiles.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    rsrs = data[:,1:]
    #plt.plot(wavelength,rsrs)
    
    return wave, rsrs

        
def inverse_planck(radiance,wave_center):
    
    wvl = wave_center * 1e-6
    L = radiance * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    T = 2 * h * c * c / (L * (wvl**5))
    T = np.log(T+1)
    T = (h * c / (k * wvl)) / T
    
    return T

def planck(temperature, wave_center):
    
    wvl = wave_center * 1e-6
    T = temperature 
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    L = (2 * h * c**2 / wvl**5) * 1 / (np.exp((h*c)/(wvl*k*T))-1)*1e-6
       
    return L

def lookup_radiance_table():
    
    wave_range = np.arange(8,14,0.01)
    #radiance_range = np.arange(1,15,0.1)
    temperature_range = np.arange(190,330,0.05)
    
    # wave_table = np.tile(wave_range,(radiance_range.shape[0],1))
    # radiance = np.tile(radiance_range,(wave_range.shape[0],1))
    # radiance = radiance.T
    # T_table = inverse_planck(radiance,wave_center)
    
    wave_table = np.tile(wave_range,(temperature_range.shape[0],1))
    temperature = np.tile(temperature_range,(wave_range.shape[0],1))
    temperature = temperature.T
        
    L_table = planck(temperature,wave_table)
    
    return L_table, temperature_range, wave_range, wave_table

def lookup_temperature_table():
    
    wave_range = np.arange(8,14,0.01)
    radiance_range = np.arange(0.01,15,0.05)
    #temperature_range = np.arange(210,330,0.05)
    
    wave_table = np.tile(wave_range,(radiance_range.shape[0],1))
    radiance = np.tile(radiance_range,(wave_range.shape[0],1))
    radiance = radiance.T
    T_table = inverse_planck(radiance,wave_table)
    
    
    return T_table, radiance_range, wave_range, wave_table


    
def calc_appTemp_from_table():
    
    L_table, temperature_range, wave_range, wave_table = lookup_radiance_table()
    
    wave, rsrs = read_RSR()
    
    for i in range(8):
        rsr = rsrs[:,i]
    
        rsr_interp = np.interp(np.squeeze(wave_table[0,:]), wave, rsr)
        rsr_interp = np.tile(rsr_interp, (L_table.shape[0],1))
        
        radiance = np.trapz(list(rsr_interp * L_table),x=wave_table,axis=1)/np.trapz(list(rsr_interp),x=wave_table,axis=1)
        
        LUT = np.empty([L_table.shape[0],2])
        LUT[:,0] = radiance
        LUT[:,1] = temperature_range
        
        np.savetxt("LUT_radiometer0"+str(i+1)+".csv", LUT, delimiter=",")
    #np.savetxt("LUT_TIRS11.csv", LUT, delimiter=",")
    #np.savetxt("LUT_spectral.csv", L_table, delimiter=",")
    #np.savetxt("LUT_temp_spectral.csv", T_table, delimiter=",")
    #np.savetxt("LUT_temp_spectral_wave.csv", wave_range, delimiter=",")
    #np.savetxt("LUT_temp_spectral_rad.csv", radiance_range, delimiter=",")
 
    radval = np.arange(2,14,0.05)
    
    for i in radiance:
          
        T = inverse_planck(i,12)
        idx = np.abs(radiance - i).argmin()
        T_rad = temperature_range[idx]
        T_tirs = appTemp(i, band='b11')
        
        plt.scatter(i,T_rad-T, c='r', s=3, marker='x')
        plt.scatter(i,T_rad-T_tirs, c='b', s=3, marker='x')
        
    
    plt.legend(('Inverse planck, lookup table diff','TIRS coeff, lookup table diff'))
    plt.xlabel('Radiance [W/$m^2$.sr.$\mu$m]')
    plt.ylabel('Temperature difference [K]')
    plt.title('TIRS11 apparent temperature calculation differences')
    #plt.xlim([8,10])
    

    
    

    
    
    
    
    
    
    
    
    