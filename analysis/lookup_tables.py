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
        
    return wave, rsr
        
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
    temperature_range = np.arange(210,330,0.05)
    
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
    radiance_range = np.arange(2,15,0.05)
    #temperature_range = np.arange(210,330,0.05)
    
    wave_table = np.tile(wave_range,(radiance_range.shape[0],1))
    radiance = np.tile(radiance_range,(wave_range.shape[0],1))
    radiance = radiance.T
    T_table = inverse_planck(radiance,wave_table)
    
    # wave_table = np.tile(wave_range,(temperature_range.shape[0],1))
    # temperature = np.tile(temperature_range,(wave_range.shape[0],1))
    # temperature = temperature.T
        
    # L_table = planck(temperature,wave_table)
    
    return T_table, radiance_range, wave_range, wave_table


def appTemp(radiance, band='b10'):
    
    if band == 'b10':
        K2 = 1321.0789
        K1 = 774.8853
        #K2 = 1320.6539 # new derived coefficients
        #K1 = 775.1682# new derived coefficients
    elif band == 'b11':
        K2 = 1201.1442
        K1 = 480.8883
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')    
        return
    
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)))
    
    return temperature
    
def calc_appTemp_from_table():
    
    L_table, temperature_range, wave_range, wave_table = lookup_radiance_table()
    #T_table, radiance_range, wave_range, wave_table = lookup_temperature_table()
    
    wave, rsr = load_RSR(sensor='TIRS10')
    
    rsr_interp = np.interp(np.squeeze(wave_table[0,:]), wave, rsr)
    rsr_interp = np.tile(rsr_interp, (L_table.shape[0],1))
    
    radiance = np.trapz(list(rsr_interp * L_table),x=wave_range,axis=1)/np.trapz(list(rsr_interp),x=wave_range,axis=1)
    
    
    temperature_coeff = appTemp(radiance, band='b10')
    
    
    bb_temperature = inverse_planck(radiance,12)
    
    plt.plot(radiance, temperature_range-temperature_coeff)
    plt.plot(radiance, temperature_range-bb_temperature)
    plt.xlabel('Radiance [W/$m^2$.sr.$\mu$m]')
    plt.ylabel('Temperature difference [K]')
    plt.title('Temperature differences using LUT as truth')
    plt.legend(('TIRS coefficients','Inverse Planck','New coefficients'))
    
    
    # calculating coefficients
    
    
    
from scipy.optimize import curve_fit

def func(radiance, K2, K1):

    return np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)))


idx = np.arange(0,2400,10)    
xdata = radiance[idx]
ydata = temperature_range[idx]

popt, pcov = curve_fit(func, xdata, ydata) 
    
    
rsr_interp = np.interp(np.squeeze(wave_table[0,:]), wave, rsr)
rsr_interp = np.tile(rsr_interp, (T_table.shape[0],1))
    
temperature = np.trapz(list(rsr_interp * T_table),x=wave_range,axis=1)/np.trapz(list(rsr_interp),x=wave_range,axis=1)   
    
idx = np.arange(0,260)    
xdata = radiance_range[idx]
ydata = temperature[idx]

popt, pcov = curve_fit(func, xdata, ydata)   
    
    

    
    
    
    
    
    
    
    
    