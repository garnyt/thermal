#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:23:37 2022

@author: tkpci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_SW_coeff_RTTOV():

    data = pd.read_csv('/dirs/data/tirs/RTTOV/analysis/TIGR_RTTOV_data3.csv')
    data = data.drop(data[data['RH'] > 90].index)
    data_array = data.to_numpy()
    
    
    # create array for skin temperature variations
    T = np.asarray([-10,-5,0,5,10,15,20]) # all 7 iterations
    T_num = T.shape[0]
    
    # read RSR of sensor
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    filename = filepath + 'TIRS.csv'
    rsr = np.genfromtxt(filename, delimiter=',')
    wavelengths = rsr[:,0]
    RSR_10 = rsr[:,1]
    RSR_11 = rsr[:,2]
        
    # spectrally sample MODIS emissivity files 
    emis10 = emis_interp(wavelengths,RSR_10)
    emis11 = emis_interp(wavelengths,RSR_11)
    
    emis_num = emis10.shape[0]
    
    # add emissivities to the data
    data_array_emis = np.repeat(data_array, emis_num, axis=0) # repeats each line 113 time
    emis10_tile = np.tile(emis10,data_array.shape[0]) # repeats all data 
    emis11_tile = np.tile(emis11,data_array.shape[0]) # repeats all data 
    
    data_array_emis = np.column_stack((data_array_emis,emis10_tile))
    data_array_emis = np.column_stack((data_array_emis,emis11_tile))
    
    # add temperatures to the data
    data_array_emis_temp = np.repeat(data_array_emis, T_num, axis=0) # repeats each line 113 time
    T_tile = np.tile(T,data_array_emis.shape[0]) # repeats all data 
    lst = T_tile+data_array_emis_temp[:,0]
    data_array_emis_temp = np.column_stack((data_array_emis_temp,lst))
    
    # add data into dataframe
    out = pd.DataFrame(data_array_emis_temp, 
                       columns =('skintemp',
                                'upwell10',
                                'down10',
                                'trans10',
                                'upwell11',
                                'down11',
                                'trans11',
                                'RH',
                                'emis10',
                                'emis11',
                                'LST'))
   
    
    out = RTTOV_rad_from_output(out)

    calc_SW_coeff_only(out)
    
    
def emis_interp(wavelength,RSR_interp):
    
    emis = []
    
    filename = '/cis/staff2/tkpci/emissivities/SoilsMineralsParsedEmissivities.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    
    for i in range(data.shape[1]-1):
        temp = np.interp(wavelength, wave, data[:,i+1])
        emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
        
    
    filename = '/cis/staff2/tkpci/emissivities/VegetationParsedEmissivities.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    
    for i in range(data.shape[1]-1):
        temp = np.interp(wavelength, wave, data[:,i+1])
        emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))

    filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]

    for i in range(data.shape[1]-1):
        temp = np.interp(wavelength, wave, data[:,i+1])
        emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
      
    emis = np.asarray(emis)         
        
    return emis
        

def LUT_radiance_apptemp(radiance, band='b10'):
    
    if band == 'b10':
        LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/TIGR/LUT_TIRS_10.csv', delimiter=',')
    elif band == 'b11':
        LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/TIGR/LUT_TIRS_11.csv', delimiter=',')
        
    rad = radiance
    LUT_rad = LUT[:,0]
    LUT_temp = LUT[:,1]
    LUT_rad = LUT_rad.T
    LUT_temp = LUT_temp.T
    
    T = np.round(np.interp(radiance, LUT_rad, LUT_temp),4) 

    return T  

def calc_SW_coeff_only(data):
    
    T10 = data['T10'].to_numpy()
    T11 = data['T11'].to_numpy()
    e10 = data['emis10'].to_numpy()
    e11 = data['emis11'].to_numpy()
    
    T_diff = (T10 - T11)/2
    T_plus =  (T10 + T11)/2
    e_mean = (e10 + e11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (e10-e11)/(e_mean**2)
    quad = (T10-T11)**2
    
    b0 = np.ones(T_diff.shape)

    y = data['LST'].to_numpy()
    
    x = []
    # b0
    x.append(b0)
    # b1
    x.append(T_plus)
    # b2
    x.append(T_plus*e_diff)
    # b3
    x.append(T_plus*e_change)
    # b4
    x.append(T_diff)
    # b5
    x.append(T_diff*e_diff)
    # b6
    x.append(T_diff*e_change)
    # b7
    x.append(quad)
        
    x = np.array(x).T
    
    coeff, residuals, rank, s = np.linalg.lstsq(x,y,rcond=None)
    
    x = x.T
    LST = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    
    #diff2 = LST-y
    rmse = np.sqrt(np.mean((LST - y)**2)) 
    print(coeff)
    print('RMSE: ',rmse)
    print('Ave diff: ',np.mean((LST - y)))


# calculate TOA radiance from RTTOV to check
def RTTOV_rad_from_output(data):

    h = 6.626 * 10**(-34);
    c = 2.998 * 10**8;
    k = 1.381 * 10**(-23);
     
    wavelengthB10 = 10.9 / 10**6
    wavelengthB11 = 12 / 10**6
     
     
    surfEmis = data['emis10'] * data['trans10'] * 2*h*c**(2) / (wavelengthB10**(5) * (np.exp((h*c) / (wavelengthB10 * data['LST'] * k))-1)) * 10**(-6)
    data['rad10'] = surfEmis + data['down10'] + data['upwell10']*(1-data['emis10'] )*data['trans10']
 
    surfEmis = data['emis11'] * data['trans11'] * 2*h*c**(2) / (wavelengthB11**(5) * (np.exp((h*c) / (wavelengthB11 * data['LST'] * k))-1)) * 10**(-6)
    data['rad11'] = surfEmis + data['down11'] + data['upwell11']*(1-data['emis11'] )*data['trans11']

    data['T10'] = LUT_radiance_apptemp(data['rad10'].to_numpy(), band='b10')
    data['T11'] = LUT_radiance_apptemp(data['rad11'].to_numpy(), band='b11')
    
    return data


    
    