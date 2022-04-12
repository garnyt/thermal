#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 12 2021

@author: tkpci

Fundction input:dataOut (dictionary)
    dataOut['rad10'] = TIRS band10 level one digital count
    dataOut['rad11'] = TIRS band11 level one digital count
    dataOut['emis10'] = emissivity for TIRS band 10
    dataOut['emis11'] = emissivity for TIRS band 11
    
Function output:
    SW_LST 
"""

import numpy as np
import cv2
import pdb
import pandas as pd
from scipy import ndimage
import re


# appTemp function converts TOA radiance of band 10 and band 11 to apparent temperature
# def apptemp(dataOut, band ='b10'):
    
#     sensor = dataOut['landsat']
#     if sensor == '08':
    
#         # K1 and K2 constants to be found in MTL.txt file for each band
#         if band == 'b10':
#             K2 = 1321.0789;
#             K1 = 774.8853;
#             radiance = dataOut['rad10']
#         elif band == 'b11':
#             K2 = 1201.1442;
#             K1 = 480.8883;
#             radiance = dataOut['rad11']
#         else:
#             print('call function with T = appTemp(radiance, band=\'b10\'')
#             return
#     elif sensor == '09':
    
#         # K1 and K2 constants to be found in MTL.txt file for each band
#         if band == 'b10':
#             K2 = 1329.2405;
#             K1 = 799.0284;
#             radiance = dataOut['rad10']
#         elif band == 'b11':
#             K2 = 1198.3494;
#             K1 = 475.6581;
#             radiance = dataOut['rad11']
#         else:
#             print('call function with T = appTemp(radiance, band=\'b10\'')
#             return
    
    
#     temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)));
    
#     return temperature

def apptemp_LUT(dataOut, band='b10'):
    
    
    sensor = dataOut['landsat']
    if sensor == '08':
        if band == 'b10':
            LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/spec_files/LUT_TIRS10.csv', delimiter=',')
            rad = dataOut['rad10']
        elif band == 'b11':
            LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/spec_files/LUT_TIRS11.csv', delimiter=',')
            rad = dataOut['rad11']
    elif sensor == '09':
        if band == 'b10':
            LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/spec_files/LUT_TIRS2_10.csv', delimiter=',')
            rad = dataOut['rad10']
        elif band == 'b11':
            LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/spec_files/LUT_TIRS2_11.csv', delimiter=',')
            rad = dataOut['rad11']

    
    T = np.zeros([rad.shape[0], rad.shape[1]])
    T = np.zeros([rad.shape[0]* rad.shape[1]])
    
    rad_flat = rad.flatten()
    T = np.interp(rad, LUT[:,0], LUT[:,1])
    T = np.reshape(T, [dataOut['rad10'].shape[0],dataOut['rad10'].shape[1]])
    
    # for i in range(rad.shape[0]):   
       
    #     T[i] = np.interp(rad[i,:], LUT[:,0], LUT[:,1])

    return T    

# convert digital count of thermal channels to radiance and apparent temperatures
def TIRS_apparent_temperatures(dataOut):
    
    f = open(dataOut['MTL_file'],"r")
    content = f.read()

    rad10_mult_index = content.index('RADIANCE_MULT_BAND_10')
    rad11_mult_index = content.index('RADIANCE_MULT_BAND_11')
    rad10_add_index = content.index('RADIANCE_ADD_BAND_10')
    rad11_add_index = content.index('RADIANCE_ADD_BAND_11')
    
    s = content[rad10_mult_index:rad10_mult_index+35]
    rad10_mult = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    s = content[rad11_mult_index:rad11_mult_index+35]
    rad11_mult = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    s = content[rad10_add_index:rad10_add_index+35]
    rad10_add = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    s = content[rad11_add_index:rad11_add_index+35]
    rad11_add = re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?',s)
    
    
    # convert band 10 and 11 to radiance values
    dataOut['rad10'] = dataOut['rad10'].astype(float) * float(rad10_mult[-1]) + float(rad10_add[-1])
    dataOut['rad11'] = dataOut['rad11'].astype(float) * float(rad11_mult[-1]) + float(rad11_add[-1])
    
    # convert band 10 and band 11 radiance to apparent temperature values
    #dataOut['T10'] = apptemp(dataOut, band ='b10')
    #dataOut['T11'] = apptemp(dataOut, band ='b11')
    dataOut['T10'] = apptemp_LUT(dataOut, band ='b10')
    dataOut['T11'] = apptemp_LUT(dataOut, band ='b11')
    
    return dataOut


def main(dataOut):

    
    # convert thermal channel DC's to radiance and to apparent temperatures
    dataOut = TIRS_apparent_temperatures(dataOut)
    if dataOut['landsat'] == '08':
        sensor = 'L8'
    elif dataOut['landsat'] == '09':
        sensor = 'L9'

    coeff = dataOut['variables'][sensor]['SW_coefficients']

    # calculate the 5x5 averaged apparent temp for SW algorithm difference terms
    kernel = np.ones((5,5),np.float32)/25
    
    appTemp_b10 = dataOut['T10']
    appTemp_b11 = dataOut['T11']
    
    try:
        if dataOut['filter'] == 'no':
            appTemp_b10_ave = appTemp_b10
            appTemp_b11_ave = appTemp_b11
            print('NOTE: No 5x5 filtering applied')
        else:
            appTemp_b10_ave = cv2.filter2D(appTemp_b10,-1,kernel)
            appTemp_b11_ave = cv2.filter2D(appTemp_b11,-1,kernel)
            print('NOTE: 5x5 filtering applied')
    except:
        appTemp_b10_ave = cv2.filter2D(appTemp_b10,-1,kernel)
        appTemp_b11_ave = cv2.filter2D(appTemp_b11,-1,kernel)
            
    where_are_NaNs = np.isnan(appTemp_b10_ave)
    appTemp_b10_ave[where_are_NaNs] = 0
    
    where_are_NaNs = np.isnan(appTemp_b11_ave)
    appTemp_b11_ave[where_are_NaNs] = 0
    
    # calculate terms for split window algorithm
    T_diff = (appTemp_b10_ave - appTemp_b11_ave)/2
    T_plus =  (appTemp_b10 + appTemp_b11)/2
    e_mean = (dataOut['emis10'] + dataOut['emis11'])/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (dataOut['emis10']-dataOut['emis11'])/(e_mean**2)
    quad = (appTemp_b10_ave-appTemp_b11_ave)**2
    
    # calculate split window LST  
    splitwindowLST = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad
     
    # set irrational values to NaN
    splitwindowLST[splitwindowLST>350] = np.nan
    
    # try:
    #     if dataOut['filter'] == 'No':
    #         dataOut['SW_LST_no_filter'] = splitwindowLST
    #     else:
    #         dataOut['SW_LST'] = splitwindowLST
    # except:
    dataOut['SW_LST'] = splitwindowLST
    
    
        
    return dataOut


    

    
    
    
    
    
    
    
    
    
    
    
    