#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:05:55 2020

@author: tkpci

Fundction input:
    b10_DC = TIRS band10 level one digital count
    b11_DC = TIRS band11 level one digital count
    ls_emis_final_b10 = emissivity for TIRS band 10
    ls_emis_final_b11 = emissivity for TIRS band 11
    
Function output:
    splitwindowLST = split window land surface temperature with gain and bias adjustment and averaging filter
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

plt.figure()


def get_coefficients():
    
    # last column is nominal SW coefficients (others are based on CWV)
    # matlab coefficients used in SW analysis of all sites
    data = {'SW_b0': [    -0.1317,    9.3846,    3.0183,   -7.9966,   28.9947,    2.2925],
            'SW_b1': [     1.0023  ,    0.9641  ,    0.9837  ,    1.0119  ,    0.8801  ,    0.9929  ],
            'SW_b2': [     0.1614  ,    0.1409  ,    0.1065  ,    0.0709  ,    0.0387  ,    0.1545  ],
            'SW_b3': [    -0.3366  ,   -0.2244  ,   -0.1309  ,   -0.0664  ,   -0.0183  ,   -0.3122  ],
            'SW_b4': [     3.8036  ,    5.9854  ,    6.2594  ,    9.2077  ,   10.2441  ,    3.7186  ],
            'SW_b5': [     5.9164  ,    5.5947  ,    7.2512  ,    7.8600  ,    9.9162  ,    0.3502  ],
            'SW_b6': [    -0.0861  ,  -11.2023  ,  -15.2243  ,  -13.5138  ,  -16.6597  ,   -3.5889  ],
            'SW_b7': [     0.0313,   -0.0216,    0.0171,   -0.1217,   -0.0814,    0.1825],
            'SW_std': [ 0.3585,0.4962,0.6397,0.8436,0.7961,0.7212]}
    
    SW_coeff = pd.DataFrame(data)
    
    return SW_coeff

# appTemp function converts TOA radiance of band 10 and band 11 to apparent temperature
def appTemp(radiance, band='b10'):
    
    # K1 and K2 constants to be found in MTL.txt file for each band
    if band == 'b10':
        K2 = 1321.0789;
        K1 = 774.8853;
    elif band == 'b11':
        K2 = 1201.1442;
        K1 = 480.8883;
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')
        return
    
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)));
    
    return temperature

# TIRS_radiance function converts TIRS band 10 and band 11 digital count to radiance with/without the 
# additional gain bias adjustment as per calval decisions - this should be imcluded in the level 2 data - just check
def TIRS_radiance(counts, band='b10', GB='yes'):
    
    radiance = counts.astype(float) * 3.3420 * 10**(-4) + 0.1

    if (band == 'b10' and GB =='yes'):
        radiance = radiance * 1.0151 - 0.14774
    elif (band == 'b11' and GB =='yes'):
        radiance = radiance * 1.06644 - 0.46326;
    
    return radiance


def calcSW(dataOut, ls_emis_final_b10, ls_emis_final_b11):

    
    b10_DC = dataOut.getRad('rad10')
    b11_DC = dataOut.getRad('rad11')
    
    # convert DC to radiance with/witout addtional gain bias adjustment
    radiance_b10_GB_adjust = TIRS_radiance(b10_DC, band='b10', GB='yes')
    radiance_b11_GB_adjust = TIRS_radiance(b11_DC, band='b11', GB='yes')
    
    # convert TOA radiance to apparent temperature 
    appTemp_b10 = appTemp(radiance_b10_GB_adjust, band='b10')
    appTemp_b11 = appTemp(radiance_b11_GB_adjust, band='b11')
    
    dataOut.setRad('t10',appTemp_b10.astype(float))
    dataOut.setRad('t11',appTemp_b11.astype(float))
    dataOut.setRad('rad10',radiance_b10_GB_adjust.astype(float))
    dataOut.setRad('rad11',radiance_b11_GB_adjust.astype(float))
          

    # calculate the 5x5 averaged app temp for SW algorithm difference terms
    kernel = np.ones((5,5),np.float32)/25
   
    appTemp_b10_ave = cv2.filter2D(appTemp_b10,-1,kernel)
    appTemp_b11_ave = cv2.filter2D(appTemp_b11,-1,kernel)
    
    where_are_NaNs = np.isnan(appTemp_b10_ave)
    appTemp_b10_ave[where_are_NaNs] = 0
    
    where_are_NaNs = np.isnan(appTemp_b11_ave)
    appTemp_b11_ave[where_are_NaNs] = 0
    
    
    # calculate terms for split window algorithm
    T_diff = (appTemp_b10_ave - appTemp_b11_ave)/2
    T_plus =  (appTemp_b10 + appTemp_b11)/2
    e_mean = (ls_emis_final_b10 + ls_emis_final_b11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (ls_emis_final_b10-ls_emis_final_b11)/(e_mean**2)
    quad = (appTemp_b10_ave-appTemp_b11_ave)**2
    
    # split window coefficients
    SW_coeff = get_coefficients()  
    
    # load nominal SW coefficients = 5
    coeff = SW_coeff.iloc[5]
       
    # calculate split window LST  
    splitwindowLST = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad
        
        
    return splitwindowLST


    
    
    
    
    
    
    
    
    
    
    
    
    
