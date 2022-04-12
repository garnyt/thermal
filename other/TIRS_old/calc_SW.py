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
import pdb
import pandas as pd
from scipy import ndimage


def get_coeffifients():
    
   
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

def appTemp_new_coeff(radiance, band='b10'):
    
    # K1 and K2 constants to be found in MTL.txt file for each band
    if band == 'b10':
        K2 = 1320.6539 # new derived coefficients
        K1 = 775.1682  # new derived coefficients
    elif band == 'b11':
        K2 = 1200.8253  # new derived coefficients
        K1 = 481.1861  # new derived coefficients
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


def calcCWV(dataOut):
    
    # CWV calculation based on 10x10km image size = must still ajust for large image
    # assuming pixel of interest is in center of image
    # coefficients
    c0 = 9.2084
    c1 = -2.110
    c2 = -6.9566
        
    appTemp_b10 = dataOut.getRad('t10')
    appTemp_b11 = dataOut.getRad('t11')
    cloud = dataOut.getRad('cloud')
    
    #cloud_new = ndimage.binary_erosion(cloud,structure=np.ones((15,15))).astype(np.int)
    
    appTemp_b10_cloud = np.multiply(appTemp_b10,cloud)
    appTemp_b11_cloud = np.multiply(appTemp_b11,cloud)
    
    appTemp_b10_cloud[appTemp_b10_cloud == 0] = np.nan
    mean_b10 = np.nanmean(appTemp_b10_cloud.flatten())
    
    appTemp_b11_cloud[appTemp_b11_cloud == 0] = np.nan
    mean_b11 = np.nanmean(appTemp_b11_cloud.flatten())
    
    ratio_top = np.nansum(np.multiply((appTemp_b10_cloud-mean_b10),(appTemp_b11_cloud-mean_b11)))   
    ratio_bottom = np.nansum(np.square(appTemp_b10_cloud-mean_b10))
    
    ratio = np.divide(ratio_top,ratio_bottom)
    
    cwv = c0 + c1*ratio + c2*np.square(ratio)
    
    return cwv

def calcCWV_fullimage(dataOut):
    
    # CWV calculation based on 10x10km image size = must still ajust for large image
    # assuming pixel of interest is in center of image
    # coefficients
    c0 = 9.2084
    c1 = -2.110
    c2 = -6.9566
    
    N = 300 # 36km in landsat pixels
        
    appTemp_b10 = dataOut.getRad('t10')
    appTemp_b11 = dataOut.getRad('t11')
    cloud = dataOut.getRad('cloud')
    
    kernel = 30
    cloud_new = ndimage.binary_erosion(cloud,structure=np.ones((kernel,kernel))).astype(np.int)
    
    appTemp_b10_cloud = np.multiply(appTemp_b10,cloud_new)
    appTemp_b11_cloud = np.multiply(appTemp_b11,cloud_new)
    appTemp_b10_cloud[appTemp_b10_cloud == 0] = np.nan
    appTemp_b11_cloud[appTemp_b11_cloud == 0] = np.nan
    
    # create CWV out map
#    cwv = np.zeros([cloud.shape[0],cloud.shape[1]])
    
    # loop through image where possible
#    row_start = int(N/2)
#    row_end = int(cloud.shape[0]-N/2)
#    col_start = int(N/2)
#    col_end = int(cloud.shape[1]-N/2)
    
    
    import gdal
    import subprocess
    lon_value = 144.843333
    lat_value =  -37.673333
    image_large = '/dirs/data/tirs/downloads/test/LC80930862019243LGN00/LC08_L1TP_093086_20190831_20190916_01_T1_B10.TIF'
    # find pixel location based on lat lon coordinates of site
    pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + image_large + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
   
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    x = int(numbers[0])
    y = int(numbers[1])
    
    pix_T10 = appTemp_b10_cloud[y,x]
    pix_T11 = appTemp_b11_cloud[y,x]
    
#    t10_sml = appTemp_b10_cloud[int(y-N/2):int(y+N/2),int(x-N/2):int(x+N/2)]
#    mean_b10 = np.nanmean(t10_sml.flatten())
#    t11_sml = appTemp_b11_cloud[int(y-N/2):int(y+N/2),int(x-N/2):int(x+N/2)]            
#    mean_b11 = np.nanmean(t11_sml.flatten())
#    
#    #using pixel
#    ratio_top = np.nansum(np.multiply((t10_sml-pix_T10),(t11_sml-pix_T11)))   
#    ratio_bottom = np.nansum(np.square(t10_sml-pix_T10))
#    
#    #using mean
#    ratio_top = np.nansum(np.multiply((t10_sml-mean_b10),(t11_sml-mean_b11)))   
#    ratio_bottom = np.nansum(np.square(t10_sml-mean_b10))
#    
#    ratio = np.divide(ratio_top,ratio_bottom)
#    
#    cwv = c0 + c1*ratio + c2*np.square(ratio)

    
    from scipy import signal
    
    scharr = np.ones([N,N])/(N**2)
    scharr_N = np.ones([N,N])
    
    T10 = appTemp_b10
    T10[np.isnan(T10)] = 0
    
    T11 = appTemp_b11
    T11[np.isnan(T11)] = 0
    
    mean_10 = signal.fftconvolve(T10, scharr, mode='same')
    mean_11 = signal.fftconvolve(T11, scharr, mode='same')
    plt.imshow(mean_10,vmin=250, vmax=300)
    
    diff_10 = T10-mean_10
    diff_11 = T10-mean_11
    sqr_10 = (T10-mean_10)**2
    
    diff = np.multiply(diff_10,diff_11)
    
    ratio_top = signal.fftconvolve(diff, scharr_N, mode='same')
    ratio_bottom = signal.fftconvolve(sqr_10, scharr_N, mode='same')
    
    ratio = np.divide(ratio_top,ratio_bottom)
    
    cwv = c0 + c1*ratio + c2*np.square(ratio)
    plt.imshow(cwv,vmin=0, vmax=8)
    
    
#    for i in range(row_end-row_start):
#        for j in range(col_end-col_start):
#    
#            # define new area
#            t10_sml = appTemp_b10_cloud[row_start+i:row_start+i+N, col_start+j:col_start+j+N]
#            mean_b10 = np.nanmean(t10_sml.flatten())
#            t11_sml = appTemp_b11_cloud[row_start+i:row_start+i+N, col_start+j:col_start+j+N]            
#            mean_b11 = np.nanmean(t11_sml.flatten())
#            
#            ratio_top = np.nansum(np.multiply((t10_sml-mean_b10),(t11_sml-mean_b11)))   
#            ratio_bottom = np.nansum(np.square(t10_sml-mean_b10))
#            
#            ratio = np.divide(ratio_top,ratio_bottom)
#            
#            cwv[int(row_start+i+N/2),int(col_start+j+N/2)] = c0 + c1*ratio + c2*np.square(ratio)
    
    return cwv


def calcSW(dataOut, ls_emis_final_b10, ls_emis_final_b11):

    
    b10_DC = dataOut.getRad('rad10')
    b11_DC = dataOut.getRad('rad11')
    
    # convert DC to radiance with/witout addtional gain bias adjustment
    radiance_b10_GB_adjust = TIRS_radiance(b10_DC, band='b10', GB='no')
    radiance_b11_GB_adjust = TIRS_radiance(b11_DC, band='b11', GB='no')
    
    # convert TOA radiance to apparent temperature 
    appTemp_b10 = appTemp_new_coeff(radiance_b10_GB_adjust, band='b10')
    appTemp_b11 = appTemp_new_coeff(radiance_b11_GB_adjust, band='b11')
    
    dataOut.setRad('t10',appTemp_b10.astype(float))
    dataOut.setRad('t11',appTemp_b11.astype(float))
    dataOut.setRad('rad10',radiance_b10_GB_adjust.astype(float))
    dataOut.setRad('rad11',radiance_b11_GB_adjust.astype(float))

   
    try:
        # calculte column water vapor
        cloud = dataOut.getRad('cloud')
        if cloud[167,167] == 0:
            cwv = 'nan'
            cwv_range = 5
        else:
            if np.shape(appTemp_b10)[0] < 1000:
                cwv = calcCWV(dataOut)
            else:
                print('  CWV not yet coded for full scene analysis - using 0 to 6.3 coefficients.')
                # assign SW coefficients 
                cwv = 9999                
            if cwv <= 2.0:
                cwv_range = 0
            elif cwv > 2 and cwv <= 3:
                cwv_range = 1
            elif cwv > 3 and cwv <= 4:
                cwv_range = 2
            elif cwv > 4 and cwv <= 5:
                cwv_range = 3
            elif cwv > 5 and cwv <= 6.3:
                cwv_range = 4
            else:
                cwv_range = 5
    except:
         cwv_range = 5
         cwv = 'nan'

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
    SW_coeff = get_coeffifients()  
    
    coeff = SW_coeff.iloc[cwv_range]
    
    # look at individual terms

    # b0 = coeff[0]
    # b1 = coeff[1]*T_plus
    # b2 = coeff[2]*T_plus*e_diff
    # b3 = coeff[3]*T_plus*e_change
    # b4 = coeff[4]*T_diff
    # b5 = coeff[5]*T_diff*e_diff
    # b6 = coeff[6]*T_diff*e_change
    # b7 = coeff[7]*quad 
    
    # import matplotlib.pyplot as plt
    # import scipy
    
    # mask = np.where(np.isinf(b2) ,0,1)
    # mask = scipy.ndimage.morphology.binary_erosion(mask,structure=np.ones((9,9)))
    
    # img = b2
    # img = np.where(np.isinf(img), 0, b0)

    # plt.imshow(img, vmin = 0, vmax = 2)
    # plt.axis('off')
    # plt.colorbar()
    # plt.title('SW term b0 = 2.2925 K')  
        
    # calculate split window LST  
    splitwindowLST_CWV = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad
     
    coeff = SW_coeff.iloc[5]   
    # calculate split window LST  
    splitwindowLST = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad
        
    splitwindowLST[splitwindowLST>350] = np.nan
        
    return splitwindowLST,splitwindowLST_CWV, cwv, cwv_range


    

    
    
    
    
    
    
    
    
    
    
    
    