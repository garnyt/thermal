#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Mon Sept 14, 2020

@author: Tania Kleynhans, Rochester Institute of Technology

Fundction input:
    b10_DC = TIRS band10 collection two digital count (if collection 1 data - see note above TIRS_radiance function)
    b11_DC = TIRS band11 collection two digital count (if collection 1 data - see note above TIRS_radiance function)
    ls_emis_final_b10 = emissivity for TIRS band 10
    ls_emis_final_b11 = emissivity for TIRS band 11
    a13_std = aster standard emissivity deviation band 13
    a14_std = aster standard emissivity deviation band 14

Function output:
    splitwindowLST = split window land surface temperature with gain and bias adjustment and averaging filter
    splitwindowQA = split window land surface temperature uncertainty product
"""

import numpy as np
import pdb
from scipy import signal
import SaveGeotif

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
# additional gain bias adjustment as per calval decisions - this should be included in the collection 2 data (along with some geometric updates)
# if input is collection 2 data, then GB='no', else "yes"
def TIRS_radiance(counts, band='b10'):

    radiance = counts.astype(float) * 3.3420 * 10**(-4) + 0.1


    return radiance

def SW_coeff():
    
    # split window coefficients as per journal article: Towards an Operational, Split Window-Derived Surface Temperature Product
    #for the Thermal Infrared Sensors Onboard Landsat 8 and 9, A.Gerace, T.Kleynhans, R.Eon, M. Montanaro
    
    coeff = []
    coeff.append(2.2925)  #b0
    coeff.append(0.9929)  #b1
    coeff.append(0.1545)  #b2
    coeff.append(-0.3122)  #b3
    coeff.append(3.7186)  #b4
    coeff.append(0.3502)  #b5
    coeff.append(-3.5889)  #b6
    coeff.append(0.1825)  #b7
    # regression error
    coeff.append(0.73) 
    
    return coeff

# calculating SW 
def calcSW(b10_DC, b11_DC, ls_emis_final_b10, ls_emis_final_b11):

    # convert DC to radiance with("yes")/witout("no") addtional gain bias adjustment
    # if collection 2 data is used, then use "no". If collection 1 data is used, then "yes"
    radiance_b10_GB_adjust = TIRS_radiance(b10_DC, band='b10')
    radiance_b11_GB_adjust = TIRS_radiance(b11_DC, band='b11')

    # convert TOA radiance to apparent temperature
    appTemp_b10 = appTemp(radiance_b10_GB_adjust, band='b10')
    appTemp_b11 = appTemp(radiance_b11_GB_adjust, band='b11')

    # calculate the 5x5 averaged app temp for SW algorithm difference terms
    # kernel = np.ones((5,5),np.float32)/25
    
    #  # note that appTemp must be float values for this function to work and fill value must be nan
    # appTemp_b10[appTemp_b10 == 0] = np.nan
    # appTemp_b11[appTemp_b11 == 0] = np.nan
    # # create fill value mask to applly to appTemp_bxx_ave after convolution
    # mask10 = np.isnan(appTemp_b10)*1   #to get 0-1 values
    # mask11 = np.isnan(appTemp_b11)*1   #to get 0-1 values
    # appTemp_b10_ave = signal.convolve2d(appTemp_b10,kernel,mode='same')
    # appTemp_b11_ave = signal.convolve2d(appTemp_b11,kernel,mode='same')
    

    # # combine and apply mask
    # mask10[mask11 == 1] = 1
    # appTemp_b10_ave[mask10 ==1] = np.nan
    # appTemp_b11_ave[mask10 ==1] = np.nan

    
    # calculate terms for split window algorithm
    T_diff = (appTemp_b10 - appTemp_b11)/2
    T_plus =  (appTemp_b10 + appTemp_b11)/2
    e_mean = (ls_emis_final_b10 + ls_emis_final_b11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (ls_emis_final_b10-ls_emis_final_b11)/(e_mean**2)
    quad = (appTemp_b10-appTemp_b11)**2

    # split window coefficients
    coeff = SW_coeff()
    

    # calculate split window LST
    splitwindowLST = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad

    return splitwindowLST, appTemp_b10, appTemp_b11


# calculate uncertainty metric by adding error in quadrature
def calculateError(T10, T11, e10, e11, a13_std, a14_std):
    
        
    # emssivity conversion coefficients to convert ASTER to Landsat emis = c0 + c1*e13 + c1*e14
    c0_val_b10 = 0.010103
    c1_val_b10 = 0.56472
    c2_val_b10 = 0.42545
    c0_val_b11 = 0.1117
    c1_val_b11 = -0.55987
    c2_val_b11 = 1.4465
    
    # regression error associated with above conversion
    c_total_error_b10 = 0.000972
    c_total_error_b11 = 0.00545
       
    # split window coefficients
    coeff = SW_coeff()
    
    b0 = coeff[0]
    b1 = coeff[1]
    b2 = coeff[2]
    b3 = coeff[3]
    b4 = coeff[4]
    b5 = coeff[5]
    b6 = coeff[6]
    b7 = coeff[7]
    b_total_error = coeff[8]
    
        
    # correlation coefficient for apparent temperature and emissivity
    # this was calculated using the 113 MODIS emissivities (spectrally sampled)
    # the appTemp (apparent temperaturebetween B10 and B11) correlation was calculated using the TIGR simulation data
    corr_emis = 0.701
    corr_appTemp = 0.999
    
    
    # uncertainty in apparent temperature - hardcoded based on Montanaro et al.: Derivation and validation of the 
    # stray light correction algorithm for the thermal infrared sensor onboard Landsat 8
    # and as discussed in virtual CalVal meeting 2020 March 
    T10_error = 0.15
    T11_error = 0.20 

    # first calculate the uncertainty in the emissivity calculation   
    #emissivity uncertainty calculation with adding the covariance term (adding error in quadrature)
    e10_error_cov = np.sqrt((c_total_error_b10)**2 + (c1_val_b10*a13_std)**2 + (c2_val_b10*a14_std)**2 + 2*corr_emis*c1_val_b10*c2_val_b10*a13_std*a14_std)
    e11_error_cov = np.sqrt((c_total_error_b11)**2 + (c1_val_b11*a13_std)**2 + (c2_val_b11*a14_std)**2 + 2*corr_emis*c1_val_b11*c2_val_b11*a13_std*a14_std)   

    
    # partial derivatives of SW algorithm (commented code at bottom shows how this was calculated)   
    T10diff = b1/2 + b2*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b3*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b4/2 + b5*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b6*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b7*(2*T10 - 2*T11)
    T11diff = b1/2 + b2*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b3*(e10 - e11)/(2*(e10/2 + e11/2)**2) - b4/2 - b5*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) - b6*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b7*(-2*T10 + 2*T11)
    E10diff = -b2*(T10 + T11)/(4*(e10/2 + e11/2)) - b2*(T10 + T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) + b3*(T10 + T11)/(2*(e10/2 + e11/2)**2) - b3*(T10 + T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3) - b5*(T10 - T11)/(4*(e10/2 + e11/2)) - b5*(T10 - T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) + b6*(T10 - T11)/(2*(e10/2 + e11/2)**2) - b6*(T10 - T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3)
    E11diff = -b2*(T10 + T11)/(4*(e10/2 + e11/2)) - b2*(T10 + T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) - b3*(T10 + T11)/(2*(e10/2 + e11/2)**2) - b3*(T10 + T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3) - b5*(T10 - T11)/(4*(e10/2 + e11/2)) - b5*(T10 - T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) - b6*(T10 - T11)/(2*(e10/2 + e11/2)**2) - b6*(T10 - T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3)
    
    
    # SW uncertainty in quadrature with correlation
    splitwindowQA = np.sqrt(b_total_error**2 + (T10diff*T10_error)**2 + (T11diff*T11_error)**2 + (E10diff*e10_error_cov)**2 + (E11diff*e11_error_cov)**2 + \
                        2*corr_appTemp*T10diff*T11diff*T10_error*T11_error + 2*corr_emis*E10diff*E11diff*e10_error_cov*e11_error_cov) 
         
     
    return splitwindowQA


def main(b10_DC, b11_DC, ls_emis_final_b10, ls_emis_final_b11, a13_std, a14_std):
    
    
    # calculate SW
    splitwindowLST, appTemp_b10, appTemp_b11 = calcSW(b10_DC, b11_DC, ls_emis_final_b10, ls_emis_final_b11)
    
    # calcualte SW uncertainty
    splitwindowQA = calculateError(appTemp_b10, appTemp_b11, ls_emis_final_b10, ls_emis_final_b11, a13_std, a14_std)
    
    return splitwindowLST, splitwindowQA   


from osgeo import gdal
import matplotlib.pyplot as plt


def test_scenes():
    
           
    filename = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_P023_R036/LC80230362019136LGN00/LC08_L1TP_023036_20190516_20200719_02_T1'
    b10Name = filename + '_B10.TIF'
    b11Name = filename + '_B11.TIF'
    e10_name = filename + '_emis10.tif'
    e11_name =  filename +  '_emis11.tif' 
    LST =  filename +  '_SW_LST.tif'      
        
    gdal.UseExceptions()
    rd=gdal.Open(b10Name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    b10= rd.GetRasterBand(1)
    b10_DC = b10.ReadAsArray()
    b10_DC =b10_DC.astype(float)
    
    b10_DC[b10_DC <= 0] = np.nan
    
    gdal.UseExceptions()
    rd=gdal.Open(b11Name)
    b11= rd.GetRasterBand(1)
    b11_DC = b11.ReadAsArray()
    b11_DC =b11_DC.astype(float)
    
    b11_DC[b11_DC <= 0] = np.nan

    gdal.UseExceptions()
    rd=gdal.Open(e10_name)
    b11= rd.GetRasterBand(1)
    ls_emis_final_b10 = b11.ReadAsArray()
    ls_emis_final_b10 =ls_emis_final_b10.astype(float)
    
    ls_emis_final_b10[ls_emis_final_b10 <= 0] = np.nan

    gdal.UseExceptions()
    rd=gdal.Open(e11_name)
    b11= rd.GetRasterBand(1)
    ls_emis_final_b11 = b11.ReadAsArray()
    ls_emis_final_b11 =ls_emis_final_b11.astype(float)
    
    ls_emis_final_b11[ls_emis_final_b11 <= 0] = np.nan

    gdal.UseExceptions()
    rd=gdal.Open(LST)
    b11= rd.GetRasterBand(1)
    SW_LST = b11.ReadAsArray()
    SW_LST =SW_LST.astype(float)
    
    SW_LST[SW_LST <= 0] = np.nan
    
    a13_std = 0
    a14_std = 0
    
    ###### plot to see if concolve ignores fill values
    
    
    temp = appTemp_b10
    temp[temp > 0] = 1
    tempave = appTemp_b11
    tempave[tempave > 0] = 1
    diff = temp-tempave
    
    diff = SW_LST - splitwindowLST
    plt.imshow(diff, vmin = -0.2, vmax = 0.2)
    plt.colorbar()
    np.nanmin(diff)
    np.nanmax(diff)
    
    splitwindowLST, splitwindowQA = main(b10_DC, b11_DC, ls_emis_final_b10, ls_emis_final_b11, a13_std, a14_std)
    
    SaveGeotif.saveGeotif(splitwindowLST, '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_P023_R036/LC80230362019136LGN00','LC08_L1TP_023036_20190516_20200719_02_T1_SW_LST_no_filter.tif', wkt_projection, geoTransform)
    
    


## how the derivative was calculated 
# import sympy

# T10, T11, e10, e11, b0,b1,b2,b3,b4,b5,b6,b7 = sympy.symbols('T10 T11 e10 e11 b0 b1 b2 b3 b4 b5 b6 b7') 
# LST = b0 + \
#     b1*(T10+T11)/2 + \
#         b2*(T10+T11)/2 * (1-((e10+e11)/2))/((e10+e11)/2) + \
#             b3*(T10+T11)/2 * (e10-e11)/(((e10+e11)/2)**2) + \
#                 b4*(T10-T11)/2 + \
#                     b5*(T10-T11)/2 *(1-((e10+e11)/2))/((e10+e11)/2) + \
#                         b6*(T10-T11)/2*(e10-e11)/(((e10+e11)/2)**2) + \
#                             b7*((T10-T11)**2)
# T10_diff = sympy.diff(LST, T10) 
# T11_diff = sympy.diff(LST, T11) 
# e10_diff = sympy.diff(LST, e10) 
# e11_diff = sympy.diff(LST, e11) 

# a13_std = dataOut.getAster('emis13_std')
# a14_std = dataOut.getAster('emis14_std')
# b10_DC = dataOut.getRad('rad10')
# b11_DC = dataOut.getRad('rad11')
# ls_emis_final_b10 = dataOut.getRad('e10')
# ls_emis_final_b11 = dataOut.getRad('e11')

# import matplotlib.pyplot as plt

# plt.subplot(1,3,1)
# plt.imshow(splitwindowLST, vmin = 290, vmax = 310)
# plt.axis("off")
# plt.colorbar()
# plt.title('SW LST')
# plt.subplot(1,3,2)
# plt.imshow(splitwindowQA, vmin = 0.5, vmax = 4)
# plt.axis("off")
# plt.colorbar()
# plt.title('SW Uncertainty')
# plt.subplot(1,3,3)
# plt.imshow(a13_std, vmin = 0, vmax = 0.1)
# plt.axis("off")
# plt.title('ASTER emis std')
# plt.colorbar()



  