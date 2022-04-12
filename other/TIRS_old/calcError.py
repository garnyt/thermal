#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:19:39 2020

dataOut - class containing radiance, apparent temperature, emissivity of ASTER, emissivity std dev of ASTER
e10 = band 10 emissivity adjusted from aster emis files
e11 = band 11 emissivity adjusted from aster emis files

@author: tkpci
"""

#import sympy
#import math
import numpy as np
import matplotlib.pyplot as plt
import calc_SW
import pdb

def calculateError(dataOut, e10, e11, cwv_range):
    
    mask = np.where(dataOut.getRad('rad10') == 0)
        
    # emssivity conversion coefficients - emisL = c0 + c1*e13 + c1*e14
    c0_val_b10 = 0.010103
    c1_val_b10 = 0.56472
    c2_val_b10 = 0.42545
    c0_val_b11 = 0.1117
    c1_val_b11 = -0.55987
    c2_val_b11 = 1.4465
    
    # CWV error values (std dev)
    cwv=[]
    cwv.append(0.0826)       #<=2  g/cm^2
    cwv.append(0.2180)       #2< x <=3 g/cm^2
    cwv.append(0.2226)       #3< x <=4 g/cm^2
    cwv.append(0.2584)       #4 < x <=5 g/cm^2
    cwv.append(0.1851)       #5 < x <= 6.3 g/cm^2
    cwv.append(0.1312)       #<= 6.3 g/cm^2
    
    # error values for regression to get to coefficients
    
    SW_coeff = calc_SW.get_coeffifients() 
    
    
    b_coeff= SW_coeff.iloc[cwv_range]
    b0 = b_coeff[0]
    b1 = b_coeff[1]
    b2 = b_coeff[2]
    b3 = b_coeff[3]
    b4 = b_coeff[4]
    b5 = b_coeff[5]
    b6 = b_coeff[6]
    b7 = b_coeff[7]
    b_total_error = b_coeff[8]
    
    cwv_reggress_error = cwv[cwv_range]
    c_total_error_b10 = 0.000972
    c_total_error_b11 = 0.00545
    
    # correlation coefficient for apparent temperature and emissivity
    # this was calculated using the 113 MODIS emissivities (spectrally sampled)
    # the appTemp correlation was calculated using the TIGR simulation data
    corr_emis = 0.7
    corr_appTemp = 1
    
    # get aster standard deviation (aster supplied data per scene)
    a13_std = dataOut.getAster('emis13_std')
    a14_std = dataOut.getAster('emis14_std')
    a13_std[mask] = np.nan
    a14_std[mask] = np.nan
    
    # get calculated apparent temperature
    T10 = dataOut.getRad('t10')
    T11 = dataOut.getRad('t11')
    
    # uncertainty in apparent temperature - hardcoded according to discussion with Matt   
    T10_error = 0.15
    T11_error = 0.2

    #emissivity uncertainty calculation (adding error in quadrature)
    e10_error = np.sqrt((c_total_error_b10)**2 + (c1_val_b10*a13_std)**2 + (c2_val_b10*a14_std)**2)
    e11_error = np.sqrt((c_total_error_b11)**2 + (c1_val_b11*a13_std)**2 + (c2_val_b11*a14_std)**2)
    e10_error[mask] = np.nan
    e11_error[mask] = np.nan
    
    #emissivity uncertainty calculation with adding the covariance term (adding error in quadrature)
    e10_error_cov = np.sqrt((c_total_error_b10)**2 + (c1_val_b10*a13_std)**2 + (c2_val_b10*a14_std)**2 + 2*corr_emis*c1_val_b10*c2_val_b10*a13_std*a14_std)
    e11_error_cov = np.sqrt((c_total_error_b11)**2 + (c1_val_b11*a13_std)**2 + (c2_val_b11*a14_std)**2 + 2*corr_emis*c1_val_b11*c2_val_b11*a13_std*a14_std)   
    e10_error_cov[mask] = np.nan
    e11_error_cov[mask] = np.nan
    
    # partial derivatives of SW algorithm    
    T10diff = b1/2 + b2*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b3*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b4/2 + b5*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b6*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b7*(2*T10 - 2*T11)
    T11diff = b1/2 + b2*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b3*(e10 - e11)/(2*(e10/2 + e11/2)**2) - b4/2 - b5*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) - b6*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b7*(-2*T10 + 2*T11)
    E10diff = -b2*(T10 + T11)/(4*(e10/2 + e11/2)) - b2*(T10 + T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) + b3*(T10 + T11)/(2*(e10/2 + e11/2)**2) - b3*(T10 + T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3) - b5*(T10 - T11)/(4*(e10/2 + e11/2)) - b5*(T10 - T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) + b6*(T10 - T11)/(2*(e10/2 + e11/2)**2) - b6*(T10 - T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3)
    E11diff = -b2*(T10 + T11)/(4*(e10/2 + e11/2)) - b2*(T10 + T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) - b3*(T10 + T11)/(2*(e10/2 + e11/2)**2) - b3*(T10 + T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3) - b5*(T10 - T11)/(4*(e10/2 + e11/2)) - b5*(T10 - T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) - b6*(T10 - T11)/(2*(e10/2 + e11/2)**2) - b6*(T10 - T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3)
    
    
    # SW uncertainty in quadrature with and without correlation
    LST_error_cov = np.sqrt(b_total_error**2 + (T10diff*T10_error)**2 + (T11diff*T11_error)**2 + (E10diff*e10_error_cov)**2 + (E11diff*e11_error_cov)**2 + \
                        2*corr_appTemp*T10diff*T11diff*T10_error*T11_error + 2*corr_emis*E10diff*E11diff*e10_error_cov*e11_error_cov) 
    LST_error_cov[mask] = np.nan
    
      
    # calculate individual uncertainty per term as % of total uncertainty
    termT = ((T10diff*T10_error)**2 + (T11diff*T11_error)**2)
    termE = ((E10diff*e10_error_cov)**2 + (E11diff*e11_error_cov)**2) 
    termTcov = (2*corr_appTemp*T10diff*T11diff*T10_error*T11_error)
    termEcov = (2*corr_emis*E10diff*E11diff*e10_error_cov*e11_error_cov)
    
    
    
    
     
    # SW uncertainty in quadrature assuming independent variables
    LST_error = np.sqrt(b_total_error**2 + (T10diff*T10_error)**2 + (T11diff*T11_error)**2 + (E10diff*e10_error)**2 + (E11diff*e11_error)**2)    
    LST_error[mask] = np.nan 
    
    return LST_error,LST_error_cov, termT,termE,termTcov,termEcov


# data =LST_error_cov

# plt.close()            
# plt.imshow(data,vmin=np.nanmin(data), vmax=np.nanmax(data))
# plt.colorbar()
# plt.axis('off')

# plt.close()            
# plt.imshow(data,vmin=0, vmax=2)
# plt.colorbar()
# plt.axis('off')
# plt.title('SW LST uncertainty - dependent terms')

# np.nanmean(LST_error_cov)
# np.nanmax(e11_error)

# # calculating the derivative
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

# LST = coeff[0] + \
#     coeff[1]*(T10+T11)/2 + \
#         coeff[2]*(T10+T11)/2 * (1-((e10+e11)/2))/((e10+e11)/2) + \
#             coeff[3]*(T10+T11)/2 * (e10-e11)/(((e10+e11)/2)**2) + \
#                 coeff[4]*(T10-T11)/2 + \
#                     coeff[5]*(T10-T11)/2 *(1-((e10+e11)/2))/((e10+e11)/2) + \
#                         coeff[6]*(T10-T11)/2*(e10-e11)/(((e10+e11)/2)**2) + \
#                             coeff[7]*((T10-T11)**2)
# T10_diff = sympy.diff(LST, T10) 
# T11_diff = sympy.diff(LST, T11) 
# e10_diff = sympy.diff(LST, e10) 
# e11_diff = sympy.diff(LST, e11) 

#emissivity uncertainty calculation with adding the covariance term (adding error in quadrature)
    # e10_error_cov = np.sqrt((c_total_error_b10)**2 + (c1_val_b10*a13_std)**2 + (c2_val_b10*a14_std)**2 + 2*corr_emis*c1_val_b10*c2_val_b10*a13_std*a14_std)
    # e11_error_cov = np.sqrt((c_total_error_b11)**2 + (c1_val_b11*a13_std)**2 + (c2_val_b11*a14_std)**2 + 2*corr_emis*c1_val_b11*c2_val_b11*a13_std*a14_std)   
    # e10_error_cov[mask] = np.nan
    # e11_error_cov[mask] = np.nan
    
    # # partial derivatives of SW algorithm    
    # T10diff = 0.365*T10 - 0.365*T11 + 2.35575 + 0.25235*(-e10/2 - e11/2 + 1)/(e10/2 + e11/2) - 1.95055*(e10 - e11)/(e10/2 + e11/2)**2
    # T11diff = -0.365*T10 + 0.365*T11 - 1.36285 - 0.09785*(-e10/2 - e11/2 + 1)/(e10/2 + e11/2) + 1.63835*(e10 - e11)/(e10/2 + e11/2)**2
    # E10diff = (-1.79445*T10 + 1.79445*T11)/(e10/2 + e11/2)**2 - (-1.79445*T10 + 1.79445*T11)*(e10 - e11)/(e10/2 + e11/2)**3 + (-0.1561*T10 - 0.1561*T11)/(e10/2 + e11/2)**2 - (-0.1561*T10 - 0.1561*T11)*(e10 - e11)/(e10/2 + e11/2)**3 - (0.07725*T10 + 0.07725*T11)/(2*(e10/2 + e11/2)) - (0.07725*T10 + 0.07725*T11)*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)**2) - (0.1751*T10 - 0.1751*T11)/(2*(e10/2 + e11/2)) - (0.1751*T10 - 0.1751*T11)*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)**2)
    # E11diff = -(-1.79445*T10 + 1.79445*T11)/(e10/2 + e11/2)**2 - (-1.79445*T10 + 1.79445*T11)*(e10 - e11)/(e10/2 + e11/2)**3 - (-0.1561*T10 - 0.1561*T11)/(e10/2 + e11/2)**2 - (-0.1561*T10 - 0.1561*T11)*(e10 - e11)/(e10/2 + e11/2)**3 - (0.07725*T10 + 0.07725*T11)/(2*(e10/2 + e11/2)) - (0.07725*T10 + 0.07725*T11)*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)**2) - (0.1751*T10 - 0.1751*T11)/(2*(e10/2 + e11/2)) - (0.1751*T10 - 0.1751*T11)*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)**2)
    