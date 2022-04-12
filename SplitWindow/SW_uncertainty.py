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
import pdb

def main(dataOut):
    
    
    mask = np.where(dataOut['rad10'] == 0)
    
    # get aster standard deviation (aster supplied data per scene)
    a13_std = dataOut['emis13_std']
    a14_std = dataOut['emis14_std']
    a13_std[mask] = np.nan
    a14_std[mask] = np.nan
    
    # get calculated apparent temperature
    T10 = dataOut['T10']
    T11 = dataOut['T11']
    # get emissivity
    e10 = dataOut['emis10']
    e11 = dataOut['emis11']
    
    if dataOut['landsat'] == '08':
        sensor = 'L8'
        sensor10 = 'TIRS_10'
        sensor11 = 'TIRS_11'
    elif dataOut['landsat'] == '09':
        sensor = 'L9'
        sensor10 = 'TIRS2_10'
        sensor11 = 'TIRS2_11'
    
    
    # get SW coefficients
    coeff = dataOut['variables'][sensor]['SW_coefficients']
    b0 = coeff[0]
    b1 = coeff[1]
    b2 = coeff[2]
    b3 = coeff[3]
    b4 = coeff[4]
    b5 = coeff[5]
    b6 = coeff[6]
    b7 = coeff[7]
    # get SW regression error
    b_total_error = dataOut['variables']['L8']['SW_regfit']
    
    # uncertainty in apparent temperature 
    T10_error = dataOut['variables'][sensor10]['landsat_uncertainty_10_K']
    T11_error = dataOut['variables'][sensor11]['landsat_uncertainty_11_K']
    # uncertainty in aster to landsat conversion coefficients
    c_total_error_b10 = dataOut['variables'][sensor10]['emis_regfit_10']
    c_total_error_b11 = dataOut['variables'][sensor11]['emis_regfit_11']
    
    # aster to landsat coefficients
    c1_val_b10 = dataOut['variables'][sensor10]['estimated_coeff_1'] 
    c2_val_b10 = dataOut['variables'][sensor10]['estimated_coeff_2'] 
    c0_val_b10 = dataOut['variables'][sensor10]['estimated_coeff_3'] 
    c1_val_b11 = dataOut['variables'][sensor11]['estimated_coeff_1'] 
    c2_val_b11 = dataOut['variables'][sensor11]['estimated_coeff_2'] 
    c0_val_b11 = dataOut['variables'][sensor11]['estimated_coeff_3'] 
    
    # correlation coefficient for apparent temperature and emissivity
    # this was calculated using the 113 MODIS emissivities (spectrally sampled)
    # the appTemp correlation was calculated using the TIGR simulation data
    corr_emis = 0.7
    corr_appTemp = 1
    

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
    
    dataOut['SW_LST_qa'] = LST_error_cov
    
    return dataOut


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
    