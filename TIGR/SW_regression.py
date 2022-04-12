# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:10:33 2019

@author: Tania Kleynhans
"""

import numpy as np
import pdb


def calc_SW_coeff(dataOut, output):
    
    
    T10 =  np.array(dataOut['T10'])
    T10 = T10.flatten()
    T11 =  np.array(dataOut['T11'])
    T11 = T11.flatten()
    
    e10 = np.array(dataOut['emis10'])
    e10 = e10.flatten()
    e11 = np.array(dataOut['emis11'])
    e11 = e11.flatten()
    
    T_diff = (T10 - T11)/2
    T_plus =  (T10 + T11)/2
    e_mean = (e10 + e11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (e10-e11)/(e_mean**2)
    quad = (T10-T11)**2
    
    b0 = np.ones(T_diff.shape)

    y = np.array(dataOut['skintemp'])
    y = y.flatten()
    
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
    
    output = test_coeff(coeff, x, y, output)
    
    return output


def test_coeff(coeff, x, y, output):
    x = x.T
    LST = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    diff = LST-y
    rmse = np.sqrt(np.mean((LST - y)**2))
    
    output['rmse_new_coeff'].append(round(rmse,2))
    output['stddev_new_coeff'].append(round(np.std(diff),2))
    output['mean_new_coeff'].append(round(np.average(diff),2))
    
#    print('Std Dev =   ' + str(round(np.std(diff),2)))
#    print('RMSE =      ' + str(round(rmse,2)))
#    print('Mean diff = ' + str(round(np.average(diff),2)))
#    
    # test data with normal coefficients calculated by 
    coeff = [ 2.04774777,  0.99370856,  0.16099904, -0.30570014,  3.63023612, -0.48333403, -4.18026657,  0.14660471] # python with 7 * 200 * 30
    LST_SW = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    
    #LST_SW = 2.2925 + 0.9929*x[1,:] + 0.1545*x[2,:] -0.3122*x[3,:] + 3.7186*x[4,:] + 0.3502*x[5,:] -3.5889*x[6,:] + 0.1825*x[7,:]
    diff = LST_SW-y
    rmse = np.sqrt(np.mean((LST_SW - y)**2))
#    print('SW coeff: Std Dev =   ' + str(round(np.std(diff),2)))
#    print('SW coeff: RMSE =      ' + str(round(rmse,2)))
#    print('SW coeff: Mean diff = ' + str(round(np.average(diff),2)))
#    #coeff = 2.74483724,  0.99120743,  0.15640275, -0.28743244,  3.41436121, 0.35403352, -3.68383071,  0.17793674]  #python coeff calc from TIGR 7 * 1386 * 113
    
    output['rmse_orig_coeff'].append(round(rmse,2))
    output['stddev_orig_coeff'].append(round(np.std(diff),2))
    output['mean_orig_coeff'].append(round(np.average(diff),2))
    
    return output
 
  
    
    

#coeff = calc_SW_coeff(dataOut)