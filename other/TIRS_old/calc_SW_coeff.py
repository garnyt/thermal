# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:10:33 2019

@author: Tania Kleynhans
"""

import numpy as np
import statsmodels.api as sm
from myLib.infoTIRS import appTemp

def calc_SW_coeff(dataOut, quad = 1):
    
    
    T10 =  appTemp(dataOut.getRad10(), band='b10')
    T11 =  appTemp(dataOut.getRad11(), band='b11')
    
    e10 = np.array(dataOut.getEmis10())
    e11 = np.array(dataOut.getEmis11())
    
    T_diff = (T10 - T11)/2
    T_plus =  (T10 + T11)/2
    e_mean = (e10 + e11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (e10-e11)/(e_mean**2)
    quad = (T10-T11)**2
    #b0 = np.ones(T_diff.shape)
    
    y = np.array(dataOut.getSkinTemp())
    
    x = []
    # b0
    #x.append(b0)
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
    x = sm.add_constant(x)
    
    #results = np.linalg.lstsq(x,y)
    
    results = sm.OLS(y, x).fit()
    coeff = results.params
    print(results.summary())
    
    test_coeff(coeff, x, y)
    
    return results, coeff


def test_coeff(coeff, x, y):
    x = x.T
    LST = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    diff = LST-y
    print('Standard Deviation = ' + str(np.std(diff)))
    print('Mean difference = ' + str(np.average(diff)))
    
    #LST = 2.2925 + 0.9929*x[1,:] + 0.1545*x[2,:] -0.3122*x[3,:] + 3.7186*x[4,:] + 0.3502*x[5,:] -3.5889*x[6,:] + 0.1825*x[7,:]
   
