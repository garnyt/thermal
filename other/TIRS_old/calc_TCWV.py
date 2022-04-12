#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:29:19 2020

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
#from read_tape6 import read_tape6
import warnings
warnings.filterwarnings("ignore")


    

def calc_TCWV_from_profile(hPa,T,RH):
 
    # T = temperature in Kelvin
    # RH = relative humidity [0-1]
    
    # constants
    avogadro=6.02214199e+23    # Avogadro Number
    loschmidt=2.6867775e+19    # Loschmidt Number
    air_mwt=28.964             # Air Molecular weight (grams)
    h2o_mwt=18.015             # H2O Molecular weigth (grams)
    c1=18.9766                 #
    c2=-14.9595                # Clausius - Clapeyron Constants
    c3=-2.43882                #
    
    p0=1013.25                 # pressure at sea level (hPa)
    t0=273.15
    g=980                      # gravity [cm/s^2]
    p_dense=997                # density of water [kg/m^3] 
    
    p_air = loschmidt*(hPa/p0)*(t0/T)   # aur density (molecules/cm^3) 
    T_a = t0/T
    b = avogadro/h2o_mwt
    r = air_mwt/h2o_mwt
    
    # calc ulate saturation vapor pressure
    e_sat = T_a*b*np.exp(c1+c2*T_a+c3*T_a**2)*10**(-6)*RH
    
    # calculate volume mixing ratio
    vmr = e_sat/(p_air-e_sat)
    vmr[np.isnan(vmr)]=0
    
    # calculate mass mixing ratio (g/Kg)
    mmr = vmr/(r*10**(-3))
    print(str(np.round(mmr,2)))
    
    # mixing ratio
    #mmr = 0.662*vmr
    #TCWV = np.trapz(np.flip(mmr),np.flip(hPa))
    
    TCWV = 1/(g*p_dense)*np.trapz(np.flip(mmr),np.flip(hPa))*1000
    
    print('Calc CWV    = '+str(np.round(TCWV,2)))
    #print('Modtran CWV = '+str(np.round(tape6['CWV'],2)))
    
    #return np.round(tape6['CWV'],2), np.round(TCWV,2)
    return  np.round(TCWV,4)
    

data_out = np.zeros([2311,2])
hPa = np.flip([2.6e-3,8.9e-3,2.4e-2,0.5E-01,0.8999997E-01,0.17E+00,0.3E+00,0.55E+00,
     0.1E+01,0.15E+01,0.223E+01,0.333E+01,0.498E+01,
     0.743E+01,0.1111E+02,0.1660001E+02,0.2478999E+02,0.3703999E+02,
     0.4573E+02,0.5646001E+02,0.6971001E+02,0.8607001E+02,0.10627E+03,
     0.1312E+03,0.16199E+03,0.2E+03,0.22265E+03,0.24787E+03,
     0.27595E+03,0.3072E+03,0.34199E+03,0.38073E+03,0.4238501E+03,
     0.4718601E+03,0.525E+03,0.5848E+03,0.65104E+03,0.72478E+03,
     0.8E+03,0.8486899E+03,0.9003301E+03,0.9551201E+03,0.1013E+04])


    
radio = np.genfromtxt('/cis/staff2/tkpci/radiosonde/melbourne_94866_20190831_0000_P_H_T_D_RH_MR.csv',delimiter=',') 
T = radio[:,2]+273.15
RH = radio[:,4]/100
hPa = radio[:,0]
MR = radio[:,5]   
    
Calc_CWV = calc_TCWV_from_profile(hPa,T,RH)     
    
    
    
for i in range(2311):
    

    filename = 'tape6_'+str(i+1)
    tape6 = read_tape6(filename)
    
    
    #hPa = tape6['hPa_profile']
    T = tape6['T_profile']
    RH = tape6['RH_profile']/100

    M_CWV, Calc_CWV = calc_TCWV_from_profile(hPa,T,RH) 
    
    data_out[i,0] = M_CWV
    data_out[i,1] = Calc_CWV    
   
x = np.arange(2311)

plt.scatter(x,data_out[:,1]-data_out[:,0], s=2)
plt.ylabel('Difference [$g/cm^2$]')
plt.xlabel('TIGR profiles')
plt.title('Calculated vs. MODTRAN total CWV')
    
# plt.scatter(x,data_out[:,1], s=2) 

#np.min(data_out[:,1])  
#np.max(data_out[:,1])  
#
#cnt = data_out[:,0]
#cnt2 = cnt[cnt < 6.3]
#cnt2 = cnt2[cnt2>0.06]

#mmr_tigr = [13.82,	12.63,	11.03,	9.361,	7.794,	5.61,	4.008,	2.881,	1.957,	1.283,	0.8636,	0.5657,	0.3842,	0.2373,	0.1544,	0.09812,	0.05988,	0.03273,	0.01114,	0.004523,	0.002581,	0.002291,	0.002262,	0.002276,	0.002338,	0.002409,	0.002614,	0.002898,	0.003018,	0.003001,	0.003118,	0.003274,	0.003447,	0.003626,	0.003805,	0.003949,	0.003968,	0.003955,	0.003675,	0.00323,	0.002739,	0.001192,	0.0001877]
#1/(g*p_dense)*np.trapz(np.flip(mmr_tigr),np.flip(hPa))*1000