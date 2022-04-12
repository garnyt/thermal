#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:19:17 2021

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt

# DC to radiance calculator
# variables

# read tape6 info into dictionary
def read_tape6(filename,filepath='/cis/staff2/tkpci/modtran/tape6_predefined/'):
    

    infile = open(filepath+filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()   
    
    tape6 = {}
    word = 'WAVLEN'
    start_ind = []
    end_ind = []
    for i in range(0,len(lines)):
        k=0
        kk = 0
        if word in lines[i]:
            
            wave = lines[i+4]
            while not str.isnumeric(wave[4:6]):
                k += 1
                wave = lines[i+4+k]
            start_ind.append(i+4+k)
            while str.isnumeric(wave[4:6]):
                k += 1
                wave = lines[i+4+k] 
            end_ind.append(i+4+k-1)                    
            for j in range(i+4+k,i+4+58):
                if 'THIS WARNING WILL NOT BE REPEATED' in lines[j]:
                    start_ind.append(j+1)
                    wave = lines[j+1] 
                    while str.isnumeric(wave[4:6]):
                        kk += 1
                        wave = lines[j+1+kk] 
                    end_ind.append(j+kk+1)   
    
    
    
    data = []
    for i in range(len(start_ind)):
        data.append(lines[start_ind[i]:end_ind[i]])

    
    results = []
    for i in range(0,len(data)):
        data1 = data[i]
        for j in range(0,len(data1)):
            prse = data1[j].split(' ')
            try:
                float(prse[6])
            except:
                pass
            else:
                while ("" in prse):
                    prse.remove("")
                results.append(prse)


    ind = []
    for i in range(0,len(results)):
        if len(results[i]) == 15:
             ind.append(i)
       
    output_data = np.asarray(results[0:ind[-1]]).astype(float)

    tape6['wavelength'] = output_data[:,1] # In Microns
    tape6['path_thermal'] = output_data[:,3] * 10**4 # In W/m2/sr/um
    tape6['ground_reflected'] = output_data[:,9] * 10**4 # In W/m2/sr/um
    tape6['transmission'] = output_data[:,14]
    tape6['radiance'] = output_data[:,12] * 10**4 # In W/m2/sr/um
    # tape6['CWV'] = np.float64(CWV)
    # tape6['RH'] = RH
    # tape6['skintemp'] = skintemp

    idx = np.argsort(tape6['wavelength'])
    tape6['wavelength'] = tape6['wavelength'][idx]
    tape6['path_thermal'] = tape6['path_thermal'][idx]
    tape6['ground_reflected'] = tape6['ground_reflected'][idx]
    tape6['transmission'] = tape6['transmission'][idx]
    tape6['radiance'] = tape6['radiance'][idx]
    
    # tape6['dewpoint'] = T_profile - ((100-relH)/5)
    # tape6['T_profile'] = T_profile
    # tape6['KM_profile'] = KM_profile
    # tape6['RH_profile'] = relH

    return tape6

A_det = 25E-6 ** 2  #m^2 area of each detector
f_num = 1.64 # f number
t = 3.49E-3  # sec integration time
tau_optics = 0.12 # this is wrong, but to make values work use it
QE = 0.01  # saw note on this stating < 1%
em = 0

filepath = '/cis/staff2/tkpci/modtran/RSR/'
filename = filepath + 'TIRS.csv'
data = np.genfromtxt(filename, delimiter=',')
wavelength = data[:,0] # micron
rsr = data[:,1]

h = 6.626 * 10**(-34) # J.s
c = 2.998 * 10**8 # m/s

tape6 = read_tape6('tape6',filepath='/cis/staff2/tkpci/modtran/tape6_predefined/')
radiance_spectral = np.interp(wavelength, tape6['wavelength'], tape6['radiance'])# In W/m2/sr/um
radiance = np.trapz(rsr * radiance_spectral,x=wavelength)/np.trapz(rsr,x=wavelength)   
 
#radiance_m = radiance_spectral * 10**-6 # W/m^2.sr.m  
value = (A_det * np.pi * (1-em))/(4 * f_num**2 * h * c) * t  * tau_optics * rsr * QE * wavelength*10**-6 * radiance_spectral * tau_optics

temp = value * rsr
temp2 = np.sum(temp)
electrons = np.trapz(value,x=wavelength) 
   
print('Radiance: ' + str(np.round(radiance)))
print('Electrons: ' + str(np.round(electrons)))

MLT_radiance_calc = (radiance + 0.1) / 3.3420E-4 
print('MLT conversion electrons: ' + str(np.round(MLT_radiance_calc,0)))

