#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 07:13:24 2022

@author: tkpci
"""

import numpy as np
import sys
sys.path.append('/cis/staff2/tkpci/Code/Python/TIGR/')
import read_tape6
import run_RTTOV
import pandas as pd
import pdb
from multiprocessing import Process, Manager,Pool, cpu_count
import csv
import os


# compare MODTRAN to RTTOV

def main_MODTRAN(emis10, emis11, num = 100):
    
    
    
    dataOut = {}
    dataOut['LST'] = []
    dataOut['emis10'] = emis10
    dataOut['emis11'] = emis11
    dataOut['T10'] = []
    dataOut['T11'] = []
    dataOut['rad10'] = []
    dataOut['rad11'] = []
    dataOut['trans10'] = []
    dataOut['trans11'] = []
    dataOut['upwell10'] = []
    dataOut['upwell11'] = []
    #dataOut['MODTRAN_rad10'] = []
    #dataOut['MODTRAN_rad11'] = []
    
    
    
    filename = 'tape6_' + str(num)
    filepath = '/cis/staff/tkpci/modtran/tape6_RTTOV/'
    tape6 = read_tape6.read_tape6(filename,filepath)
    
    wavelength = tape6['wavelength']
    
    
    # download RSR for specific sensor - nominal
    RSR10 = load_RSR(sensor='TIRS10')
    RSR11 = load_RSR(sensor='TIRS11')
    
    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
    
    
    T = tape6['skintemp']
    dataOut['LST'] = T
    
    dataOut['rad10'],dataOut['trans10'],dataOut['upwell10'],dataOut['down10']= calc_TOA_radiance(tape6,RSR10_interp,T, emis10)
    dataOut['rad11'],dataOut['trans11'],dataOut['upwell11'],dataOut['down11']= calc_TOA_radiance(tape6,RSR11_interp,T, emis11)
    dataOut['T10'] = LUT_radiance_apptemp(dataOut['rad10'], band = 'b10')
    dataOut['T11'] = LUT_radiance_apptemp(dataOut['rad11'], band = 'b11')
    dataOut['RH'] = tape6['RH']
    dataOut['CWV'] = tape6['CWV']
    
   
    # print('MODTRAN:')
    # for key in dataOut:
    #     print(key, ' : ', dataOut[key])
        
    return dataOut
     
 

def load_RSR(sensor='TIRS10',center=10.9,FWHM=0.5):
    
    # load RSR of specified sensor
    
    filepath = '/cis/staff/tkpci/modtran/RSR/'
    if sensor == 'TIRS10':
        filename = filepath + 'TIRS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'TIRS11':
        filename = filepath + 'TIRS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,2]
    elif sensor == 'MODIS31':
        filename = filepath + 'MODIS_31.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,2]
        rsr = data[:,3]
    elif sensor == 'MODIS32':
        filename = filepath + 'MODIS_32.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,2]
        rsr = data[:,3]

    RSR = {}
    RSR['wave']= wave
    RSR['rsr'] = rsr
        
    return RSR

def interp_RSR(wavelength,RSR, toplot = 1):
    # interp RSR to tape6 wavelangth
    RSR_interp = np.interp(wavelength, RSR['wave'], RSR['rsr'])

    if toplot == 1:
        plt.plot(wavelength, RSR_interp)
        
    return RSR_interp
        
def calc_TOA_radiance(tape6,RSR_interp,T,emis):
    # calculate TOA radiance
    h = 6.626 * 10**(-34);
    c = 2.998 * 10**8;
    k = 1.381 * 10**(-23);
    

    wavelength = tape6['wavelength'] * 10**(-6)
    
    surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (wavelength**(5) * (np.exp((h*c) / (wavelength * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
    
    radiance = np.trapz(list(RSR_interp * radiance_spectral),x=list(wavelength),axis=0)/np.trapz(list(RSR_interp),x=list(wavelength),axis=0)
    
    trans = np.trapz(list(RSR_interp * tape6['transmission']),x=list(wavelength),axis=0)/np.trapz(list(RSR_interp),x=list(wavelength),axis=0)
    
    downwell = np.trapz(list(RSR_interp * tape6['ground_reflected']),x=list(wavelength),axis=0)/np.trapz(list(RSR_interp),x=list(wavelength),axis=0)
   
    upwell = np.trapz(list(RSR_interp * tape6['path_thermal']),x=list(wavelength),axis=0)/np.trapz(list(RSR_interp),x=list(wavelength),axis=0)
   
    total_rad = np.trapz(list(RSR_interp * tape6['total_rad']),x=list(wavelength),axis=0)/np.trapz(list(RSR_interp),x=list(wavelength),axis=0)
    
    return np.round(radiance,2), np.round(trans,2), np.round(upwell,2), np.round(downwell,2)

def appTemp(radiance, band='b10'):
    
    if band == 'b10':
        K2 = 1321.0789;
        K1 = 774.8853;
    elif band == 'b11':
        K2 = 1201.1442;
        K1 = 480.8883;
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')    
        return
    
    temperature = np.round(np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1))),2);
    
    return temperature

def LUT_radiance_apptemp(radiance, band='b10'):
    
    if band == 'b10':
        LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/TIGR/LUT_TIRS_10.csv', delimiter=',')
    elif band == 'b11':
        LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/TIGR/LUT_TIRS_11.csv', delimiter=',')
        
    rad = radiance
    LUT_rad = LUT[:,0]
    LUT_temp = LUT[:,1]
    LUT_rad = LUT_rad.T
    LUT_temp = LUT_temp.T
    
    T = np.round(np.interp(radiance, LUT_rad, LUT_temp),3)
    

    return T    

# calculate TOA radiance from RTTOV to check
def RTTOV_rad_from_output(out):

    RTTOV_calc = {}
    h = 6.626 * 10**(-34);
    c = 2.998 * 10**8;
    k = 1.381 * 10**(-23);
     
    wavelengthB10 = 10.9 / 10**6
    wavelengthB11 = 12 / 10**6
     
     
    surfEmis = out['RTTOV']['emis10'] * out['RTTOV']['trans10'] * 2*h*c**(2) / (wavelengthB10**(5) * (np.exp((h*c) / (wavelengthB10 * out['RTTOV']['LST'] * k))-1)) * 10**(-6)
    RTTOV_calc['rad10'] = surfEmis + out['RTTOV_no_emis']['down10'] + out['RTTOV_no_emis']['upwell10']*(1-out['RTTOV']['emis11'] )*out['RTTOV']['trans10']
    RTTOV_calc['rad10'] = np.round(RTTOV_calc['rad10'],3)
    
 
    surfEmis = out['RTTOV']['emis11'] * out['RTTOV']['trans11'] * 2*h*c**(2) / (wavelengthB11**(5) * (np.exp((h*c) / (wavelengthB11 * out['RTTOV']['LST'] * k))-1)) * 10**(-6)
    RTTOV_calc['rad11'] = surfEmis + out['RTTOV_no_emis']['down11'] + out['RTTOV_no_emis']['upwell11']*(1-out['RTTOV']['emis11'] )*out['RTTOV']['trans11']
    RTTOV_calc['rad11'] = np.round(RTTOV_calc['rad11'],3)

    RTTOV_calc['T10'] = np.round(LUT_radiance_apptemp(RTTOV_calc['rad10'], band='b10'),3)
    RTTOV_calc['T11'] = np.round(LUT_radiance_apptemp(RTTOV_calc['rad11'], band='b11'),3)
    
    return RTTOV_calc


def run_one_profile(num = 150):
    
    #num =150
    emis10 = 0.98
    emis11 = 0.98
    
    dataOut = main_MODTRAN(emis10, emis11, num)
    
    dataRTTOV = run_RTTOV.run_RTTOV(emis10, emis11, num) 
    dataRTTOV2 = run_RTTOV.run_RTTOV(0.00001, 0.00001, num) 
    
    data = pd.DataFrame.from_dict(dataOut,orient='index')
    rttov = pd.DataFrame.from_dict(dataRTTOV,orient='index')
    rttov2 = pd.DataFrame.from_dict(dataRTTOV2,orient='index')
    
    out = pd.concat([data, rttov, rttov2], axis=1)
    
    #pdb.set_trace()
    
    out.columns = ['MODTRAN', 'RTTOV', 'RTTOV_no_emis']
    
    RTTOV_calc = RTTOV_rad_from_output(out)
    rttov_calc = pd.DataFrame.from_dict(RTTOV_calc,orient='index')
    
    out = pd.concat([data, rttov, rttov2, rttov_calc], axis=1)
    
    out.columns = ['MODTRAN', 'RTTOV', 'RTTOV_no_emis', 'RTTOV_calc']
    
    out['Difference'] = np.round(out['MODTRAN'] - out['RTTOV_calc'],3)
    
    out['Units'] = ['K','','','K','K','W/m^2/sr/um','W/m^2/sr/um','','','W/m^2/sr/um','W/m^2/sr/um','W/m^2/sr/um','W/m^2/sr/um','%','g/cm^2']
    
    #pd.set_option('display.max_columns', None)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    print(out)


def run_all_profiles(num):

    emis10 = 0.98
    emis11 = 0.98
    
    out = pd.DataFrame(columns =('LST','RTTOV_LST','MODTRAN T10','MODTRAN T11','MODTRAN rad10','MODTRAN rad11','RTTOV T10','RTTOV T11','RTTOV rad10','RTTOV rad11','diff_rad10','diff_rad11', 'RH', 'CWV'))

    
    print(num)
    dataOut = main_MODTRAN(emis10, emis11, num)
    dataRTTOV = run_RTTOV.run_RTTOV(emis10, emis11, num) 
    
    out.loc[0] = [dataOut['LST'],dataRTTOV['LST'],dataOut['T10'],dataOut['T11'],dataOut['rad10'],dataOut['rad11'],dataRTTOV['T10'],dataRTTOV['T11'],dataRTTOV['rad10'],dataRTTOV['rad11'], dataOut['rad10']- dataRTTOV['rad10'],dataOut['rad11'] - dataRTTOV['rad11'], dataOut['RH'], dataOut['CWV']]
 
    write_data_line_by_line(out)
    

def write_data_line_by_line(out):
    
 
    headers = ",".join(out.keys())
    values = ",".join(str(e) for e in out.iloc[0])     
    
    
    # write data to txt file
    filename_out = '/dirs/data/tirs/RTTOV/analysis/MODTRAN_RTTOV_compare_RH.csv'
    
    if not os.path.isfile(filename_out):
        
        with open(filename_out, mode='w') as file_out:
            csv.excel.delimiter=';'
            file_writer = csv.writer(file_out, dialect=csv.excel)
        
            file_writer.writerow([headers])
            file_writer.writerow([values])
            
    else:
        with open(filename_out, mode='a') as file_out:
            csv.excel.delimiter=';'
            file_writer = csv.writer(file_out, dialect=csv.excel)
        
            file_writer.writerow([values])


def main():
    
       
    myProcesses = []
    profiles = 2311
    
    for val in range(profiles):        
        myProcesses.append(Process(target=run_all_profiles, args=(val,)))

        
    print(str(val+1) + " instances created")
    
    cores = 20
    
    iter = int(profiles/cores)
    print("Running " + str(cores) + " processes at a time")
    
    for i in range(iter+1):
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > profiles:
            end_cnt = profiles
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
            
        process.close() 
        
# run all the  profiles with multiprocessing - output will be saved to a csv  (change name in "write_data_line_by_line" )       
#main() 

# run one profiles - enter TIGR profile number
run_one_profile(num = 2100)  
 
    
 
