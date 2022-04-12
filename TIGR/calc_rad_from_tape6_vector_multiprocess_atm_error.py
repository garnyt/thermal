#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:12:21 2020

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pdb
import pandas as pd
from multiprocessing import Process, Manager,Pool, cpu_count
import os
from datetime import datetime
import warnings
import csv
import re

warnings.filterwarnings("ignore")


def calc_rad_from_tape6(dir_files,start_val, stop_val):
    
    
    # run tape6 once to get wavelength that RSR and emis can be done outside of loop
    tape6 = read_tape6('tape6_tape5_1029_T_-2.5_cwv_-10.0_skintemp_285.55',filepath='/cis/staff/tkpci/modtran/tape6_TIGR_uncertainty_sml/')
    wavelength_orig = tape6['wavelength']
    tape6['lst_file'] = []
    
     
    #standard TIRS bands
    center10 = 10.9
    FWHM10 = 0.59
    center11 = 12
    FWHM11 = 1.01
    
       
    #RSR to use for analysis
    RSR10 = load_RSR(center10,FWHM10, sensor='TIRS10')    #TIRS10 = 10.9  & 0.59
    RSR11 = load_RSR(center11,FWHM11, sensor='TIRS11')    #TIRS11 = 12.0  &1.01 
    # interpolate to wavelength
    RSR10_interp_orig = interp_RSR(wavelength_orig, RSR10, toplot = 0)
    #RSR11_interp_orig = interp_RSR(wavelength_orig, RSR11, toplot = 0)
    
        
    # create dictionary to hold data
    dataOut = {}
    dataOut['rad10'] = []
    dataOut['rad11'] = []
    dataOut['emis10'] = []
    dataOut['emis11'] = []
    dataOut['skintemp'] = []
    dataOut['CWV'] = []
    dataOut['RH'] = []
    dataOut['T10'] = []
    dataOut['T11'] = []
    dataOut['trans10'] = []
    dataOut['trans11'] = []
    dataOut['T_error'] = []
    dataOut['cwv_error'] = []
    dataOut['tape6'] = []
    
    temp =re.findall('[-+]?\d*\.\d+|\d+','tape6_tape5_1029_T_-2.5_cwv_-10.0_skintemp_285.55')
    
    T_error = []
    cwv_error = []
    skintemp = []
    
    T_error.append(float(temp[3]))
    cwv_error.append(float(temp[4]))
    skintemp.append(float(temp[5]))
    
    tape6['lst_file'] = float(temp[5])
    
    old_tape6 = tape6
    cnt = start_val
    
    try:
        
        done = 1
        RH90 = 1

        for i in range(stop_val-start_val):
            
            if 'tape6' in dir_files[cnt]:
                
                tape6 = read_tape6(dir_files[cnt] , filepath='/cis/staff/tkpci/modtran/tape6_TIGR_uncertainty_sml/')
    
                if tape6['RH']< 90:
                    
                    temp =re.findall('[-+]?\d*\.\d+|\d+',dir_files[cnt])
                    
                    tape6['lst_file'] = float(temp[5])
                    
                    T_error.append(float(temp[3]))
                    cwv_error.append(float(temp[4]))
                    #get skintemp from filename
                    skintemp.append(float(temp[5]))
                                       
                    #print('Busy with ',dir_files[cnt] )
                    new_tape6 = {**old_tape6, **tape6}
                    for key, value in new_tape6.items():
                        if key in old_tape6 and key in tape6:
                            if isinstance(tape6[key],float):
                                new_tape6[key] = [np.hstack([value , np.squeeze(np.asarray(old_tape6[key]))])]
                            else:
                                new_tape6[key] = [np.vstack([value , np.squeeze(np.asarray(old_tape6[key]))])]
                    old_tape6 = new_tape6
                    done += 1
                else:
                    RH90 += 1
                
            cnt+=1
         
        print('Completed tape6: ',done)
        print('RH > 90 files: ', RH90)
        tape6 = new_tape6 
        
        
        tape6['skintemp'] = tape6['lst_file']
    
        #pdb.set_trace()         
            
        # calculate radiance for both bands
        T = np.asarray([-10,-5,0,5,10,15,20]) # all 7 iterations
        T_num = T.shape[0]
        
        # just to get number of emis files
        emis10, emis_spectral = emis_interp(wavelength_orig,RSR10_interp_orig, num=0)
        
        emis_num = emis10.shape[0]
    
        wavelength = np.squeeze(np.asarray(tape6['wavelength'])) * 10**(-6)
        wavelength = np.transpose(wavelength)
        files_num = wavelength.shape[1]
        wavelength = np.repeat(wavelength,emis_num*T_num,axis=1)
           
        skintemp = np.squeeze(np.asarray(tape6['skintemp']))
        skintemp = np.repeat(skintemp[:,np.newaxis],wavelength.shape[0],axis=1)
        skintemp = np.transpose(skintemp)
        skintemp = np.repeat(skintemp,emis_num*T_num,axis=1)
        
        T = np.repeat(T[np.newaxis,:],wavelength.shape[0],axis=0)
        
        T = np.repeat(T,emis_num,axis=1)
        T = np.tile(T,files_num)
        
        T = T+skintemp
        skintemp_row = T[0,:]
        
        trans = np.squeeze(np.asarray(tape6['transmission']))
        trans = np.transpose(trans)
        
        
        path = np.squeeze(np.asarray(tape6['path_thermal']))
        path = np.transpose(path)
        ground = np.squeeze(np.asarray(tape6['ground_reflected']))
        ground = np.transpose(ground)
        
        trans = np.repeat(trans,emis_num*T_num,axis=1)
        path = np.repeat(path,emis_num*T_num,axis=1)
        ground = np.repeat(ground,emis_num*T_num,axis=1)
        
        tape6_vector = {}
        tape6_vector['wavelength'] = wavelength
        tape6_vector['path_thermal'] = path
        tape6_vector['ground_reflected'] = ground
        tape6_vector['transmission'] = trans
        
        dataOut['skintemp'] = skintemp_row
    
        temp = np.asarray(tape6['CWV'] )
        CWV_bin = temp 
        CWV_bin = np.where(temp < 2, 1, CWV_bin)
        CWV_bin = np.where(temp > 2, 2, CWV_bin)
        CWV_bin = np.where(temp > 3, 3, CWV_bin)
        CWV_bin = np.where(temp > 4, 4, CWV_bin)
        CWV_bin = np.where(temp > 5, 5, CWV_bin)
        CWV_bin = np.where(temp > 6, 6, CWV_bin)
        
        dataOut['CWV_bin'] = np.repeat(np.squeeze(CWV_bin),7*emis_num,axis=0)
    
    
        CWV = np.squeeze(np.asarray(tape6['CWV']))
        CWV = np.repeat(CWV,7*emis_num,axis=0)
        RH = np.squeeze(np.asarray(tape6['RH']))
        RH = np.repeat(RH,7*emis_num,axis=0)
        
        dataOut['CWV'] = CWV 
        dataOut['RH'] = RH
        
        dataOut['T_error'] = np.repeat(T_error,7*emis_num,axis=0)
        dataOut['cwv_error'] = np.repeat(cwv_error,7*emis_num,axis=0)
        
        dataOut = loop_through_data(dataOut,RSR10,RSR11,wavelength_orig,wavelength,T_num,files_num,tape6_vector,T)
        dataOut = calc_SW_and_save(dataOut)
    except:
        print('This iteration did not work')
        


def main_processes():
    
    startTime = datetime.now()
       
    myProcesses = []
    
    filepath='/cis/staff/tkpci/modtran/tape6_TIGR_uncertainty_sml/'
    dir_files = os.listdir(filepath)
    
    num_iter = 1
        
    iterations = int(len(dir_files)/num_iter)
    start_val = 0
    stop_val = start_val + iterations
    
    for val in range(100):        
        myProcesses.append(Process(target=calc_rad_from_tape6, args=(dir_files, start_val, stop_val,)))
        calc_rad_from_tape6(dir_files, start_val, stop_val)
        start_val += stop_val
        stop_val += iterations
        

        
    print(str(iterations) + " instances created")
    
    
    cores = 20
    print("Running " + str(cores) + " processes at a time")
    iter = int(num_iter/cores)
    
    for i in range(iter+1):
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > num_iter:
            end_cnt = num_iter
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.close()
        
    

    print('\nTime elasped: ', datetime.now() - startTime)
        

     
########################################

def loop_through_data(dataOut,RSR10,RSR11,wavelength_orig,wavelength,T_num,files_num,tape6_vector,T):
    
    
        
    RSR10_new = {}
    RSR10_new['wave'] = RSR10['wave']
    RSR11_new = {}
    RSR11_new['wave'] = RSR11['wave']
    
    RSR10_new['rsr'] = RSR10['rsr']
    RSR11_new['rsr'] = RSR11['rsr']
        
    RSR10_interp_new = interp_RSR(wavelength_orig, RSR10_new, toplot = 0)
    RSR11_interp_new = interp_RSR(wavelength_orig, RSR11_new, toplot = 0)
    
    RSR10_interp = np.repeat(RSR10_interp_new[:,np.newaxis],wavelength.shape[1],axis=1)
    RSR11_interp = np.repeat(RSR11_interp_new[:,np.newaxis],wavelength.shape[1],axis=1)
               
    # get UCSB emissivity files and interpolate to wavelenght
    emis10, emis_spectral10 = emis_interp(wavelength_orig,RSR10_interp_new, num=0)
    emis11, emis_spectral11 = emis_interp(wavelength_orig,RSR11_interp_new, num=0)
    
    emis_spectral10 = np.transpose(emis_spectral10)
    emis_spectral11 = np.transpose(emis_spectral11)
    
    emis_spectral10 = np.tile(emis_spectral10,T_num*files_num)
    emis_spectral11 = np.tile(emis_spectral11,T_num*files_num)
    
    emis10 = np.tile(emis10,T_num*files_num)
    emis11 = np.tile(emis11,T_num*files_num)

    rad10, T10_planck, trans10 = calc_TOA_radiance2(tape6_vector,RSR10_interp,T,emis_spectral10, wavelength)
    rad11, T11_planck,trans11 = calc_TOA_radiance2(tape6_vector,RSR11_interp,T,emis_spectral11, wavelength)
    dataOut['rad10'] = rad10
    dataOut['rad11'] = rad11
    dataOut['emis10'] = emis10
    dataOut['emis11'] = emis11

    # calc appTemp from TIRS band centers lookup table
    dataOut['T10'] = LUT_radiance_apptemp(rad10, band='b10')
    dataOut['T11'] = LUT_radiance_apptemp(rad11, band='b11')
    dataOut['trans10'] = trans10
    dataOut['trans11'] = trans11
    
    

    # calct appTemp using TIRS coefficients in MTL file    
    # dataOut['T10_coeff'] = appTemp(rad10, band='b10')
    # dataOut['T11_coeff'] = appTemp(rad11, band='b11')

    
    # # calc appTemp from TIRS band centers lookup table
    # dataOut['T10_LUT'] = LUT_radiance_apptemp(rad10, band='b10')
    # dataOut['T11_LUT'] = LUT_radiance_apptemp(rad11, band='b11')
       
    return dataOut
   

###########################################read in tape6 file ###########################################
    
def read_tape6(filename,filepath='/cis/staff2/tkpci/modtran/tape6/'):

    infile = open(filepath+filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()

    word = 'REL H'
    start_ind = []
    start_ind_T = []
    start_ind_KM= []
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            relH = lines[i+3]
            #pdb.set_trace()
            RH_line = relH[30:33]
            RH_line = RH_line.replace(" ","")
            while str.isnumeric(RH_line):
                start_ind.append(float(relH[30:36]))
                start_ind_T.append(float(relH[22:28]))
                start_ind_KM.append(float(relH[4:11]))
                k += 1
                relH = lines[i+3+k]
                RH_line = relH[30:33]
                RH_line = RH_line.replace(" ","")
            
    relH = np.asarray(start_ind)
    T_profile = np.asarray(start_ind_T)
    KM_profile = np.asarray(start_ind_KM)
    RH = max(relH)
    #print('RH: ' + str(max(relH)))

    word = 'GM / CM2'
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            CWV_line = lines[i+2]
            CWV = float(CWV_line[16:24])
            break
    
    word = 'USER INPUT DATA:'
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            skintemp_line = lines[i+1]
            skintemp = float(skintemp_line[21:31])
            break
    
    
    
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
    tape6['CWV'] = np.float64(CWV)
    tape6['RH'] = RH
    tape6['skintemp'] = skintemp

    idx = np.argsort(tape6['wavelength'])
    tape6['wavelength'] = tape6['wavelength'][idx]
    tape6['path_thermal'] = tape6['path_thermal'][idx]
    tape6['ground_reflected'] = tape6['ground_reflected'][idx]
    tape6['transmission'] = tape6['transmission'][idx]
    
    tape6['dewpoint'] = T_profile - ((100-relH)/5)
    tape6['T_profile'] = T_profile
    tape6['KM_profile'] = KM_profile
    tape6['RH_profile'] = relH

    return tape6

##################################### calc toa radiance ###############################################

def load_RSR(center, FWHM, sensor='TIRS10'):
    
    # load RSR of specified sensor
    
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
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
    elif sensor == 'ASTER13':
        filename = filepath + 'ASTER_13.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'ASTER14':
        filename = filepath + 'ASTER_14.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'DRS4':
        filename = filepath + 'muri_band_response_curves.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,4]
    elif sensor == 'DRS5':
        filename = filepath + 'muri_band_response_curves.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,5]
    elif sensor == 'DRS6':
        filename = filepath + 'muri_band_response_curves.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,6]
    elif sensor == 'ECOSTRESS4':
        filename = filepath + 'ECOSTRESS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,4]
    elif sensor == 'ECOSTRESS5':
        filename = filepath + 'ECOSTRESS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,5]
    elif sensor == 'GAUSS':
        wave = np.arange(7.9,14.1,0.01)
        rsr = norm.pdf(wave,center,FWHM/2.3548)
        rsr = rsr/max(rsr)
        #plt.plot(wave,rsr)
    elif sensor == 'RECT':
        wave = np.arange(7.9,14.1,0.01)
        rsr = np.zeros(wave.shape)
        rsr = np.where((wave > center - FWHM/2) & (wave < center + FWHM/2),1,0)
        #plt.plot(wave,rsr)
    
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
        

def inverse_planck(radiance_spectral,wavelength):
    
    wvl = wavelength #* 1e-6

    L = radiance_spectral * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    Temp = (2 * h * c * c) / (L * (wvl**5))
    Temp2 = np.log(Temp+1)
    appTemp_spectral = (h * c )/ (k * wvl *Temp2)
    
    return appTemp_spectral

def inverse_planck_center(radiance_spectral,wave_center):
    

    wvl = wave_center * 1e-6

    L = radiance_spectral * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    Temp = (2 * h * c * c) / (L * (wvl**5))
    Temp2 = np.log(Temp+1)
    appTemp = (h * c )/ (k * wvl *Temp2)
    
    return appTemp


def calc_TOA_radiance2(tape6,RSR_interp,T,emis,wavelength):
    # calculate TOA radiance
    h = 6.626 * 10**(-34)
    c = 2.998 * 10**8
    k = 1.381 * 10**(-23)

    surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (tape6['wavelength']**(5) * (np.exp((h*c) / (tape6['wavelength'] * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
    
    appTemp_spectral = inverse_planck(radiance_spectral,wavelength)
    appTemp = np.trapz(list(RSR_interp * appTemp_spectral),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR_interp),x=list(tape6['wavelength']),axis=0)
       
    radiance = np.trapz(list(RSR_interp * radiance_spectral),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR_interp),x=list(tape6['wavelength']),axis=0)
    
    trans  = np.trapz(list(RSR_interp * tape6['transmission']),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR_interp),x=list(tape6['wavelength']),axis=0)
    
    print('Rad, trans and app temp calculated.')
    return radiance, appTemp,trans


######################################### get emissivity files ############################################
    
def emis_interp(wavelength,RSR_interp, num = 0):
    
    if num != 1:
    
        emis_spectral = []
        emis = []
        
        filename = '/cis/staff2/tkpci/emissivities/SoilsMineralsParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
#            plt.plot(wavelength,temp)
#            plt.title('UCSB Soil/Mineral Emissivities')
#            plt.ylim([0.5, 1])
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        
        filename = '/cis/staff2/tkpci/emissivities/VegetationParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            #plt.plot(wave,temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
    
        filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            #plt.plot(wave,temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        # filename = '/cis/staff2/tkpci/emissivities/rock_ems_v2_10.csv'
        # data = np.genfromtxt(filename, delimiter=',')
        # wave = data[:,0]
        
        # for i in range(data.shape[1]-1):
        #     temp = np.interp(wavelength, wave, data[:,i+1])
        #     emis_spectral.append(temp)
        #     #plt.plot(wavelength,temp)
        #     emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
        
    else:
        
        emis_spectral = []
        emis = []
        
        filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities_1.csv'
        data = np.genfromtxt(filename, delimiter=',')
        data = data[:,0:2]
        wave = data[:,0]
    
        
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
    
    return emis, emis_spectral


############################################## calc app temp for B10 and B11

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
    
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)));
    
    return temperature



def LUT_radiance_apptemp(radiance, band='b10'):
    
    if band == 'b10':
        LUT = np.loadtxt('LUT_TIRS10.csv', delimiter=',')
    elif band == 'b11':
        LUT = np.loadtxt('LUT_TIRS11.csv', delimiter=',')
        
    rad = np.tile(radiance, (LUT.shape[0],1))
    LUT_rad = np.tile(LUT[:,0], (radiance.shape[0],1))
    LUT_rad = LUT_rad.T
    
    A = np.abs(LUT_rad-rad)
    A = np.matrix(A)
    idx = A.argmin(0)
   
    T = LUT[idx[0,:],1]
    T = np.squeeze(T)

    return T    
   

def calc_SW_and_save(dataOut):  

    # calc appTemp using TIRS coefficients in MTL file
    T10=  np.array(dataOut['T10'])
    T11 =  np.array(dataOut['T11'])
       
    e10 = np.array(dataOut['emis10'])
    e11 = np.array(dataOut['emis11'])
    
    y = np.array(dataOut['skintemp'])
    
    T10 = T10.flatten() 
    T11 = T11.flatten() 
    e10 = e10.flatten() 
    e11 = e11.flatten() 
    
    T_diff = (T10 - T11)/2
    T_plus =  (T10 + T11)/2
    e_mean = (e10 + e11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (e10-e11)/(e_mean**2)
    quad = (T10-T11)**2
    
    b0 = np.ones(T_diff.shape)

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
        
    x = np.array(x)
    
    
    # test data with normal coefficients calculated by 
    coeff = [2.2925,0.9929,0.1545,-0.3122,3.7186,0.3502,-3.5889,0.1825] # TIRS 1386 * 113 * 7
    
    LST_SW = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    
    dataOut['SW_LST'] = LST_SW
    
    diff = y - LST_SW
    dataOut['diff'] = diff
            
    
    
    #headers = ",".join(dataOut.keys())   
    
    
    # write data to txt file
    filename_out = '/dirs/data/tirs/tania-dir/SW_uncertainty_atm_error_231100.csv'
    
    # if not os.path.isfile(filename_out):
        
    #     with open(filename_out, mode='w') as file_out:
    #         csv.excel.delimiter=';'
    #         file_writer = csv.writer(file_out, dialect=csv.excel)
        
    #         file_writer.writerow([headers])
            
    
    # for rows in range(dataOut['rad10'].shape[0]): 
    #     temp = []
    #     for key in dataOut.keys():
    #         temp.append(dataOut[key][0])
    #     values = ",".join(str(e) for e in temp) 
        
    #     with open(filename_out, mode='a') as file_out:
    #         csv.excel.delimiter=';'
    #         file_writer = csv.writer(file_out, dialect=csv.excel)
        
    #         file_writer.writerow([values])
    
    del dataOut['tape6']        
    df = pd.DataFrame(dataOut)
    df.to_csv(filename_out) 
    
    return dataOut
                    
    
#main_processes()   
        




   
