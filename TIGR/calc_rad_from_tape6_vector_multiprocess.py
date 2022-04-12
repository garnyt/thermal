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

warnings.filterwarnings("ignore")


def calc_rad_from_tape6():
    
    startTime = datetime.now()
    
    
    path = '/cis/staff/tkpci/Code/Python/TIGR/multiprocess_out/'
    files = os.listdir(path)
    
    for f in files:
        try:
            os.remove(path + f)
        except:
            print('File '+f+ ' not removed' )    
    
    
    # run tape6 once to get wavelength that RSR and emis can be done outside of loop
    tape6 = read_tape6('tape6_118',filepath='/cis/staff/tkpci/modtran/tape6/')
    wavelength_orig = tape6['wavelength']
    
    
    # download RSR for specific sensor - nominal
    #RSR10 = load_RSR(sensor='TIRS10')
    #RSR11 = load_RSR(sensor='TIRS11')
    
    #standard TIRS bands
    center10 = 10.9
    FWHM10 = 0.59
    center11 = 12
    FWHM11 = 1.01
    
    # ASTER bands 13 and 14
    # center10 = 10.6
    # FWHM10 = 0.7
    # center11 = 11.3
    # FWHM11 = 0.7
    
    #DRS bands 5 an 6
    #center10 = 10.923
    #center11 = 11.638

    
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
    
    cntr = 1
    
    reduced_TIGR = np.genfromtxt('tigr_200_profiles_index.csv', delimiter=',')    
    reduced_TIGR = reduced_TIGR.astype(int)
    
    #uncomment for all 2311 profiles
    reduced_TIGR = np.arange(2311)
    
    old_tape6 = tape6
    #for counter in range(2):    
    #    counter += 1
    for counter in reduced_TIGR[1:]:
        cntr +=1
        
        filename = 'tape6_' + str(counter+1)  # reduced_TIGR starts at 0 and tape6 files at 1
        tape6 = read_tape6(filename , filepath='/cis/staff/tkpci/modtran/tape6/')
        
        if tape6['RH']< 90:

            new_tape6 = {**old_tape6, **tape6}
            for key, value in new_tape6.items():
               if key in old_tape6 and key in tape6:
                   if isinstance(tape6[key],float):
                       new_tape6[key] = [np.hstack([value , np.squeeze(np.asarray(old_tape6[key]))])]
                   else:
                       new_tape6[key] = [np.vstack([value , np.squeeze(np.asarray(old_tape6[key]))])]
            old_tape6 = new_tape6
           
    tape6 = new_tape6            
        
    # calculate radiance for both bands
    T = np.asarray([-10,-5,0,5,10,15,20]) # all 7 iterations
    T_num = T.shape[0]
    #wave_num = wavelength_orig.shape[0]
    
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

    
    CWV = np.squeeze(np.asarray(tape6['CWV']))
    CWV = np.repeat(CWV,7*emis_num,axis=0)
    RH = np.squeeze(np.asarray(tape6['RH']))
    RH = np.repeat(RH,7*emis_num,axis=0)
    
    dataOut['CWV'] = CWV 
    dataOut['RH'] = RH

    # loop throught options if needed, else make move = 0
    #output = {} 
    
    # add nosie to emis
#    noise10 = np.random.uniform(low=0, high=0.04, size=7*200*30)-0.02
#    #noise11 = noise10
#    noise11 = np.random.uniform(low=0, high=0.04, size=7*200*30)-0.02

    # no noise added
    #noise10 = 0
    #noise11 = 0
    
    #move = np.arange(-0.5,0.6,0.1)
    #move = np.arange(-0.5,0.6,0.1)+0.2  ## if you have center at -2 and -2
    #move = np.arange(-1.25,-0.15,0.1)  ## if you have TIRS center at 10.15 and 11.35
    
    #uncomment for no iterations
    move = np.arange(0,1,1)
   
    move = np.round(np.where(abs(move) < 0.001,0,move),2)
    move = list(move)
    
    total = len(move)**2
    
    loops = np.zeros([total,2])
    loops[:,0] = np.tile(move,len(move))
    loops[:,1] = np.repeat(move,len(move), axis=0)
    
    myProcesses = []
    
    output={}
    output['band10'] = [] 
    output['band11'] = []
    output['rmse_new_coeff'] = []
    output['stddev_new_coeff'] = []
    output['mean_new_coeff'] = []
    output['rmse_orig_coeff'] = []
    output['stddev_orig_coeff'] = []
    output['mean_orig_coeff'] = []
    output['rmse_new_coeff_error'] = []
    output['stddev_new_coeff_error'] = []
    output['mean_new_coeff_error'] = []
    output['rmse_orig_coeff_error'] = []
    output['stddev_orig_coeff_error'] = []
    output['mean_orig_coeff_error'] = []
    
    for val in range(loops.shape[0]):        
        myProcesses.append(Process(target=loop_through_data, args=(loops, val, output, dataOut,RSR10,RSR11,wavelength_orig,wavelength,T_num,files_num,tape6_vector,T,center10,center11,)))

        
    print(str(val+1) + " instances created")
    #counter1 = 0
    
    cores = 20
    iter = int(loops.shape[0]/cores)
    print("Running " + str(cores) + " processes at a time")
    
    for i in range(iter+1):
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > loops.shape[0]:
            end_cnt = loops.shape[0]
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
        
    
#    for process in myProcesses[counter1:counter1+40]:
#        counter1 += 1
#        process.start()
#    
#    print(str(counter1) + " instances started ")
#        
#        
#    for process in myProcesses[counter2:counter2+40]:
#        counter2 += 1
#        #print("joining instance " + str(counter))
#        process.join()
#        
#    
#    for process in myProcesses[counter1:]:
#        counter1 += 1
#        process.start()
#    
#    print(str(counter1) + " instances started ")
#        
#        
#    for process in myProcesses[counter2:]:
#        counter2 += 1
#        #print("joining instance " + str(counter))
#        process.join()

    print('\nTime elasped: ', datetime.now() - startTime)
    return dataOut
    

     
########################################

def loop_through_data(loops, val, output, dataOut,RSR10,RSR11,wavelength_orig,wavelength,T_num,files_num,tape6_vector,T,center10,center11):
    
    
    k = loops[val,0]
    j = loops[val,1]
    
    move10 = k
    move11 = j
    
    output['band10'].append(k)
    output['band11'].append(j)
                            
    shift10 = k
    shift11 = j
    widen10 = 0
    widen11 = 0
    
    # shift10 = 0
    # shift11 = 0
    # widen10 = k
    # widen11 = j

    
    dataOut['rad10'] = []
    dataOut['rad11'] = []
    dataOut['emis10'] = []
    dataOut['emis11'] = []
    dataOut['T10'] = []
    dataOut['T11'] = []

    
    # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
#    RSR10_interp_new = shift_widen_bands(RSR10_interp_orig, wavelength_orig, shift10, widen10)
#    RSR11_interp_new = shift_widen_bands(RSR11_interp_orig, wavelength_orig, shift11, widen11)
    RSR10_new = {}
    RSR10_new['wave'] = RSR10['wave']
    RSR11_new = {}
    RSR11_new['wave'] = RSR11['wave']
    
    RSR10_new['rsr'] = shift_widen_bands(RSR10['rsr'], RSR10['wave'], shift10, widen10)
    RSR11_new['rsr'] = shift_widen_bands(RSR11['rsr'], RSR11['wave'], shift11, widen11)
        
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
    dataOut['T10_coeff'] = appTemp(rad10, band='b10')
    dataOut['T11_coeff'] = appTemp(rad11, band='b11')

    
    # calc appTemp from TIRS band centers lookup table
    dataOut['T10_LUT'] = LUT_radiance_apptemp(rad10, band='b10')
    dataOut['T11_LUT'] = LUT_radiance_apptemp(rad11, band='b11')
    
    
    # calc appTemp with planck using only center wavelength    
    dataOut['T10_planck_center_fixed'] = inverse_planck_center(rad10,center10)
    dataOut['T11_planck_center_fixed'] = inverse_planck_center(rad11,center11)
    
    # calc appTemp with planck using only center wavelength    
    dataOut['T10_planck_center_new'] = inverse_planck_center(rad10,center10+k)
    dataOut['T11_planck_center_new'] = inverse_planck_center(rad11,center11+j)
    
    
    # calc appTemp with planck spectrally before trapz
    dataOut['T10_planck_spectral'] = T10_planck
    dataOut['T11_planck_spectral'] = T11_planck
    
    
#    SW_regression_multiprocess.calc_SW_coeff(dataOut, output)
    calc_SW_coeff(dataOut, output, move10,move11)


###########################################shift or widen bands #########################################
    
def shift_widen_bands(RSR_interp, wavelength, shift = 0, widen = 0):
    
    new_RSR_interp = RSR_interp
    
    if (shift == 0) and (widen == 0):
        new_RSR_interp = RSR_interp
    else:
        min_wave = min(wavelength)
        max_wave = max(wavelength)
        size_wave = wavelength.shape[0]
        delta_wave = (max_wave - min_wave)/(size_wave)
        
        if shift != 0:
            to_move = int(shift/delta_wave)
            to_move = -to_move
            new_RSR_interp = np.hstack([RSR_interp[to_move:],RSR_interp[:to_move]])
           
        if widen != 0:
            RSR_interp = new_RSR_interp
            rsr = np.where(RSR_interp >0.01,1,0)
            rsr_wave = wavelength*rsr
            minval = np.min(rsr_wave[np.nonzero(rsr_wave)])
            maxval = np.max(rsr_wave[np.nonzero(rsr_wave)])
            #centerval = (maxval-minval)/2+minval
            
            min_idx = np.where(wavelength == minval)
            max_idx = np.where(wavelength == maxval)
            
            new_RSR = np.hstack([0,RSR_interp[min_idx[0][0]:max_idx[0][0]+1],0])
            
            cnt = sum(rsr)+2
            #to_widen = int((widen/delta_wave))
            new_min = minval - widen/2
            new_max = maxval + widen/2
            step = (new_max - new_min)/cnt
            wave = np.arange(new_min,new_max,step)
            diff = wave.shape[0] - new_RSR.shape[0]
            if diff > 0:
                 new_RSR = np.hstack([new_RSR,0])
            if diff < 0:
                new_RSR = new_RSR[:diff]
                
            new_RSR_interp = np.interp(wavelength, wave, new_RSR)
        
    
    return new_RSR_interp



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
    
    #transmission = tape6['transmission']
    
    trans  = np.trapz(list(RSR_interp * tape6['transmission']),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR_interp),x=list(tape6['wavelength']),axis=0)
    
    
    return radiance, appTemp,trans


######################################### get emissivity files ############################################
    
def emis_interp(wavelength,RSR_interp, num = 0):
    
    if num != 1:
    
        emis_spectral = []
        emis = []
        
        filename = '/cis/staff2/tkpci/emissivities/SoilsMineralsParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
#            plt.plot(wavelength,temp)
#            plt.title('UCSB Soil/Mineral Emissivities')
#            plt.ylim([0.5, 1])
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        
        filename = '/cis/staff2/tkpci/emissivities/VegetationParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            #plt.plot(wave,temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
    
        filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities.csv'
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



def calc_SW_coeff(dataOut, output, move10, move11):
    
    # calc appTemp using TIRS coefficients in MTL file
    T10_coeff =  np.array(dataOut['T10_coeff'])
    T11_coeff =  np.array(dataOut['T11_coeff'])
    
    # calc appTemp from TIRS band center LUT
    T10_LUT =  np.array(dataOut['T10_LUT'])
    T11_LUT =  np.array(dataOut['T11_LUT'])
    
    # calc appTemp using planck for center wavelenght fixed for orig coeff
    T10_planck_center_fixed =  np.array(dataOut['T10_planck_center_fixed'])
    T11_planck_center_fixed =  np.array(dataOut['T11_planck_center_fixed'])
    
    # calc appTemp using planck for center wavelenght for each new iteration of changed center
    T10_planck_center_new =  np.array(dataOut['T10_planck_center_new'])
    T11_planck_center_new =  np.array(dataOut['T11_planck_center_new'])
    
    # calc appTemp using planck spectrally before trapz
    T10_planck_spectral =  np.array(dataOut['T10_planck_spectral'])
    T11_planck_spectral =  np.array(dataOut['T11_planck_spectral'])
    
    
    # assign which appTemp to use
    T10_orig = T10_planck_center_fixed
    T11_orig = T11_planck_center_fixed
    T10_new = T10_planck_spectral
    T11_new = T11_planck_spectral
    
    e10 = np.array(dataOut['emis10'])
    e11 = np.array(dataOut['emis11'])
    
    y = np.array(dataOut['skintemp'])
    
    coeff_orig, x_orig = calc_SW_coeff_only(e10,e11,T10_orig,T11_orig,y)
    coeff_new, x_new= calc_SW_coeff_only(e10,e11,T10_new,T11_new,y)
       
    rmse = test_coeff(coeff_orig,coeff_new, x_orig,x_new , y, output, move10, move11,T10_orig, T11_orig,T10_new,T11_new,e10,e11)
    
    return rmse


def calc_SW_coeff_only(e10,e11,T10,T11,y):
    
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
        
    x = np.array(x).T
    
    coeff, residuals, rank, s = np.linalg.lstsq(x,y,rcond=None)
    
    return coeff, x
    

def test_coeff(coeff_orig,coeff_new, x_orig,x_new, y, output, move10, move11,T10_orig,T11_orig,T10_new,T11_new,e10,e11):
    
    # calcualate new coefficients and rmse
    x_new = x_new.T
    LST_new = coeff_new[0] + coeff_new[1]*x_new[1,:] + coeff_new[2]*x_new[2,:] + coeff_new[3]*x_new[3,:] + coeff_new[4]*x_new[4,:] + coeff_new[5]*x_new[5,:] + coeff_new[6]*x_new[6,:] + coeff_new[7]*x_new[7,:]
    diff_new = LST_new-y
    rmse_new = np.sqrt(np.mean((LST_new - y)**2))
    
    output['rmse_new_coeff'].append(round(rmse_new,2))
    output['stddev_new_coeff'].append(round(np.std(diff_new),2))
    output['mean_new_coeff'].append(round(np.average(diff_new),2))
    
    if ((move10 == 0) & (move11 == 0)):
        np.set_printoptions(suppress=True)
        print(np.round(coeff_new,5))
        print(rmse_new)
    
    error_new = calculateError(coeff_new, rmse_new, T10_new,T11_new,e10,e11)
    rmse_error_new = np.sqrt(np.mean((error_new)**2))
    
    output['rmse_new_coeff_error'].append(round(rmse_error_new,2))
    output['stddev_new_coeff_error'].append(round(np.std(diff_new),2))
    output['mean_new_coeff_error'].append(round(np.average(diff_new),2))
    
    

    
    # test data with normal coefficients calculated by 
    #coeff = [2.2925,0.9929,0.1545,-0.3122,3.7186,0.3502,-3.5889,0.1825] # TIRS 1386 * 113 * 7
    #coeff = [-1.3628,	1.0086,	0.1539,	-0.1168,	2.0938,	-4.4729,	-21.8117	,0.2949] # TIRS 1386 * 133 * 7 (with rocks)
    coeff = [ 2.61019,0.99143 ,0.16112,-0.3189,3.79615,-0.66482,-4.52396,0.15421] # python with 7 * 200 * 30 - GAUSS
    #coeff = [1.38512,1.00025,0.14679,-0.34823,4.47638,-1.23695,-11.14722,0.21139] # python with 7 * 200 * 30 - TIRS with -0.2 shift
    #coeff = [1.42337,0.99387,0.17248,-0.2815, 3.13621,0.23197,-3.85478 ,0.14456]# python with 7 * 200 * 30 - TIRS with 0.2 shift
    #coeff = [ 1.0202 ,  0.99821,  0.15724 ,-0.30154  ,3.66915 ,-1.14584 ,-8.48704 , 0.21967] # DRS 7 * 30 * 1386

    x = x_orig.T
    LST_orig = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    
    diff2 = LST_orig-y
    rmse2 = np.sqrt(np.mean((LST_orig - y)**2))
    output['rmse_orig_coeff'].append(round(rmse2,2))
    output['stddev_orig_coeff'].append(round(np.std(diff2),2))
    output['mean_orig_coeff'].append(round(np.average(diff2),2))    
    
    error2 = calculateError(coeff, rmse2, T10_orig,T11_orig,e10,e11)
    rmse2 = np.sqrt(np.mean((error2)**2))

   
    output['rmse_orig_coeff_error'].append(round(rmse2,2))
    output['stddev_orig_coeff_error'].append(round(np.std(diff2),2))
    output['mean_orig_coeff_error'].append(round(np.average(diff2),2))
 
        
    filename = 'band10_band11_' + str(move10) + '_' + str(move11)
        
    df = pd.DataFrame(output)
    df.to_csv('/cis/staff2/tkpci/Code/Python/TIGR/multiprocess_out/'+filename+'.csv') 
        
    return rmse_new


def calculateError(SW_coeff, rmse,T10,T11,e10,e11):
    
  
    
    # error values for regression to get to coefficients
    b_total_error = rmse
    
    b_coeff= SW_coeff
    #b0 = b_coeff[0]
    b1 = b_coeff[1]
    b2 = b_coeff[2]
    b3 = b_coeff[3]
    b4 = b_coeff[4]
    b5 = b_coeff[5]
    b6 = b_coeff[6]
    b7 = b_coeff[7]

   
    # correlation coefficient for apparent temperature and emissivity
    # this was calculated using the 113 MODIS emissivities (spectrally sampled)
    # the appTemp correlation was calculated using the TIGR simulation data
    corr_emis = 0.7
    corr_appTemp = 1
       
    # uncertainty in apparent temperature - hardcoded according to discussion with Matt   
    T10_error = 0 #0.15
    T11_error = 0 #0.2
    

    #check if rock emis is part of analysis
    #emissivity uncertainty calculation (adding error in quadrature)
    if e10.shape[0] == 56000:
        # reading in TES error from Eon data - 2020-06-23        
        filename = '/cis/staff2/tkpci/emissivities/B10_ems_pd2_error_rock.csv'
    elif e10.shape[0] == 42000:
        # reading in TES error from Eon data - 2020-06-01        
        filename = '/cis/staff2/tkpci/emissivities/B10_ems_pd_error.csv'
        
    e10_error = np.genfromtxt(filename, delimiter=',')
    e10_error = e10_error.flatten()/100            # this is in %
    e10_error = np.repeat(e10_error,7,axis=0)
    #e10_error = np.tile(e10_error,7)
    
    if e11.shape[0] == 56000:
        # reading in TES error from Eon data - 2020-06-23        
        filename = '/cis/staff2/tkpci/emissivities/B11_ems_pd2_error_rock.csv'
    elif e11.shape[0] == 42000:
        # reading in TES error from Eon data - 2020-06-01        
        filename = '/cis/staff2/tkpci/emissivities/B11_ems_pd_error.csv'
    

    e11_error = np.genfromtxt(filename, delimiter=',') 
    e11_error = e11_error.flatten()/100            # this is in %
    e11_error = np.repeat(e11_error,7,axis=0)   
    #e11_error = np.tile(e11_error,7)    

       
    # partial derivatives of SW algorithm    
    T10diff = b1/2 + b2*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b3*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b4/2 + b5*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b6*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b7*(2*T10 - 2*T11)
    T11diff = b1/2 + b2*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) + b3*(e10 - e11)/(2*(e10/2 + e11/2)**2) - b4/2 - b5*(-e10/2 - e11/2 + 1)/(2*(e10/2 + e11/2)) - b6*(e10 - e11)/(2*(e10/2 + e11/2)**2) + b7*(-2*T10 + 2*T11)
    E10diff = -b2*(T10 + T11)/(4*(e10/2 + e11/2)) - b2*(T10 + T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) + b3*(T10 + T11)/(2*(e10/2 + e11/2)**2) - b3*(T10 + T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3) - b5*(T10 - T11)/(4*(e10/2 + e11/2)) - b5*(T10 - T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) + b6*(T10 - T11)/(2*(e10/2 + e11/2)**2) - b6*(T10 - T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3)
    E11diff = -b2*(T10 + T11)/(4*(e10/2 + e11/2)) - b2*(T10 + T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) - b3*(T10 + T11)/(2*(e10/2 + e11/2)**2) - b3*(T10 + T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3) - b5*(T10 - T11)/(4*(e10/2 + e11/2)) - b5*(T10 - T11)*(-e10/2 - e11/2 + 1)/(4*(e10/2 + e11/2)**2) - b6*(T10 - T11)/(2*(e10/2 + e11/2)**2) - b6*(T10 - T11)*(e10 - e11)/(2*(e10/2 + e11/2)**3)
    
    
    # SW uncertainty in quadrature with and without correlation
    LST_error = np.sqrt(b_total_error**2 + (T10diff*T10_error)**2 + (T11diff*T11_error)**2 + (E10diff*e10_error)**2 + (E11diff*e11_error)**2 + \
                        2*corr_appTemp*T10diff*T11diff*T10_error*T11_error + 2*corr_emis*E10diff*E11diff*e10_error*e11_error) 
    
    
    return LST_error
   

# def plot_RSR():
    
#     tape6 = read_tape6('tape6_1214',filepath='/cis/staff2/tkpci/modtran/tape6/')
#     wavelength = tape6['wavelength']
#     trans = tape6['transmission']
#     trans = trans/np.max(trans)
#     # download RSR for specific sensor - nominal
    
#     center10 = 10.9
#     FWHM10 = 0.59
#     center11 = 12
#     FWHM11 = 1.01
    
#     RSR10 = load_RSR(center10, FWHM10, sensor='TIRS10')
#     RSR11 = load_RSR(center11, FWHM11, sensor='TIRS11')
#     RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
#     RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
#     plt.plot(wavelength, RSR10_interp,'k', label = 'TIRS 10')
#     plt.plot(wavelength, RSR11_interp,'k', label = 'TIRS 11')
#     plt.plot(wavelength, trans, linewidth=0.5, c='0.55')
    
#     center10 = 10.15
#     FWHM10 = 0.7
#     center11 = 11,35
#     FWHM11 = 0.7
    
#     #RSR to use for analysis
#     RSR10 = load_RSR(center10,FWHM10,sensor='GAUSS')    #10.9   #eon 10.8
#     RSR11 = load_RSR(center11,FWHM11,sensor='GAUSS')    #12.0   #eon 12.1
#     # interpolate to wavelength
    
#     shift10 = 0
#     shift11 = 0
#     widen10 = 0.11
#     widen11 = -0.31
    
#     RSR10_new = {}
#     RSR10_new['wave'] = RSR10['wave']
#     RSR11_new = {}
#     RSR11_new['wave'] = RSR11['wave']
    
    
#     # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
#     RSR10_new['rsr'] = shift_widen_bands(RSR10['rsr'], RSR10['wave'], shift10, widen10)
#     RSR11_new['rsr'] = shift_widen_bands(RSR11['rsr'], RSR11['wave'], shift11, widen11)
    
#     RSR10_interp = interp_RSR(wavelength, RSR10_new, toplot = 0)
#     RSR11_interp = interp_RSR(wavelength, RSR11_new, toplot = 0)
    
#     plt.plot(wavelength, RSR10_interp,'b--', label = 'TIRS 10')
#     plt.plot(wavelength, RSR11_interp,'r--', label = 'TIRS 11')
    
#     shift10 = 0.5
#     shift11 = 0.5
#     widen10 = 0
#     widen11 = 0
    
#     RSR10_new = {}
#     RSR10_new['wave'] = RSR10['wave']
#     RSR11_new = {}
#     RSR11_new['wave'] = RSR11['wave']
    
    
#     # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
#     RSR10_new['rsr'] = shift_widen_bands(RSR10['rsr'], RSR10['wave'], shift10, widen10)
#     RSR11_new['rsr'] = shift_widen_bands(RSR11['rsr'], RSR11['wave'], shift11, widen11)
    
#     RSR10_interp = interp_RSR(wavelength, RSR10_new, toplot = 0)
#     RSR11_interp = interp_RSR(wavelength, RSR11_new, toplot = 0)
    
# #    RSR10 = load_RSR(sensor='TIRS10',center=10.9,FWHM=0.59)    #10.9   #eon 10.8
# #    RSR11 = load_RSR(sensor='TIRS11',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
# #    # interpolate to wavelength
# #    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
# #    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
# #    # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
# #    RSR10_interp = shift_widen_bands(RSR10_interp, wavelength, shift10, widen10)
# #    RSR11_interp = shift_widen_bands(RSR11_interp, wavelength, shift11, widen11)
    
#     plt.plot(wavelength, RSR10_interp,'b--', label = 'TIRS 10')
#     plt.plot(wavelength, RSR11_interp,'r--', label = 'TIRS 11')
    
    
#     plt.ylim([0,1.05])
#     plt.title('Landsat band 10 and 11 RSRs')
#     plt.xlabel('Wavelength [um]')
#     plt.xlim([8,14])
#     #plt.legend()
#     plt.draw()
#     plt.show()


#dataOut = calc_rad_from_tape6()