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
#import SW_regression
import pandas as pd




def calc_rad_from_tape6():
    
    # run tape6 once to get wavelength that RSR and emis can be done outside of loop
    tape6 = read_tape6('tape6_118',filepath='/cis/staff/tkpci/modtran/tape6/')
    wavelength_orig = tape6['wavelength']
    
    # download RSR for specific sensor - nominal
    #RSR10 = load_RSR(sensor='TIRS10')
    #RSR11 = load_RSR(sensor='TIRS11')
    
    #RSR to use for analysis
    #RSR09 = load_RSR(sensor='GAUSS',center=9.75,FWHM=0.1) 
    RSR10 = load_RSR(sensor='TIRS10',center=10.9,FWHM=0.59)    
    RSR11 = load_RSR(sensor='TIRS11',center=12.0,FWHM=1.01)    
    # interpolate to wavelength
    #RSR09_interp_orig = interp_RSR(wavelength_orig, RSR09, toplot = 0)
    RSR10_interp_orig = interp_RSR(wavelength_orig, RSR10, toplot = 0)
    RSR11_interp_orig = interp_RSR(wavelength_orig, RSR11, toplot = 0)
    
    # create dictionary to hold data
    dataOut = {}
    #dataOut['rad09'] = []
    dataOut['T10'] = []
    dataOut['T11'] = []
    dataOut['rad10'] = []
    dataOut['rad11'] = []
    #dataOut['emis09'] = []
    dataOut['emis10'] = []
    dataOut['emis11'] = []
    dataOut['skintemp'] = []
    dataOut['emis_class'] = []
    #dataOut['CWV'] = []
    #dataOut['RH'] = []
    #dataOut['T09'] = []
    
    
    cntr = 1
    
    #reduced_TIGR = np.arange(2311)
    
    reduced_TIGR = np.genfromtxt('/cis/staff/tkpci/Code/Python/TIGR/tigr_200_profiles_index.csv', delimiter=',')    
    reduced_TIGR = reduced_TIGR.astype(int)
    
    old_tape6 = tape6

    for counter in reduced_TIGR[1:]:
        cntr +=1
        
        filename = 'tape6_' + str(counter+1)  # reduced_TIGR starts at 0 and tape6 files at 1
        tape6 = read_tape6(filename , filepath='/cis/staff/tkpci/modtran/tape6/')

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
    T = np.asarray([-5,0,5,10,15]) # all 7 iterations
    T_num = T.shape[0]
    #wave_num = wavelength_orig.shape[0]
    
    # just to get number of emis files
    emis10, emis_spectral = emis_interp(wavelength_orig,RSR10_interp_orig, num=0, sensor = 'TIRS10')
    
    emis_num = emis10.shape[0]
    
    
    
    wavelength = np.squeeze(np.asarray(tape6['wavelength'])) * 10**(-6)
    wavelength = np.transpose(wavelength)
    files_num = wavelength.shape[1]
    wavelength = np.repeat(wavelength,emis_num*T_num,axis=1)
    
    
    #  T = np.asarray([-10,-5,0,5,10,15,20])    
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
    CWV = np.repeat(CWV,T.shape[0]*emis_num,axis=0)
    RH = np.squeeze(np.asarray(tape6['RH']))
    RH = np.repeat(RH,T.shape[0]*emis_num,axis=0)
    
    
    #RSR09_interp = np.repeat(RSR09_interp_orig[:,np.newaxis],wavelength.shape[1],axis=1)
    RSR10_interp = np.repeat(RSR10_interp_orig[:,np.newaxis],wavelength.shape[1],axis=1)
    RSR11_interp = np.repeat(RSR11_interp_orig[:,np.newaxis],wavelength.shape[1],axis=1)
               
    # get UCSB emissivity files and interpolate to wavelenght
    #emis09, emis_spectral09 = emis_interp(wavelength_orig,RSR09_interp_orig, num=0)
    emis10, emis_spectral10 = emis_interp(wavelength_orig,RSR10_interp_orig, num=0, sensor = 'TIRS10')
    emis11, emis_spectral11 = emis_interp(wavelength_orig,RSR11_interp_orig, num=0, sensor = 'TIRS11')
    
    #emis_spectral09 = np.transpose(emis_spectral09)
    emis_spectral10 = np.transpose(emis_spectral10)
    emis_spectral11 = np.transpose(emis_spectral11)
    
    #emis_spectral09 = np.tile(emis_spectral09,T_num*files_num)
    emis_spectral10 = np.tile(emis_spectral10,T_num*files_num)
    emis_spectral11 = np.tile(emis_spectral11,T_num*files_num)
    
    
    #emis09 = np.tile(emis09,T_num*files_num)
    emis10 = np.tile(emis10,T_num*files_num)
    emis11 = np.tile(emis11,T_num*files_num)
    #emis_class_all = np.tile(emis_class,T_num*files_num)

    #rad09 = calc_TOA_radiance(tape6_vector,RSR09_interp,T,emis_spectral09)
    rad10 = calc_TOA_radiance(tape6_vector,RSR10_interp,T,emis_spectral10)
    rad11 = calc_TOA_radiance(tape6_vector,RSR11_interp,T,emis_spectral11)
    
    T10 = appTemp_new_coeff(rad10, band='b10')
    T11 = appTemp_new_coeff(rad11, band='b11')
    
    
    dataNN = {}
    
    #dataNN['rad09'] = rad09
    dataNN['T10'] = T10
    dataNN['T11'] = T11
    #dataNN['emis09'] = emis09
    dataNN['emis10'] = emis10
    dataNN['emis11'] = emis11
    #dataNN['emis_class'] = emis_class_all
    
    dataNN['skintemp'] = skintemp_row
    
    
    df = pd.DataFrame.from_dict(dataNN)
    
    df.to_csv('T10_11_emis10_11_skintemp_200_30_water_emis_fixed_TlessRange.csv', header=None,index=None)
             
    

###########################################read in tape6 file ###########################################
    
def read_tape6(filename,filepath='/cis/staff/tkpci/modtran/tape6/'):

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
    elif sensor == 'GAUSS':
        wave = np.arange(7.9,14.1,0.05)
        rsr = norm.pdf(wave,center,FWHM/2.3548)
        rsr = rsr/max(rsr)
        #plt.plot(wave,rsr)
    elif sensor == 'RECT':
        wave = np.arange(7.9,14.1,0.05)
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
        
def calc_TOA_radiance(tape6,RSR_interp,T,emis):
    # calculate TOA radiance
    h = 6.626 * 10**(-34);
    c = 2.998 * 10**8;
    k = 1.381 * 10**(-23);
    
   
    surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (tape6['wavelength']**(5) * (np.exp((h*c) / (tape6['wavelength'] * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
    
    radiance = np.trapz(list(RSR_interp * radiance_spectral),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR_interp),x=list(tape6['wavelength']),axis=0)
    
    return radiance


######################################### get emissivity files ############################################
    
def emis_interp(wavelength,RSR_interp, num = 0, sensor = 'TIRS10'):
    
    if num != 1:
    
        filename = '/cis/staff/tkpci/emissivities/SoilsMineralsParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        
        emis_spectral = []
        emis = []
        emis_class = []
        
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            emis_class.append(2)
        
        filename = '/cis/staff/tkpci/emissivities/VegetationParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            emis_class.append(3)
    
        filename = '/cis/staff/tkpci/emissivities/WaterIceSnowParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            if sensor == 'TIRS10':
                emis.append(0.988)
                break
            elif sensor == 'TIRS11': 
                emis.append(0.986448696154591)
                break
            else:
                emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
           
            
            emis_class.append(4)
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
        
    else:
        
        emis_spectral = []
        emis = []
        
        filename = '/cis/staff/tkpci/emissivities/WaterIceSnowParsedEmissivities_1.csv'
        data = np.genfromtxt(filename, delimiter=',')
        data = data[:,0:2]
        wave = data[:,0]
    
        
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            emis_class.append(4)
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
    
    return emis, emis_spectral


############################################## calc app temp for B10 and B11

def appTemp_new_coeff(radiance, band='b10'):
    
    # K1 and K2 constants to be found in MTL.txt file for each band
    if band == 'b10':
        K2 = 1320.6539 # new derived coefficients
        K1 = 775.1682  # new derived coefficients
    elif band == 'b11':
        K2 = 1200.8253  # new derived coefficients
        K1 = 481.1861  # new derived coefficients
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')
        return
    
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)));
    
    return temperature


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


#calc_rad_from_tape6()
