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
import SW_regression
import pandas as pd


def calc_rad_from_tape6():
    
    # recreate TIGR tape6 incase they have been overwritten
    
    
    
    
    # run tape6 once to get wavelength that RSR and emis can be done outside of loop
    tape6 = read_tape6('tape6_118',filepath='/cis/staff/tkpci/modtran/tape6/')
    wavelength_orig = tape6['wavelength']
    
        
    #RSR to use for analysis
    RSR10 = load_RSR(sensor='TIRS2_10',center=11.05,FWHM=0.59)    #10.9   #eon 10.8
    RSR11 = load_RSR(sensor='TIRS2_11',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
    
    #RSR10 = load_RSR(sensor='MOD31',center=11.05,FWHM=0.59)    #10.9   #eon 10.8
    #RSR11 = load_RSR(sensor='MOD32',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
    
    # interpolate to wavelength
    RSR10_interp_orig = interp_RSR(wavelength_orig, RSR10, toplot = 1)
    RSR11_interp_orig = interp_RSR(wavelength_orig, RSR11, toplot = 1)
    
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
    
    cntr = 1
    
    # reduced_TIGR = np.genfromtxt('tigr_200_profiles_index.csv', delimiter=',')
    # reduced_TIGR = reduced_TIGR.astype(int)
    
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
    #reduced set
    #T = np.asarray([-5,0,5,10,15]) # all 7 iterations
    T_num = T.shape[0]
    #wave_num = wavelength_orig.shape[0]
    
    # just to get number of emis files
    emis10, emis_spectral = emis_interp(wavelength_orig,RSR10_interp_orig, num=0)
    
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
    
    dataOut['CWV'] = CWV 
    dataOut['RH'] = RH

            
    dataOut['rad10'] = []
    dataOut['rad11'] = []
    dataOut['emis10'] = []
    dataOut['emis11'] = []
    dataOut['T10'] = []
    dataOut['T11'] = []

          
    RSR10_interp = np.repeat(RSR10_interp_orig[:,np.newaxis],wavelength.shape[1],axis=1)
    RSR11_interp = np.repeat(RSR11_interp_orig[:,np.newaxis],wavelength.shape[1],axis=1)
               
    # get UCSB emissivity files and interpolate to wavelenght
    emis10, emis_spectral10 = emis_interp(wavelength_orig,RSR10_interp_orig, num=0)
    emis11, emis_spectral11 = emis_interp(wavelength_orig,RSR11_interp_orig, num=0)
    
    emis_spectral10 = np.transpose(emis_spectral10)
    emis_spectral11 = np.transpose(emis_spectral11)
    
    emis_spectral10 = np.tile(emis_spectral10,T_num*files_num)
    emis_spectral11 = np.tile(emis_spectral11,T_num*files_num)
    
    dataOut['emis10'] = np.tile(emis10,T_num*files_num)
    dataOut['emis11'] = np.tile(emis11,T_num*files_num)
    
    #TIRS2_10  = calc_TOA_radiance(tape6_vector,RSR10_interp,T,emis_spectral10)
    #TIRS2_11 = calc_TOA_radiance(tape6_vector,RSR11_interp,T,emis_spectral11)

    dataOut['rad10']  = calc_TOA_radiance(tape6_vector,RSR10_interp,T,emis_spectral10)
    dataOut['rad11'] = calc_TOA_radiance(tape6_vector,RSR11_interp,T,emis_spectral11)
    
    

    
    dataOut['T10'] = LUT_radiance_apptemp(rad10, band='TIRS2_b10')
    dataOut['T11'] = LUT_radiance_apptemp(rad11, band='TIRS2_b11')
    
    #dataOut['T10'] = appTemp(rad10, band='b10')
    #dataOut['T11'] = appTemp(rad11, band='b11')
        
    coeff =  calc_SW_coeff (dataOut)    
             
    return coeff, dataOut, rad10,rad11


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
    elif sensor == 'TIRS2_10':
        filename = filepath + 'TIRS2.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
        
    elif sensor == 'TIRS2_11':
        filename = filepath + 'TIRS2.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,2]
        
    elif sensor == 'MOD31':
        filename = filepath + 'MODIS_31.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
        
    elif sensor == 'MOD32':
        filename = filepath + 'MODIS_32.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
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

def LUT_radiance_apptemp(radiance, band='b10'):
    
    if band == 'b10':
        LUT = np.loadtxt('LUT_TIRS10.csv', delimiter=',')
    elif band == 'b11':
        LUT = np.loadtxt('LUT_TIRS11.csv', delimiter=',')
    elif band == 'TIRS2_b10':
        LUT = np.loadtxt('LUT_TIRS2_10.csv', delimiter=',')
    elif band == 'TIRS2_b11':
        LUT = np.loadtxt('LUT_TIRS2_11.csv', delimiter=',')
        
    rad = np.tile(radiance, (LUT.shape[0],1))
    LUT_rad = np.tile(LUT[:,0], (radiance.shape[0],1))
    LUT_rad = LUT_rad.T
    
    A = np.abs(LUT_rad-rad)
    A = np.matrix(A)
    idx = A.argmin(0)
   
    T = LUT[idx[0,:],1]
    T = np.squeeze(T)

    return T  

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

######################################### get emissivity files ############################################
    
def emis_interp(wavelength,RSR_interp, num = 0):
    
    if num != 1:
    
        filename = '/cis/staff/tkpci/emissivities/SoilsMineralsParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        
        emis_spectral = []
        emis = []
        
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        
        filename = '/cis/staff/tkpci/emissivities/VegetationParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
    
        filename = '/cis/staff/tkpci/emissivities/WaterIceSnowParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
            
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
            
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
    
    return emis, emis_spectral


############################################## calc app temp for B10 and B11

# def appTemp(radiance, band='b10'):
    
#     if band == 'b10':
#         K2 = 1321.0789;
#         K1 = 774.8853;
#     elif band == 'b11':
#         K2 = 1201.1442;
#         K1 = 480.8883;
#     else:
#         print('call function with T = appTemp(radiance, band=\'b10\'')    
#         return
    
#     temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)));
    
#     return temperature


def calc_SW_coeff(dataOut):
    
    
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
    
    x = x.T
    LST = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:] + coeff[3]*x[3,:] + coeff[4]*x[4,:] + coeff[5]*x[5,:] + coeff[6]*x[6,:] + coeff[7]*x[7,:]
    y = dataOut['skintemp']
    rmse = np.sqrt(np.mean((LST - y)**2))
    print('RMSE: ',rmse)
    std = np.std(LST-y)
    print('STD: ', std)
    diff = np.mean(LST-y)
    
    return coeff


#coeff, dataOut, rad10, rad11 = calc_rad_from_tape6()
