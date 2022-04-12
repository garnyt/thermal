#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:29:37 2020

@author: tkpci
"""

import numpy as np
import readTape6
import run_modtran
import read_TIGR
import getRSR
import create_tape5
import matplotlib.pyplot as plt
import pdb


def main():
    
    
    # define center wavelengths
    wave_centers = [8.28, 8.63, 9.07, 10.6, 12.05]
    
    cnt = len(wave_centers)
     
    FWHM_values = [0.354,0.310,0.396,0.410, 0.611]
    
    # max emis to start with
    e_max = np.ones(cnt) * 0.99
    
    # create wavelengths to interpolate to
    wave = np.arange(7.9,14,0.01)
    
    # create table for RSR's
    RSR = np.zeros([len(wave),cnt])
   
    # create / open RSR
    for i in range(cnt):
         RSR[:,i] = getRSR.load_RSR(wave_centers[i], wave, sensor='GAUSS', FWHM = FWHM_values[i])

    # read TIGR and write tape5_files (all 2311)
    # for -5 K uncertainty, make uncertainty_T = -5
    # for -20% uncertainty make emis -20 (for no uncertainty = 0)
    uncertainty_T = 0
    uncertainty_wv = 0
    read_TIGR.read_TIGR(uncertainty_T, uncertainty_wv) 
    
    # run modtran for those you want
    reduced_TIGR = np.genfromtxt('/cis/staff2/tkpci/Code/Python/TIGR/tigr_200_profiles_index.csv', delimiter=',')    
    reduced_TIGR = reduced_TIGR.astype(int)
    #uncomment for all 2311 profiles
    #reduced_TIGR = np.arange(2311)
  
    # read all tape6 files into one dictionary
    old_tape6 = []
    
    

    for counter in reduced_TIGR[1:]:

        tape6_name = 'tape6_' + str(counter+1)  # reduced_TIGR starts at 0 and tape6 files at 1
        tape5_name = 'tape5_' + str(counter+1) 
        # run MODTRAN for all relevant tape5 files
        run_modtran.runModtran(tape5_name, tape6_name)
        # read in tape6 file
        tape6 = readTape6.read_tape6(tape6_name , filepath='/cis/staff2/tkpci/modtran/tape6_TES/')
        
        wavelength = np.asarray(tape6['wavelength'])
        tape6['wavelength'] = wave
        for key, value in tape6.items():
            if key in ['path_thermal','ground_reflected','transmission']:
                tape6[key] = np.interp(wave, wavelength, np.asarray(tape6[key]))
               
        
        # for first tape6 file with RH < 90 = populate dictionary
        # if tape6['RH']< 90 and not(old_tape6):
        if not(old_tape6):
            old_tape6 = tape6
        
        new_tape6 = {**old_tape6, **tape6}
        for key, value in new_tape6.items():
           if key in old_tape6 and key in tape6:
               if isinstance(tape6[key],float):
                   new_tape6[key] = [np.hstack([value , np.squeeze(np.asarray(old_tape6[key]))])]
               else:
                   new_tape6[key] = [np.vstack([value , np.squeeze(np.asarray(old_tape6[key]))])]
        old_tape6 = new_tape6
            
    tape6 = new_tape6 
    
    # calculate variables for TES - Tb, Rb 
    dataTES = calcVariablesTES(tape6,RSR,wave, uncertainty_T)
    
    eb,Rb = NEM(dataTES, e_max, wave_centers, N=12)

    beta = RAT(dataTES,eb)

    tes_T, eb_max = MMD(beta, Rb, wave_centers)
    
    eb,Rb = NEM(dataTES, eb_max, wave_centers, N=1)
    
    beta = RAT(dataTES,eb)

    tes_T, eb_new = MMD(beta, Rb, wave_centers)
    
    return tes_T, eb_new, dataTES


# calculate all variables needed for TES
def calcVariablesTES(tape6,RSR,wave, uncertainty_T):
    
    # create dictionary to hold data
    dataTES = {}
    
    
    # calculate TOA radiance
    h = 6.626 * 10**(-34)
    c = 2.998 * 10**8
    k = 1.381 * 10**(-23)
    
    # find skin temperature (lowest layer temperature) of all profiles
    T = np.squeeze(np.asarray(tape6['T_profile']))[:,0] + uncertainty_T
    T_num = T.shape[0] 
    # open emissivity files
    emis = emis_open(wave)
    emis_num = emis.shape[0]
    
    # tile to get # of profiles * # of emissivities
    T = np.repeat(T[np.newaxis,:],wave.shape[0],axis=0)
    # repeat each value # of emis times - so e.g. 30 T[0], 30 x T[1]... (np.tile makes 30 copies of T
    T = np.repeat(T,emis_num,axis=1)
    #T = np.tile(T,emis.shape[0])
    
    emis = emis.T
    emis = np.tile(emis, T_num)
    
    trans = np.squeeze(np.asarray(tape6['transmission']))
    trans = np.transpose(trans)
    path = np.squeeze(np.asarray(tape6['path_thermal']))
    path = np.transpose(path)
    ground = np.squeeze(np.asarray(tape6['ground_reflected']))
    ground = np.transpose(ground)
    
    trans = np.repeat(trans,emis_num,axis=1)
    path = np.repeat(path,emis_num,axis=1)
    ground = np.repeat(ground,emis_num,axis=1)
    wavelengths = np.repeat(np.reshape(wave,(wave.shape[0],-1)),emis_num*T_num,axis=1) * 10**(-6)
    
    surfEmis = emis * trans * 2*h*c**(2) / (wavelengths**(5) * (np.exp((h*c) / (wavelengths * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + path + ground*(1-emis)  # Note - MODTRAN ground reflect already includes transmission
    
    dataTES['L'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))
    dataTES['Ld'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))
    dataTES['tau'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))
    dataTES['Lup'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))
    dataTES['T'] = np.squeeze(T[0,:])
    dataTES['emis'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))
    dataTES['Rb'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))
    dataTES['Ls'] = np.zeros((wavelengths.shape[1],RSR.shape[1]))

    
    for i in range(RSR.shape[1]):
        
        RSR_interp = RSR[:,i]
        RSR_interp = np.repeat(np.reshape(RSR_interp,(RSR_interp.shape[0],-1)),emis_num*T_num,axis=1)
        
    
        dataTES['L'][:,i] = np.trapz(list(RSR_interp * radiance_spectral),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
        
        
        # note: MODTRAN ground reflecance already includes transmission - need to remove this
        temp = []
        temp = ground/trans
        temp = np.where(np.isinf(temp),0,temp)
        
        dataTES['Ld'][:,i] = np.trapz(list(RSR_interp * temp),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
        
        dataTES['tau'][:,i] = np.trapz(list(RSR_interp * trans),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
        
        dataTES['Lup'][:,i] = np.trapz(list(RSR_interp * path),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
       
        dataTES['emis'][:,i] = np.trapz(list(RSR_interp * emis),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
        
        temp = []
        temp = surfEmis/trans
        temp = np.where(np.isnan(temp),0,temp)
        
        dataTES['Rb'][:,i] = np.trapz(list(RSR_interp * temp),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
        
        temp = []
        temp = (radiance_spectral - path)/trans
        temp = np.where(np.isinf(temp),0,temp)
        
        dataTES['Ls'][:,i] = np.trapz(list(RSR_interp * temp),x=list(wavelengths),axis=0)/np.trapz(list(RSR_interp),x=list(wavelengths),axis=0)
        
    # plot to check if values makes sense
    # for i in range(5):
    #     plt.scatter(temp[:,i],temp2[:,i])
    #     plt.scatter(T[i,:],dataTES['L'][:,i])
    
     # temp = dataTES['Ls'] - ((1-dataTES['emis'])*dataTES['Ld'])
     # temp2 = dataTES['Rb']

     # np.max(temp2-temp)
    
    return dataTES




# normalized emissivity method
def NEM(dataTES, e_max, wave_centers, N=12):
    
    Ls = dataTES['Ls']
    L = dataTES['L']
    Ld = dataTES['Ld']
    
    # threshold
    thresh = 0.05 #W/m^2.sr.um
    diff = 1
    cnt = 0
    
   
    if e_max.shape[0] < 6:
        e_max = e_max.reshape((e_max.shape[0],1))
        e_max = np.repeat(e_max, L.shape[0],axis=1)
        e_max = e_max.T
    # else:
    #     #pdb.set_trace()
    #     e_max = e_max.reshape((e_max.shape[0],1))
    #     e_max = np.repeat(e_max, len(wave_centers),axis=1)
    
    wave_centers = np.asarray(wave_centers)
    
    wave_centers = wave_centers.reshape((wave_centers.shape[0],1))
    wave_centers = np.repeat(wave_centers, L.shape[0],axis=1)
    wave_centers = wave_centers.T
       
    # calculate radiance per band  
    Rb = Ls - (1-e_max)*Ld 
    
    # estimate temperature accross all bands
    Tb = inverse_planck_center(Rb,wave_centers,e_max)
    
    # find brightest temperature
    T = np.max(Tb,axis=1)
        
    # find emissivity for each band with max temperature
    T = T.reshape((T.shape[0],1))
    T = np.repeat(T, L.shape[1],axis=1)
    Lbb = planck(T, wave_centers)  
    eb = Rb/Lbb
    final_eb = eb
    
    converge = np.ones([eb.shape[0],eb.shape[1]])
    
    # while diff > thresh or cnt < N:
    while cnt < N:
        
        cnt += 1
        
        # check for resonable emis values
        T[eb < 0.5] = 0
        eb[eb < 0.5] = 0
        T[eb > 1] = 0
        eb[eb > 1] = 0
         
        
        # calculate radiance per band  
        Rb_ = Ls - (1-eb)*Ld 
        
        # estimate temperature accross all bands
        Tb = inverse_planck_center(Rb_,wave_centers,eb)
        
        # find brightest temperature
        T = np.max(Tb,axis=1)
        
        # find emissivity for each band with max temperature
        T = T.reshape((T.shape[0],1))
        T = np.repeat(T, L.shape[1],axis=1)
                
        # find emissivity for each band with max temperatuer
        Lbb = planck(T, wave_centers)  
        eb = Rb_/Lbb
        
        #pdb.set_trace()
        # check if Rb_ converges, else stop at previous value
        diff = abs(Rb-Rb_)
        
        conv_check = np.sum(diff > converge, axis=1)
        conv_check = conv_check.reshape((conv_check.shape[0],1))
        conv_check = np.tile(conv_check, Rb.shape[1])
        
        # set new convergence values
        converge = diff
        
        mask = (conv_check < Rb.shape[1])
        
        final_eb[mask] = eb[mask]
        Rb[mask]  = Rb_[mask] 
        
    
    return final_eb, Rb


# ratio method
def RAT(dataTES,eb):
    
    L = dataTES['L']

    # beta-spectrum calculation
    eb_mean = np.mean(eb,axis=1)
    eb_mean =  eb_mean.reshape((eb_mean.shape[0],1))
    eb_mean = np.repeat(eb_mean, L.shape[1],axis=1)
    
    beta = eb/eb_mean
    
    return beta


# minimum-maximum difference
def MMD(beta, Rb, wave_centers):
    
    # lab derived constants
    a = 0.994
    b = 0.687
    c = 0.737
    
    # find spectral constrast (MMD maximum minimum difference)
    mmd = np.max(beta, axis=1) - np.min(beta, axis=1)
    
    mmd =  mmd.reshape((mmd.shape[0],1))
    mmd = np.repeat(mmd, Rb.shape[1],axis=1)
    
    # predict minimum emissivity 
    e_min = a - b * mmd**c
    
    # calculate tes emissivity
    beta_min = np.min(beta,axis=1)
    beta_min =  beta_min.reshape((beta_min.shape[0],1))
    beta_min = np.repeat(beta_min, Rb.shape[1],axis=1)
    eb = beta * e_min /beta_min
    
    
    # calcualte TES temperature using the wavelngths band with highest emissivity
    idx = np.argmax(eb, axis = 1)
    cnt = 0
    
    Rb_new = np.zeros(Rb.shape[0])
    wave_centers_new = np.zeros(Rb.shape[0])
    eb_new = np.zeros(Rb.shape[0])
    
    for i in idx:
        
        Rb_new[cnt] = Rb[cnt,i]
        eb_new[cnt] = eb[cnt,i]
        wave_centers_new[cnt] = wave_centers[i]
        
        cnt += 1
        
    tes_T = inverse_planck_center(Rb_new,wave_centers_new,eb_new)
    
    return tes_T, eb

    
def planck(Tb, wave_centers):
    
    wvl = np.asarray(wave_centers )* 1e-6 # convert to meter
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23    

    Lb = (2 * h * c**2 / wvl**5) * 1 / (np.exp((h*c)/(wvl*k*Tb))-1)*1e-6
       
    return Lb    
        
    
def inverse_planck_center(Rb,wave_centers,e_max):
    

    wvl = np.asarray(wave_centers) * 1e-6 # convert to meter

    L = Rb * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    Temp = (2 * h * c * c * e_max) / (L * (wvl**5))
    Temp2 = np.log(Temp+1)
    Tb = (h * c )/ (k * wvl *Temp2)
    
    return Tb


def emis_open(wave):
    
   
    emis_spectral = []

    
    filename = '/cis/staff2/tkpci/emissivities/SoilsMineralsParsedEmissivities_10.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave_emis = data[:,0]
    
    for i in range(data.shape[1]-1):
        temp = np.interp(wave, wave_emis, data[:,i+1])
        emis_spectral.append(temp)
        
    
    filename = '/cis/staff2/tkpci/emissivities/VegetationParsedEmissivities_10.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave_emis = data[:,0]

    for i in range(data.shape[1]-1):
        temp = np.interp(wave, wave_emis, data[:,i+1])
        emis_spectral.append(temp)
       
    filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities_10.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave_emis = data[:,0]

    for i in range(data.shape[1]-1):
        temp = np.interp(wave, wave_emis, data[:,i+1])
        emis_spectral.append(temp)
        
    # filename = '/cis/staff2/tkpci/emissivities/rock_ems_v2_10.csv'
    # data = np.genfromtxt(filename, delimiter=',')
    # wave_emis = data[:,0]
    
    # for i in range(data.shape[1]-1):
    #     temp = np.interp(wave, wave_emis, data[:,i+1])
    #     emis_spectral.append(temp)
    

    emis_spectral = np.asarray(emis_spectral)
        

    return emis_spectral

def TESanalysis(tes_T, eb_new, dataTES):
    
    T = dataTES['T']
    emis = dataTES['emis']
    
    diff_T = T- tes_T  #diff_T[np.logical_not(np.isnan(diff_T))]
    difference_array = diff_T[np.logical_not(np.isnan(diff_T))]
    squared_array = np.square(difference_array)
    
    rmse = np.sqrt(squared_array.mean())
    print('Temperature RMSE = ',rmse)
    
    diff_e = emis - eb_new  
    difference_array = diff_e[np.logical_not(np.isnan(diff_e))]
    squared_array = np.square(difference_array)
    
    rmse = np.sqrt(squared_array.mean())
       
    

# tes_T, eb_new, dataTES = main() 

# diff = tes_T - dataTES['T']
# temp = diff[~np.isnan(diff)] 
# mse = np.nanmean((tes_T - dataTES['T'])**2)
# print('Temperature mse: ', np.round(mse,3))
# plt.scatter(dataTES['T'], diff)


# for i in range(5):
#     diff = (eb_new[:,i] - dataTES['emis'][:,i])**2
#     diff = diff[~np.isinf(diff)]
#     diff = diff[~np.isnan(diff)]
#     print('Emissivity mse: ',np.round(np.mean(diff),5))

