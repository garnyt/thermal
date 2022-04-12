# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:39:06 2019

@author: Tania Kleynhans
"""
#imports

import numpy as np
import matplotlib.pyplot as plt


def emis_interp(RSR):
    
    filename = '/cis/staff/tkpci/emissivities/SoilsMineralsParsedEmissivities.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    
    emis_spectral = []
    emis = []
    
    wavelength = RSR['wave']
    RSR_interp = RSR['rsr']
    
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
    
    return emis, emis_spectral

def load_RSR(sensor='TIRS10'):
    
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

    
    RSR = {}
    RSR['wave']= wave
    RSR['rsr'] = rsr
        
    return RSR

def calc_coeff(tirs,a13,a14):

    
    y = tirs
    
    x = []
    # b0
    b0 = np.ones(a13.shape)
    x.append(b0)
    # b1
    x.append(a13)
    # b2
    x.append(a14)
        
    x = np.array(x).T
    
    coeff, residuals, rank, s = np.linalg.lstsq(x,y,rcond=None)
    
    x = x.T
    emis = coeff[0] + coeff[1]*x[1,:] + coeff[2]*x[2,:]
    rmse = np.sqrt(np.mean((emis - y)**2))
    print('RMSE: ',rmse)
    std = np.std(emis-y)
    print('STD: ', std)
    print(coeff)
    
    return coeff

def main():
    
    
    # calculate landsat emis coeeficients from aster
    # change sensor as needed
    RSR = load_RSR(sensor='TIRS10')
    emis, emis_spectral = emis_interp(RSR)
    RSR = load_RSR(sensor='ASTER13')
    a13, emis_spectral = emis_interp(RSR)
    RSR = load_RSR(sensor='ASTER14')
    a14, emis_spectral = emis_interp(RSR)
    
    coeff = calc_coeff(emis,a13,a14)
    
    
    # calculate snow coefficients
    RSR = load_RSR(sensor='TIRS2_11')
    filename = '/cis/staff/tkpci/emissivities/SnowParsedEmissivity.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    wave = np.flip(wave)
    snow = data[:,1]
    snow = np.flip(snow)
    
    temp = np.interp(RSR['wave'], wave, snow)
    #plt.plot(RSR['wave'],temp)
    snow_emis = np.trapz(list(temp * RSR['rsr']),x=list(RSR['wave']))/np.trapz(list(RSR['rsr']),x=list(RSR['wave']))

    # calculate water coefficients
    RSR = load_RSR(sensor='TIRS2_11')
    filename = '/cis/staff/tkpci/emissivities/WaterParsedEmissivity_salt.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    wave = np.flip(wave)
    water = data[:,1]
    water = np.flip(water)
    
    temp = np.interp(RSR['wave'], wave, water)
    #plt.plot(RSR['wave'],temp)
    water_emis = np.trapz(list(temp * RSR['rsr']),x=list(RSR['wave']))/np.trapz(list(RSR['rsr']),x=list(RSR['wave']))
    print(water_emis)
    
    # calculate veg coefficients
    RSR = load_RSR(sensor='TIRS2_10')
    filename = '/cis/staff/tkpci/emissivities/VegetationParsedEmissivities.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    wave = np.flip(wave)
    water = data[:,1]
    water = np.flip(water)
    
    temp = np.interp(RSR['wave'], wave, water)
    #plt.plot(RSR['wave'],temp)
    water_emis = np.trapz(list(temp * RSR['rsr']),x=list(RSR['wave']))/np.trapz(list(RSR['rsr']),x=list(RSR['wave']))
    print(water_emis)



    # check mean emissivity values

    RSR = load_RSR(sensor='TIRS11')
    
    filename = '/cis/staff/tkpci/emissivities/WaterIceSnowParsedEmissivities.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wave = data[:,0]
    
    wavelength = RSR['wave']
    RSR_interp = RSR['rsr']

    emis = []
    for i in range(data.shape[1]-1):
        #plt.plot(wave, data[:,i+1])
        temp = np.interp(wavelength, wave, data[:,i+1])
        emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
        #print(emis[i])
        

    np.mean(emis)








