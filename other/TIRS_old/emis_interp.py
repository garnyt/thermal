# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:39:06 2019

@author: Tania Kleynhans
"""
#imports

import numpy as np


def emis_interp(wavelength,RSR_interp):
    
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
    
    return emis, emis_spectral
