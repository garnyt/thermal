"""
Created on Fri Nov  8 12:31:06 2019

@author: Tania Kleynhans

calculate TOA Radiance with associated emissivity, ground temperatures and RSR
"""
import numpy as np
import matplotlib.pyplot as plt

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
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']
    
    radiance = np.trapz(list(RSR_interp * radiance_spectral),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength))
    
    return radiance

    
    
    