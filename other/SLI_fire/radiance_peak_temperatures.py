#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:08:53 2021

@author: tkpci
"""
import numpy as np
import matplotlib.pyplot as plt

h = 6.626 * 10**(-34)
c = 2.998 * 10**8
k = 1.381 * 10**(-23)

waves = np.arange(0.5,14,0.005)*10**(-6)
temperatures = [250,300,350,400,500,600,700,800,900,1000,1100,1200]
rad = np.zeros(waves.shape[0])

for T in temperatures:
    i=0
    
    for wave in waves:
    
        rad[i] = 2*h*c**(2) / (wave**(5) * (np.exp((h*c) / (wave * T * k))-1)) * 10**(-6)
        i+=1
        
    plt.plot(waves*10**(6), rad)
    
plt.legend(('250K','300K','350K','400K','500K','600K','700K','800K','900K','1000K','1100K','1200K'))
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Radiance [W/$m^2 sr \mu m$]')
plt.yscale('symlog')