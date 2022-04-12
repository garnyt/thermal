# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:51 2019

@author: Tania Kleynhans

Calculate gain and bias, and apparent temperature for TIRS
"""

import numpy as np

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

def TIRS_radiance(counts, band='b10', GB='yes'):
    
    radiance = counts * 3.3420 * 10**(-4) + 0.1

    if (band == 'b10' and GB =='yes'):
        radiance = radiance * 1.0151 - 0.14774
    elif (band == 'b11' and GB =='yes'):
        radiance = radiance * 1.06644-0.46326;
    
    return radiance
        
        
        
    
    

