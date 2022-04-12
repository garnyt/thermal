#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:40:10 2022

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt

#run SW_coeff_calc in TIGR directory - once with  MODIS RSRs

# MOD31 = dataOut['rad10']
# MOD32 = dataOut['rad11']

# #run SW_coeff_calc in TIGR directory - and once with TIRS2 RSRs

# TIRS2_10 = dataOut['rad10']
# TIRS2_11 = dataOut['rad11']



# regress for one of the below sensors
y = TIRS2_10
y = TIRS2_11

b0 = np.ones(TIRS2_10.shape)

y = y.flatten()

x = []
# b0
x.append(b0)
# b1
x.append(MOD32)
# b2
#x.append(MOD32)
    
x = np.array(x).T

coeff, residuals, rank, s = np.linalg.lstsq(x,y,rcond=None)

diff = TIRS2_11 - (coeff[0] + coeff[1]*MOD32)# + coeff[2]*MOD32)
print(np.std(diff))
diff = TIRS2_10 - (coeff[0] + coeff[1]*MOD31)# + coeff[2]*MOD32)
print(np.std(diff))






