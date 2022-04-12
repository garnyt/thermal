#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:36:27 2022

@author: tkpci
"""

import numpy as np
import pdb

def create_tape5():
    
    tape5_info = {}
    filename = 'tape5_RTTOV'
    tape5_info['filepath'] = '/cis/staff/tkpci/modtran/tape5_RTTOV/'
    
    
    tape5_info['tempK'] = np.flip(t_ex)
    tape5_info['ppmv'] = np.flip(q_ex)
    tape5_info['preshPa'] = np.flip(p_ex)
    tape5_info['altKm'] = -np.log(np.divide(np.flip(p_ex),1100))*7
    
    #pdb.set_trace()
    # check if all variables are there
    if 'lat'not in tape5_info:
        tape5_info['lat'] = 43.1566   # Rochester NY lat
    if 'lon'not in tape5_info:
        tape5_info['lon'] = 77.6088   # Rochester NY lat
    if 'lst'not in tape5_info:
        tape5_info['lst'] = tape5_info['tempK'][0]   # assign last layer of temperature
    if 'emissivity'not in tape5_info:
        tape5_info['emissivity'] = 0           # assign albedo as 1
    if 'lev'not in tape5_info:
        tape5_info['lev'] = len(tape5_info['tempK'])
    if 'alt'not in tape5_info:
        tape5_info['alt'] = tape5_info['altKm'][0]    
    if 'time'not in tape5_info:
        tape5_info['time'] = 12.0
    if 'iday'not in tape5_info:
        tape5_info['iday'] = 1


    
    # create tape5 file 
    filename = tape5_info['filepath'] +filename    
    f = open(filename,"w+")
    
    # write header information
    f.write("TS  7    2    2   -1    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['lst'],1-tape5_info['emissivity']))
    f.write("T   4F   0   0.00000       1.0       1.0 F F F         0.000\n");
    f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
    f.write("   %2.0f    0    0\n" % tape5_info['lev']);
    
    # write radiosonde or reanlysis data
    for i in range(tape5_info['lev']):
        f.write("  %8.3f %0.3e %0.3e %0.3e %0.3e %0.3eAAA             \n" % (tape5_info['altKm'][i],tape5_info['preshPa'][i],tape5_info['tempK'][i],tape5_info['ppmv'][i],0,0))
    
    # write footer information
    f.write("   705.000  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
    f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
    f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
    f.write("     8.000    14.000     0.020     0.025RM        M  A   \n");
    f.write("    0\n");
    f.close()