"""
Created  2022

@author: Tania Kleynhans

Creating a tape5 file to send to MODTRAN with standard profiles

1- tropical
2-mid-lat summer
3-mid-lat winter
4-sub-arctic summer
5-sub-arctic winter
6-1976 US standard atmoshpere
"""

import numpy as np
import pdb

def create_tape5(tape5_info={},filename='midlatSummer'):
    
    #pdb.set_trace()
    # check if all variables are there
    if 'profile' not in tape5_info:
        tape5_info['profile'] = 2
    if 'lat'not in tape5_info:
        tape5_info['lat'] = 43.1566   # Rochester NY lat
    if 'lon'not in tape5_info:
        tape5_info['lon'] = 77.6088   # Rochester NY lat
    if 'lst'not in tape5_info:
        tape5_info['lst'] = 300   # chosen temp
    if 'emissivity'not in tape5_info:
        tape5_info['emissivity'] = 0           # assign albedo as 1
    if 'alt'not in tape5_info:
        tape5_info['alt'] = 0    
    if 'time'not in tape5_info:
        tape5_info['time'] = 12.0
    if 'iday'not in tape5_info:
        tape5_info['iday'] = 1
    if 'filepath'not in tape5_info:
        tape5_info['filepath'] = '/cis/staff/tkpci/modtran/tape5/'
    
    
    # create tape5 file 
    full_filename = tape5_info['filepath'] +filename    
    f = open(full_filename,"w+")
    
    # write header information
    f.write("TS  %1.0f    2    2   -1    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['profile'],tape5_info['lst'],1-tape5_info['emissivity']))
    f.write("T   4F   0   0.00000       1.0       1.0 F F F         0.000\n");
    f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
    #f.write("   %2.0f    0    0\n" % tape5_info['lev']);

    # write footer information
    f.write("   750.000  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
    f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
    f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
    f.write("     8.000    14.000     0.020     0.025RM        M  A   \n");
    f.write("    0\n");
    f.close()
    

