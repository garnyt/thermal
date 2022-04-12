"""
Created on Mon Nov  4 09:29:02 2019

@author: Tania Kleynhans

Creating a tape5 file to send to MODTRAN with radiosonde / reanalysis data

Input variables: 

Output variables:
    tape5_info = {'lst': 300 }.....................................will default to lowest air temperature level if none is provided
    tape5_info['emissivity'] = 0                                   default
    tape5_info['tempK']=[234,455,654,123.6]
    tape5_info['mmr']=[] mass mxing ratio (water vapor)
    tape5_info['preshPa']=[0.3,0.58,100,254]
    tape5_info['altKm']=[1,200,3000,5000]
    tape5_info['lat']= 43.1566                                     Rochester NY default
    tape5_info['lon']= 77.6088                                     Rochester NY default
    tape5_info['lev']= 43                                          will default to number of temperature levels entered
    tape5_info['time'] = 12.0                                      will default to 12pm GMT if none is provided
    tape5_info['iday'] = 1.........................................will default to 1 if none is provided
    tape5_info['filepath'] = 'D:/Users/tkpci/Documents/DIRS_Research/MODTRAN/tape5/'
"""

import numpy as np

def create_tape5(tape5_info,filename):
    
    
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
    if 'filepath'not in tape5_info:
        tape5_info['filepath'] = 'D:/Users/tkpci/Documents/DIRS_Research/MODTRAN/tape5/'
    if 'gKg' not in tape5_info:
        tape5_info['gKg'] = np.zeros(len(tape5_info['tempK']))
    
    # create tape5 file 
    filename = tape5_info['filepath'] +filename    
    f = open(filename,"w+")
    
    # uncomment for less levels
    # tape5_info['lev'] = 7
    
    # write header information
    f.write("TS  7    2    2   -1    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['lst'],1-tape5_info['emissivity']))
    f.write("T   4F   0   0.00000       1.0       1.0 F F F         0.000\n");
    f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
    f.write("   %2.0f    0    0\n" % tape5_info['lev']);
    
    # write radiosonde or reanlysis data
    for i in range(tape5_info['lev']):
        f.write("  %8.3f %0.3e %0.3e %0.3e %0.3e %0.3eAAC C           \n" % (tape5_info['altKm'][i],tape5_info['preshPa'][i],tape5_info['tempK'][i],tape5_info['mmr'][i],0,tape5_info['gKg'][i]))
    
    # write footer information
    f.write("     3.900  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
    f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
    f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
    f.write("     8.000    14.000     0.020     0.025RM        M  A   \n");
    f.write("    0\n");
    f.close()
    

