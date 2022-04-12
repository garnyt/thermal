#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:23:48 2021

@author: tkpci
"""

import numpy as np
import pdb

def parse_radiosonde_data(time_0_12,day_val,month_val,year_val, grnd_alt = 0):
    
    file = '/cis/staff/tkpci/radiosonde/radiosonde_reno_nv_2013_2021.txt'
    
    #new_reading = '    254      0      1      APR    2013'
    new_reading = ('    254     %2.0f     %2.0f      %3s    %4.0f' % (time_0_12, day_val, month_val, year_val))
    
    #f.write("TS  7    2    2   -1    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['lst'],1-tape5_info['emissivity']))
    
    
    infile = open(file, 'r')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()  
    
    data_line = np.zeros([1,6])
    data_out = np.empty([1,6])
   
    for i in range(0,len(lines)):
        
        if new_reading in lines[i]:
            
            tempold = 0
            
            newline = lines[i+4]
            lat = float(lines[i+1][23:28])
            lon = float(lines[i+1][29:35])
            k=0
            
            while '    254     ' not in newline:
                
                
                temp = float(newline[21:28])  # check if temperature is valid
                alt = float(newline[15:21])
                
                if temp != 99999 and alt >= grnd_alt and tempold != temp:
                    
                
                    data_line[0,0] = float(newline[8:14])/10            # pressure  tenths of milibar - converted to milibar
                    data_line[0,1] = float(newline[15:21])/1000              # height [m] - converted to km
                    data_line[0,2] = float(newline[21:28])/10 +273.15   # temp tenths of celcius - converted to Kelvin
                    data_line[0,3] = float(newline[28:35])/10 +273.15   # dewpt tenths of celcius - converted to Kelvin
                    data_line[0,4] = float(newline[35:42])      # wind dir degrees
                    data_line[0,5] = float(newline[42:49])      # wind speed tenths of m/s or knots - depending on user choice
                    
                    data_out = np.vstack([data_out,data_line])
                tempold = float(newline[21:28]) 
                k += 1
                newline = lines[i+4+k]
                
                
    
                
    tape5_info = {}
    tape5_info['lat'] = lat
    tape5_info['lon'] = lon
    tape5_info['time'] = time_0_12
    tape5_info['altKm'] = data_out[1:,1]
    tape5_info['preshPa'] = data_out[1:,0]
    tape5_info['tempK'] = data_out[1:,2]
    tape5_info['dewpK'] = data_out[1:,3]
                
    return data_out, tape5_info
            
            

# time_0_12 = 12
# day_val = date_out.day
# month_val = month.upper()
# year_val = 2013

# data_out, tape5_info = parse_radiosonde_data(time_0_12,day_val,month_val,year_val)





