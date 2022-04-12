"""
Created on Mon Nov  4 12:04:48 2019

@author: Tania Kleynhans

Read TIGR data and write tape 5 files
"""

import csv
import numpy as np
from datetime import datetime
import create_tape5
#import run_modtran  
import read_tape6 
import calc_TOA_radiance
import time
#from dataTIGR import DataTIGR
#import emis_interp
import pdb

def read_TIGR():
    filename = '/cis/staff2/tkpci/TIGR/tigr1.csv'
    tape5_info = {'emissivity':0}
    alt_cut_off = 42 # 
    tape5_info['filepath'] = '/cis/staff/tkpci/modtran/tape5/'
    
    
    pressure = [2.6e-3,8.9e-3,2.4e-2,0.5E-01,0.8999997E-01,0.17E+00,0.3E+00,0.55E+00,
     0.1E+01,0.15E+01,0.223E+01,0.333E+01,0.498E+01,
     0.743E+01,0.1111E+02,0.1660001E+02,0.2478999E+02,0.3703999E+02,
     0.4573E+02,0.5646001E+02,0.6971001E+02,0.8607001E+02,0.10627E+03,
     0.1312E+03,0.16199E+03,0.2E+03,0.22265E+03,0.24787E+03,
     0.27595E+03,0.3072E+03,0.34199E+03,0.38073E+03,0.4238501E+03,
     0.4718601E+03,0.525E+03,0.5848E+03,0.65104E+03,0.72478E+03,
     0.8E+03,0.8486899E+03,0.9003301E+03,0.9551201E+03,0.1013E+04]
    tape5_info['preshPa'] = np.flip(pressure)
    tape5_info['altKm'] = -np.log(np.divide(np.flip(pressure),1013))*7

    counter = 0
    # read data into array
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        data = np.empty((60087,6), dtype=object)

        for row in readCSV:
            for i in range(6):
                try:
                    data[counter,i] = np.float(row[i])
                except:
                    data[counter,i] = 0
            counter +=1

    #dataOut = DataTIGR()
    counter = 0
    inc = 0
    
    #reduced_TIGR = np.genfromtxt('tigr_200_profiles_index.csv', delimiter=',') # reduced TIGR starts at 0
    
    while counter < 26*2310+10:
        
        if counter < 26*2310+10:
        #if inc in reduced_TIGR:
        
            tape5_info['lat'] = data[counter,3]/100
            tape5_info['lon']= data[counter,2]/100
            
            if tape5_info['lon'] < 180:
                tape5_info['lon']= (data[counter,2]/100)*(-1)
            else:
                tape5_info['lon']= abs(360-data[counter,2]/100)    
            
            year = round(data[counter,4]/10000)
            month = round(data[counter,4]/100)-(round(data[counter,4]/10000)*100)
            
            day = data[counter,4]-(year * 10000 + month*100)
            if day == 0:
                day = 1
            
            if (datetime(year, month, int(day))- datetime(year, 1, 1)).days == 0:
                tape5_info['iday'] = 1
            else:        
                tape5_info['iday'] = (datetime(year, month, int(day))- datetime(year, 1, 1)).days
            
            tempk = []
            cwv = [] # this is not dewpoint, but water vapor profiles  - need to multiply by 1000 - g/kg
            gKg = []
            
            for i in range(7):      
                for j in range(6):
                    tempk.append(data[counter+i+1,j])
                    cwv.append(data[counter+i+10,j])
                    gKg.append(data[counter+i+18,j])
            
            tempk.append(data[counter+i+2,0])
            cwv.append(data[counter+i+11,0])
            gKg.append(data[counter+i+19,0])
                
            tape5_info['tempK']=np.flip(tempk)
            tape5_info['cwv']=np.flip(cwv)*1000
            tape5_info['gKg']=np.flip(gKg)*1000
            tape5_info['lst'] = tape5_info['tempK'][0]
            tape5_info['alt'] = tape5_info['altKm'][0]  
            
            tape5_info['preshPa'] = tape5_info['preshPa'][0:alt_cut_off]
            tape5_info['cwv']=tape5_info['cwv'][0:alt_cut_off]
            tape5_info['gKg']=tape5_info['gKg'][0:alt_cut_off]
            tape5_info['tempK'] = tape5_info['tempK'][0:alt_cut_off]
            tape5_info['altKm'] = tape5_info['altKm'][0:alt_cut_off]  
            

            # create tape5 file
            filename = 'tape5_'+str(inc)
            print(filename)
            create_tape5.create_tape5(tape5_info,filename)
            #create_tape5.create_tape5_uncertainty(tape5_info,filename, cwv_val = 10, temperature_val = 5)
            
        counter += 26
        inc += 1
        
read_TIGR()        
        
        
        
        
        
        
        
        