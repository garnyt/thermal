#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:06:52 2021

@author: tkpci

Donwload geos reanalysis data and get profiles for MODTRAN based on Surfrad locations and dates
"""

import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pdb

# create tape 5    
# def create_tape5(tape5_info,filename):
    
    
#     # check if all variables are there
#     if 'lat'not in tape5_info:
#         tape5_info['lat'] = 43.1566   # Rochester NY lat
#     if 'lon'not in tape5_info:
#         tape5_info['lon'] = 77.6088   # Rochester NY lat
#     if 'lst'not in tape5_info:
#         tape5_info['lst'] = 300    # lower fire temp
#     if 'emissivity'not in tape5_info:
#         tape5_info['emissivity'] = 0           # assign albedo as 1
#     if 'alt'not in tape5_info:
#         tape5_info['alt'] = tape5_info['altKm'][0] + 0.1  # sea level   
#     if 'time'not in tape5_info:
#         tape5_info['time'] = 12.0
#     if 'iday'not in tape5_info:
#         tape5_info['iday'] = 1
#     if 'filepath'not in tape5_info:
#         tape5_info['filepath'] = '/cis/staff2/tkpci/modtran/tape5_geos/'
#     if 'lev'not in tape5_info:
#         tape5_info['lev'] = len(tape5_info['tempK'])

    
#     # create tape5 file 
#     filename = tape5_info['filepath'] +filename    
#     f = open(filename,"w+")

#      # write header information
#     f.write("TS  7    2    2   -1    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['lst'],1-tape5_info['emissivity']))
#     f.write("T   4F   0   0.00000       1.0       1.0 F F F         0.000\n");
#     f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
#     f.write("   %2.0f    0    0\n" % tape5_info['lev']);
    
#     # write radiosonde or reanlysis data
#     for i in range(tape5_info['lev']):
#         f.write("  %8.3f %0.3e %0.3e %0.3e %0.3e %0.3eAAF             \n" % (tape5_info['altKm'][i],tape5_info['preshPa'][i],tape5_info['tempK'][i],tape5_info['dewpK'][i],0,0))
    
#     # write footer information
#     f.write("   705.000  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
#     f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
#     f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
#     f.write("     8.000    14.000     0.020     0.025RM        M  A   \n");
#     f.write("    0\n");
#     f.close()
    
        
# run MODTRAN    
def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    
    #command = 'cp /cis/staff2/tkpci/modtran/tape5_geos/' + tape5_name + ' /cis/staff2/tkpci/modtran/tape5/tape5'
    
    #os.system(command) 
    
    command = 'cd /cis/staff2/tkpci/modtran/tape5_predefined\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff2/tkpci/modtran/tape5_predefined/tape6 /cis/staff2/tkpci/modtran/tape6_predefined/' + tape6_name 
    os.system(command) 
    
    command = 'rm /cis/staff2/tkpci/modtran/tape5_predefined/*'
    os.system(command)
    
# read tape6 info into dictionary
def read_tape6(filename,filepath='/cis/staff2/tkpci/modtran/tape/'):
    

    infile = open(filepath+filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()  
    
    word = 'REL H'
    start_ind = []
    start_ind_T = []
    start_ind_KM= []
    start_ind_pHa=[]
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            relH = lines[i+3]
            #pdb.set_trace()
            RH_line = relH[30:33]
            RH_line = RH_line.replace(" ","")
            while str.isnumeric(RH_line):
                start_ind.append(float(relH[30:36]))
                start_ind_T.append(float(relH[22:28]))
                start_ind_KM.append(float(relH[4:11]))
                start_ind_pHa.append(float(relH[11:19]))
                k += 1
                relH = lines[i+3+k]
                RH_line = relH[30:33]
                RH_line = RH_line.replace(" ","")
            
    relH = np.asarray(start_ind)
    T_profile = np.asarray(start_ind_T)
    KM_profile = np.asarray(start_ind_KM)
    pHa_profile = np.asarray(start_ind_pHa)
    
    tape6 = {}
    word = 'WAVLEN'
    start_ind = []
    end_ind = []
    for i in range(0,len(lines)):
        k=0
        kk = 0
        if word in lines[i]:
            
            wave = lines[i+4]
            while not str.isnumeric(wave[4:6]):
                k += 1
                wave = lines[i+4+k]
            start_ind.append(i+4+k)
            while str.isnumeric(wave[4:6]):
                k += 1
                wave = lines[i+4+k] 
            end_ind.append(i+4+k-1)                    
            for j in range(i+4+k,i+4+58):
                if 'THIS WARNING WILL NOT BE REPEATED' in lines[j]:
                    start_ind.append(j+1)
                    wave = lines[j+1] 
                    while str.isnumeric(wave[4:6]):
                        kk += 1
                        wave = lines[j+1+kk] 
                    end_ind.append(j+kk+1)   
    
    word = 'GM / CM2'
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            CWV_line = lines[i+2]
            CWV = float(CWV_line[16:24])
            break
    
    data = []
    for i in range(len(start_ind)):
        data.append(lines[start_ind[i]:end_ind[i]])

    
    results = []
    for i in range(0,len(data)):
        data1 = data[i]
        for j in range(0,len(data1)):
            prse = data1[j].split(' ')
            try:
                float(prse[6])
            except:
                pass
            else:
                while ("" in prse):
                    prse.remove("")
                results.append(prse)


    ind = []
    for i in range(0,len(results)):
        if len(results[i]) == 15:
             ind.append(i)
       
    output_data = np.asarray(results[0:ind[-1]]).astype(float)

    tape6['wavelength'] = output_data[:,1] # In Microns
    tape6['path_thermal'] = output_data[:,3] * 10**4 # In W/m2/sr/um
    tape6['ground_reflected'] = output_data[:,9] * 10**4 # In W/m2/sr/um
    tape6['transmission'] = output_data[:,14]
    tape6['radiance'] = output_data[:,12] * 10**4 # In W/m2/sr/um
    tape6['CWV'] = np.float64(CWV)
    # tape6['RH'] = RH
    # tape6['skintemp'] = skintemp

    idx = np.argsort(tape6['wavelength'])
    tape6['wavelength'] = tape6['wavelength'][idx]
    tape6['path_thermal'] = tape6['path_thermal'][idx]
    tape6['ground_reflected'] = tape6['ground_reflected'][idx]
    tape6['transmission'] = tape6['transmission'][idx]
    tape6['radiance'] = tape6['radiance'][idx]
    
    
    
    # # calculate dewpoint from temperature and RH
    # logRH = np.ma.log(relH/100)
    # logRH = logRH.filled(0)
    # mask = np.copy(logRH)
    # mask[mask != 0] =1
    # dewpoint = (243.5 * (logRH + (17.67*(T_profile-273.15))/(243.5+(T_profile-273.15)))) \
    #     /(17.67 - (logRH + (17.67*(T_profile-273.15))/(243.5+(T_profile-273.15)))) + 273.15
     
    # # calculate water vapor pressure
    # dewpoint = dewpoint * mask
    # logDP = np.ma.log(dewpoint)
    # logDP = logDP.filled(0)
    # vapor_pressure = 0.01 * np.exp(1.391499 - 0.048640239 * dewpoint + (0.41764768*10**(-4))* dewpoint**2 \
    #                                -(0.14452093*10**(-7))*dewpoint**3 + 6.5459673 * logDP - 5800.2206/(dewpoint+0.00001)) #[mb]
    
    # # calculate water vapor mixing ratio
    # # note mb = hPa = 100 pa = 100 N/m^2 = 100 kg/(s^2.m) (N = kg.m/s^2)
    # mixing_ratio = 0.622*((vapor_pressure)/(pHa_profile-vapor_pressure)) #g/kg
    
    # # calculate total column water vapor
    # g = 9.807  #m/s^2 acceleration due to gravity
    # #pw = 1000 #kg/m^3 water density
    # cwv_calc = 1/(g) * np.trapz(np.flip(mixing_ratio), np.flip(pHa_profile)) # s/m^2 * g/kg * 100kg / s^2.m = g/cm^2.100
    # cwv_calculated = cwv_calc/100*1000  #still not following the units...
    
    #pdb.set_trace()
    
    tape6['T_profile'] = T_profile
    tape6['KM_profile'] = KM_profile
    tape6['RH_profile'] = relH
    tape6['pHa_profile'] = pHa_profile  #[mb]
    
    #pdb.set_trace()
    tape6 = calc_CWV_from_RH_profile(tape6, relH.shape[0]-1)

    return tape6

def calc_CWV_from_RH_profile(tape6, stop):
    #pdb.set_trace()
    
    T_profile = tape6['T_profile']
    relH = tape6['RH_profile'] 
    pHa_profile = tape6['pHa_profile']+0.00001  
    
    # calculate dewpoint from temperature and RH
    logRH = np.ma.log(relH/100)
    logRH = logRH.filled(0)
    mask = np.copy(logRH)
    mask[mask != 0] =1
    dewpoint = (243.5 * (logRH + (17.67*(T_profile-273.15))/(243.5+(T_profile-273.15)))) \
        /(17.67 - (logRH + (17.67*(T_profile-273.15))/(243.5+(T_profile-273.15)))) + 273.15
     
    # calculate water vapor pressure
    dewpoint = dewpoint * mask
    logDP = np.ma.log(dewpoint)
    logDP = logDP.filled(0)
    vapor_pressure = 0.01 * np.exp(1.391499 - 0.048640239 * dewpoint + (0.41764768*10**(-4))* dewpoint**2 \
                                   -(0.14452093*10**(-7))*dewpoint**3 + 6.5459673 * logDP - 5800.2206/(dewpoint+0.00001)) #[mb]
    
    # calculate water vapor mixing ratio
    # note mb = hPa = 100 pa = 100 N/m^2 = 100 kg/(s^2.m) (N = kg.m/s^2)
    mixing_ratio = 0.622*((vapor_pressure)/(pHa_profile-vapor_pressure)) #g/kg
    
    # calculate total column water vapor
    g = 9.807  #m/s^2 acceleration due to gravity
    #pw = 1000 #kg/m^3 water density
    cwv_calc = 1/(g) * np.trapz(np.flip(mixing_ratio[0:stop]), np.flip(pHa_profile[0:stop])) # s/m^2 * g/kg * 100kg / s^2.m = g/cm^2.100
    cwv_calculated = cwv_calc/100*1000  #still not following the units...
    
    tape6['dewpoint'] = dewpoint
    tape6['cwv_calculated'] = cwv_calculated
    
    return tape6


def main():
    
    
    filepath='/cis/staff/tkpci/modtran/tape6/'
    
    stops = np.asarray([10,15,20])
    cwv_out = np.zeros((stops.shape[0]+1,len(os.listdir(filepath))+1))
    counter = 1
    
    #for cwv in cwvs:
    for name in os.listdir(filepath):
      
        try:
            
            tape6 = read_tape6(name, filepath='/cis/staff/tkpci/modtran/tape6/')
            cwv_out[3,0] = tape6['KM_profile'][-1]
            cwv_out[3,counter] = tape6['cwv_calculated']
            cnt = 0
            
            
            for stop in stops:
                #pdb.set_trace()
                cwv_out[cnt,0] = tape6['KM_profile'][stop]
                tape6 = calc_CWV_from_RH_profile(tape6, stop)
                cwv_out[cnt,counter] = tape6['cwv_calculated']
                cnt+=1
            counter+=1   
            #legents.append(('Profile: ' + str(profile)+ ', CWV: '+ str(cwv)))
            
            
            #plot_stuff(fire_data['tape6'], fire_data['wavelength'],fire_data['rsr'], legents)
        except:
            print('Error: Profile: ', name, ', CWV: ', str(cwv_out[1,counter]))
            counter+=1
    
    x = np.arange(1,2312,1)
    for i in range(3):
        plt.scatter(x,cwv_out[3,1:]-cwv_out[i,1:], s=2)
        #plt.xlim[0,6]
        
    plt.legend(('6.1 km','9.9 km','15 km'))
    plt.title('CWV difference between 90 km atmosphere and lower ')
    plt.ylabel('Difference in CW [g/cm^2]')
    plt.xlabel('TIGR profile')
    
    return cwv_out
    
cwv_out = main()
            
            
def plot_data():
    filename = '/dirs/data/tirs/downloads/Surfrad/results/analysis_level2_trans.csv'
        
    out = pd.read_csv(filename)
    
    SW = np.array(out['SRlessL8_LST'])
    SC = np.array(out['SRlessSC_LST'])
    CWV = np.array(out['cwv_t'])
    CWV_other = np.array(out['cwv_other']) 
    cloud = np.array(out['SC_CDIST']) 
    SW=SW[CWV > 0]
    SC=SC[CWV > 0]
    CWV_other=CWV_other[CWV > 0]
    cloud=cloud[CWV > 0]
    CWV=CWV[CWV > 0]

    plt.scatter(CWV,abs(SW), s=2) 
    plt.scatter(CWV_other,abs(SW), s=2) 
    #plt.scatter(cloud[cloud > 1],abs(SW[cloud > 1]), s=2) 
    #plt.ylim([0,10])
    #plt.xlim([0,10])
    plt.xlabel('Total CWV [g/$cm^2$]')
    plt.ylabel('Absolute difference [K]')
    plt.title('LST difference (SW-Surfrad) vs. CWV (GEOS-FP)')
    plt.legend(('Self derived CWV coefficients','Ren et. al. coefficients'))




      
            
            