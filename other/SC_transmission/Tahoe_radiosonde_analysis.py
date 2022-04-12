#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:52:54 2021

@author: tkpci
"""

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
#import get_geos
import os
import matplotlib.pyplot as plt
import parse_radiosonde
import pdb

# create tape 5    
def create_tape5(tape5_info,tape5_name):
    
    
    # check if all variables are there
    if 'lat'not in tape5_info:
        tape5_info['lat'] = 43.1566   # Rochester NY lat
    if 'lon'not in tape5_info:
        tape5_info['lon'] = 77.6088   # Rochester NY lat
    if 'lst'not in tape5_info:
        tape5_info['lst'] = tape5_info['tempK'][0]   # lower fire temp
    if 'emissivity'not in tape5_info:
        tape5_info['emissivity'] = 0.98           # assign albedo as 1
    if 'alt'not in tape5_info:
        tape5_info['alt'] = tape5_info['altKm'][0]    
    if 'time'not in tape5_info:
        tape5_info['time'] = 12.0
    if 'iday'not in tape5_info:
        tape5_info['iday'] = 1
    if 'filepath'not in tape5_info:
        tape5_info['filepath'] = '/cis/staff2/tkpci/modtran/tape5_geos/'
    if 'lev'not in tape5_info:
        tape5_info['lev'] = len(tape5_info['tempK'])

    
    # create tape5 file 
    filename = tape5_info['filepath'] +tape5_name    
    f = open(filename,"w+")

     # write header information
    f.write("TS  7    2    2    0    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['lst'],1-tape5_info['emissivity']))
    f.write("T   4F   0   0.00000       1.0       1.0 F F F         0.000\n");
    f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
    f.write("   %2.0f    0    0\n" % tape5_info['lev']);
    
    # write radiosonde or reanlysis data
    for i in range(tape5_info['lev']):
        f.write("  %8.3f %0.3e %0.3e %0.3e %0.3e %0.3eAAF             \n" % (tape5_info['altKm'][i],tape5_info['preshPa'][i],tape5_info['tempK'][i],tape5_info['dewpK'][i],0,0))
    
    # write footer information
    f.write("   705.000  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
    f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
    f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
    f.write("     8.000    14.000     0.020     0.025RM        M  A   \n");
    f.write("    0\n");
    f.close()
    
        
# run MODTRAN    
def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    
    #command = 'cp /cis/staff2/tkpci/modtran/tape5_geos/' + tape5_name + ' /cis/staff2/tkpci/modtran/tape5/tape5'
    
    #os.system(command) 
    
    command = 'cd /cis/staff2/tkpci/modtran/tape5_geos\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff2/tkpci/modtran/tape5_geos/tape6 /cis/staff2/tkpci/modtran/tape6_geos/' + tape6_name 
    os.system(command) 
    
    command = 'rm /cis/staff2/tkpci/modtran/tape5_geos/*'
    os.system(command)
    
# read tape6 info into dictionary
def read_tape6(filename,filepath='/cis/staff2/tkpci/modtran/tape6_geos/'):
    

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
    
    #pdb.set_trace()
    
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
    
    
    tape6['T_profile'] = T_profile
    tape6['KM_profile'] = KM_profile
    tape6['RH_profile'] = relH
    tape6['pHa_profile'] = pHa_profile  #[mb]
    
    tape6 = calc_CWV_from_RH_profile(tape6, relH.shape[0])

    return tape6

def calc_CWV_from_RH_profile(tape6, stop):
    #pdb.set_trace()
    
    T_profile = tape6['T_profile']
    relH = tape6['RH_profile'] 
    pHa_profile = tape6['pHa_profile']  
    
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
    
    #filename = '/dirs/data/tirs/downloads/Surfrad/results/analysis_level2_CWVinfo.csv'
    filename = '/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_level2_trans.csv'
    out_dir = '/dirs/data/tirs/downloads/'
    
    out = pd.read_csv(filename)
    
    # lat = np.array(out['SR_lat'])
    # lon = np.array(out['SR_lon'])
    lat = np.array(out['Buoy_lat'])
    lon = np.array(out['Buoy_lon'])
    time = np.array(out['L8_time'])
    date = np.array(out['L8_date'])
    
        
    
    tape5_name = 'tape5'
    tape6_name = 'tape6'
   
    
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    filename = filepath + 'TIRS.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wavelength = data[:,0]
    rsr10 = data[:,1]
    rsr11 = data[:,2]
    ratio = np.zeros(len(lat))
    cwv_t = np.zeros(len(lat))
    cwv_other = np.zeros(len(lat))
    cwv_modtran = np.zeros(len(lat))
    cwv_calculated = np.zeros(len(lat))
   


    for i in range(len(lat)):
        print(i)
        
        
        date1 = date[i]
        hour = int(np.floor(time[i]))
        minute = int((time[i]-hour)*60)
        date_out = datetime.datetime.strptime(date1+' '+str(hour)+' '+str(minute), '%m/%d/%Y %H %M')
        #tape5_info['time'] = hour
        
        month = date_out.strftime("%b")
        
        try:
            
            
            #profile = get_geos.geos_process(date_out, lat[i], lon[i], out_dir)
            data_out, tape5_info = parse_radiosonde.parse_radiosonde_data(12,date_out.day, month.upper() ,date_out.year)
            create_tape5(tape5_info,tape5_name)
            runModtran(tape5_name,tape6_name)
            tape6 = read_tape6('tape6',filepath='/cis/staff/tkpci/modtran/tape6_geos/')
            wave = tape6['wavelength']
            transmission = tape6['transmission']
            trans = np.interp(wavelength, wave, transmission)
            
            trans10 = np.trapz(list(rsr10 * trans),x=list(wavelength),axis=0)/np.trapz(list(rsr10),x=list(wavelength),axis=0)
            trans11 = np.trapz(list(rsr11 * trans),x=list(wavelength),axis=0)/np.trapz(list(rsr11),x=list(wavelength),axis=0)
            ratio[i] = trans11/trans10
            #cwv_t[i] = 1.421*ratio[i]**2 - 15.643*ratio[i] + 14.258
            #cwv_other[i] = -9.674*ratio[i]**2 + 0.653*ratio[i] + 9.088
            cwv_modtran[i] = tape6['CWV']
            cwv_calculated[i] = tape6['cwv_calculated']
            print('CALCULTAED CWV: ', tape6['cwv_calculated'])
            print('MODTRAN CWV', tape6['CWV'])
        except:
            ratio[i] = 0
            
    #ratio = list(ratio)
    #cwv_t = list(cwv_t)
    #cwv_other = list(cwv_other)
    cwv_modtran = list(cwv_modtran)
    cwv_calculated = list(cwv_calculated)
    #out['ratio'] = ratio
    #out['cwv_t'] = cwv_t
    #out['cwv_other'] = cwv_other
    out['cwv_modtran_radiosonde'] = cwv_modtran
    out['cwv_calculated_radiosonde'] = cwv_calculated
    
    #out.to_csv('/dirs/data/tirs/downloads/Surfrad/results/analysis_level2_trans.csv',index = False)
    out.to_csv('/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_level2_trans_radiosonde.csv',index = False)
    
    return out
    
out = main()
            
            
def plot_data():
    filename = '/dirs/data/tirs/downloads/Buoy_results/buoy_analysis_level2_trans_radiosonde.csv'
        
    out = pd.read_csv(filename)
    
    SW = np.array(out['BuoylessL8_LST'])
    SC = np.array(out['BuoylessSC_LST'])
    CWV_radio = np.array(out['cwv_modtran_radiosonde'])
    CWV_GEOS = np.array(out['cwv_modtran'])
    #CWV_other = np.array(out['cwv_other']) 
    #cloud = np.array(out['SC_CDIST']) 
    SW=SW[CWV_radio > 0]
    SC=SC[CWV_radio > 0]
    #CWV_other=CWV_other[CWV > 0]
    #cloud=cloud[CWV > 0]
    
    CWV_GEOS=CWV_GEOS[CWV_radio > 0]
    CWV_radio=CWV_radio[CWV_radio > 0]

    plt.scatter(CWV_GEOS,CWV_radio, s=2) 
    #plt.scatter(CWV_other,abs(SW), s=2) 
    #plt.scatter(cloud[cloud > 1],abs(SW[cloud > 1]), s=2) 
    #plt.ylim([0,10])
    #plt.xlim([0,10])
    plt.xlabel('Total CWV [g/$cm^2$]')
    plt.ylabel('Absolute difference [K]')
    plt.title('LST difference (SW-Surfrad) vs. CWV (GEOS-FP)')
    plt.legend(('Self derived CWV coefficients','Ren et. al. coefficients'))




      
            
            