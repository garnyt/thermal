#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:58:49 2021

@author: tkpci

Calculate potential TOA radiance based on varying MODTRAN atmospheric profiles, water vapor, and RSR to fall into low atmospheric transmission area
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import pdb


# create RSR
def create_RSR(center, FWHM):
    
    #wave = np.arange(center-2,center+2,0.01)
    wave = np.arange(2,15,0.01)
    rsr = norm.pdf(wave,center,FWHM/2.3548)
    rsr = rsr/max(rsr)
    #plt.plot(wave,rsr)
    
    return rsr, wave
  
# create tape 5    
def create_tape5(tape5_info,filename):
    
    
    # check if all variables are there
    if 'lat'not in tape5_info:
        tape5_info['lat'] = 43.1566   # Rochester NY lat
    if 'lon'not in tape5_info:
        tape5_info['lon'] = 77.6088   # Rochester NY lat
    if 'lst'not in tape5_info:
        tape5_info['lst'] = 500    # lower fire temp
    if 'emissivity'not in tape5_info:
        tape5_info['emissivity'] = 0           # assign albedo as 1
    if 'alt'not in tape5_info:
        tape5_info['alt'] = 0   # sea level   
    if 'time'not in tape5_info:
        tape5_info['time'] = 12.0
    if 'iday'not in tape5_info:
        tape5_info['iday'] = 1
    if 'filepath'not in tape5_info:
        tape5_info['filepath'] = '/cis/staff/tkpci/modtran/tape5_predefined/'

    
    # create tape5 file 
    filename = tape5_info['filepath'] +filename    
    f = open(filename,"w+")

    
    # write header information
    f.write("TS  %1.0f    2    2   -1    0    0    0    0    0    0    1    1    0 %6.3f %6.2f\n" % (tape5_info['profile'],tape5_info['lst'],1-tape5_info['emissivity']))
    #f.write("T   4F   0   0.00000     g %1.1f         1.0 F F F         0.000\n" % tape5_info['cwv']);
    f.write("T   4F   0   0.00000       %1.1f         1.0 F F F         0.000\n" % tape5_info['cwv']);
    f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
    #f.write("   %2.0f    0    0\n" % tape5_info['lev']);

    # write footer information
    f.write("   750.000  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
    f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
    f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
    f.write("     6.500    14.000     0.020     0.025RM        M  A   \n");
    f.write("    0\n");
    f.close()
    
# run MODTRAN    
def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    
    command = 'cp /cis/staff2/tkpci/modtran/tape5_predefined/' + tape5_name + ' /cis/staff2/tkpci/modtran/tape5/tape5'
    
    os.system(command) 
    
    command = 'cd /cis/staff/tkpci/modtran/tape5\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff/tkpci/modtran/tape5/tape6 /cis/staff/tkpci/modtran/tape6_predefined/' + tape6_name 
    os.system(command) 
    
    command = 'rm /cis/staff/tkpci/modtran/tape5/*'
    os.system(command)
    
# calc apparent temperature with inverse Planck    
def inverse_planck(radiance_spectral,wavelength):
    
    wvl = wavelength * 1e-6

    L = radiance_spectral * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    Temp = (2 * h * c * c) / (L * (wvl**5))
    Temp2 = np.log(Temp+1)
    appTemp_spectral = (h * c )/ (k * wvl *Temp2)
    
    return appTemp_spectral
    
# calc TOA radiance 
def calc_TOA_radiance(tape6,rsr,T,emis,wavelength):
    # calculate TOA radiance
    h = 6.626 * 10**(-34)
    c = 2.998 * 10**8
    k = 1.381 * 10**(-23)
    
    wave = tape6['wavelength'] * 1e-6
    appTemp = np.zeros(len(T))
    radiance = np.zeros(len(T))
    
    
    for i in range(len(T)):
    
        surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (wave**(5) * (np.exp((h*c) / (wave * T [i]* k))-1)) * 10**(-6)
        radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
        
        radiance_spectral = np.interp(wavelength, tape6['wavelength'], radiance_spectral)
        
        
        appTemp_spectral = inverse_planck(radiance_spectral,wavelength)
        appTemp[i] = np.trapz(list(rsr * appTemp_spectral),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
        
        
        radiance[i] = np.trapz(list(rsr * radiance_spectral),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
    
    
    return radiance, appTemp 

# read tape6 info into dictionary
def read_tape6(filename,filepath='/cis/staff/tkpci/modtran/tape6_predefined/'):
    

    infile = open(filepath+filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()   
    
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
    
   

    return tape6
   
def plot_stuff(tape6, wavelength,rsr, legents):
    
    trans = np.interp(wavelength, tape6['wavelength'], tape6['transmission'])
    plt.plot(wavelength, trans, linewidth=1)
    plt.tight_layout(rect=[0,0,0.8,1])
    #plt.plot(wavelength, rsr)
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmission')
    plt.xlim([3,5])
    plt.ylim([0,1])
    #plt.legend(legents)
    plt.legend(legents,bbox_to_anchor=(1.04,1), loc="upper left")
    
    


def get_fire_parameters(fire_data):
    
    
    emis_model = 0.98    #Giglio et al. 1999
    #emis_model = 1    
    
    temp_range = np.arange(500,1300,100)
    #temp_range = np.asarray([1200])
    radiance = np.zeros(temp_range.shape[0])
    appTemp = np.zeros(temp_range.shape[0])
    
  
    

    counter2 = -1
    for j in temp_range:
        counter2 += 1
        radiance[counter2], appTemp[counter2] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr'] , j, emis_model, fire_data['wavelength'])
        

    fire_data['radiance'] = radiance
    fire_data['appTemp'] = appTemp
    fire_data['temp_range'] = temp_range
    
    return fire_data


def main():
    
    tape5_name = 'tape5'
    tape6_name = 'tape6'
    FWHM = 0.15
    center = 9.75
    emis = 1
    temperature = np.arange(250,1250,50)     #skintemp

    
    tape5_info= {}
    tape5_info['emissivity'] = emis   #fixed for fire
    #tape5_info['profile'] = 2   # 1 = tropical, 2 = mid-lat summer, 3 = mid-lat winter etc.
        # fire temperature
    #tape5_info['cwv'] = 2.0
    
    profiles = [1,2,3,4,5,6]
    #profiles = [2]
    #cwvs = np.arange(0.5,6.5,1)
    cwvs = [0]
    
    legent = []
    radiance = np.zeros((len(temperature),len(profiles)))
    appTemp = np.zeros((len(temperature),len(profiles)))
    
    
    #for cwv in cwvs:
    for profile in profiles:
        
        for cwv in cwvs:
        #for profile in profiles:
            fire_data = {}
            
            
        
                
            tape5_info['lst'] = 300
            tape5_info['profile'] = profile
            tape5_info['cwv'] = cwv
    
            try:
                create_tape5(tape5_info,tape5_name)
                runModtran(tape5_name,tape6_name)
                fire_data['tape6'] = read_tape6('tape6',filepath='/cis/staff/tkpci/modtran/tape6_predefined/')
                fire_data['rsr'],fire_data['wavelength']  = create_RSR(center, FWHM)
                # radiance, appTemp = calc_TOA_radiance(fire_data['tape6'],fire_data['rsr'],temperature,emis,fire_data['wavelength'])
                # plt.plot(temperature, radiance)
                # plt.xlabel('Temperature [K]')
                # plt.ylabel('Radiance [W/$m^2$/sr/$\mu$m]')
                # #plt.xlim([250,1200])
                # plt.xlim([250,350])
                # #plt.ylim([0,8])
                # legent.append('profile: '+str (profile))
                # plt.title('Radiances for 7.5 $\mu$m band looking at fire temperatures')
                plt.plot(fire_data['tape6']['wavelength'] ,fire_data['tape6']['transmission'] ,linewidth=1)
                plt.legend(('Tropical','Mid-lat summer','Mid-lat winter','Sub-arctic summer','Sub-arctic winter','U.S. Standard')) 
   
                
            except:
                print('Error: Profile: ', str(profile), ', CWV: ', str(cwv))
                    
    #plt.legend(legent) 
            
            
                    
    #plt.plot(fire_data['tape6']['wavelength'] ,fire_data['tape6']['transmission'] ,linewidth=1)
    # plt.plot(fire_data['wavelength'], fire_data['rsr'])
    # plt.xlim([6.5,14])
    # plt.legend(('Tropical','Mid-lat summer','Mid-lat winter','Sub-arctic summer','Sub-arctic winter','U.S. Standard','Potential RSR')) 
    # #plt.legend(('Transmission','Potential RSR'))
    # plt.xlabel('Wavelength [$\mu$m]')
    # plt.ylabel('Transmission')
    
    # filepath = '/cis/staff/tkpci/modtran/RSR/'
    # filename = filepath + 'TIRS.csv'
    # data = np.genfromtxt(filename, delimiter=',')
    # fire_data['wavelength_TIRS10'] = data[:,0]
    # fire_data['rsr_TIRS10'] = data[:,1]
    
    
    

    #fire_data = get_fire_parameters(fire_data)
    
main()

    
    
def radiance_to_DC():
    
    A_det = 25E-6 ** 2  #m^2
    f_num = 1.64
    t = 3.49E-3  # sec
    tau_optics = 0.76 # 0.12 no idea - this is a guess to make values work; 0.76 (T10 - Matt info)
    QE = 0.01  # saw note on this stating < 1%
    em = 0
    
    FWHM = 0.15
    center = 9.75
    rsr_4um, wavelength_4um  = create_RSR(center, FWHM)
    
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    filename = filepath + 'TIRS.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wavelength = data[:,0] # micron
    rsr = data[:,1]
    
        
    h = 6.626 * 10**(-34) # J.s
    c = 2.998 * 10**8 # m/s
    
    tape5_name = 'tape5'
    tape6_name = 'tape6'
    FWHM = 0.15
    #center = 
    emis = 1
    T = 400     #skintemp

    
    tape5_info= {}
    tape5_info['emissivity'] = emis   #fixed for fire
    tape5_info['profile'] = 2   # 1 = tropical, 2 = mid-lat summer, 3 = mid-lat winter etc.
    tape5_info['cwv'] = 1
    
    temp_range = np.arange(250,1250,50)
    data = np.zeros([temp_range.shape[0],4])
    data_4um = np.zeros([temp_range.shape[0],4])
    i = 0
    
    for T in temp_range:
        tape5_info['lst'] = T    # fire temperature
        
        
        create_tape5(tape5_info,tape5_name)
        runModtran(tape5_name,tape6_name)
        
        tape6 = read_tape6('tape6',filepath='/cis/staff/tkpci/modtran/tape6_predefined/')
        radiance_spectral = np.interp(wavelength, tape6['wavelength'], tape6['radiance'])# In W/m2/sr/um
        radiance = np.trapz(rsr * radiance_spectral,x=wavelength)/np.trapz(rsr,x=wavelength)  
        radiance_spectral_4um = np.interp(wavelength_4um, tape6['wavelength'], tape6['radiance'])# In W/m2/sr/um
        radiance_4um = np.trapz(rsr_4um * radiance_spectral_4um,x=wavelength_4um)/np.trapz(rsr_4um,x=wavelength_4um) 
         
        #radiance_m = radiance_spectral * 10**-6 # W/m^2.sr.m  
        value = (A_det * np.pi * (1-em))/(4 * f_num**2 * h * c) * t  * tau_optics * rsr * QE * wavelength*10**-6 * radiance_spectral  * tau_optics
        value_4um = (A_det * np.pi * (1-em))/(4 * f_num**2 * h * c) * t  * tau_optics * rsr_4um * QE * wavelength_4um*10**-6 * radiance_spectral_4um  * tau_optics
        
        electrons = np.trapz(value,x=wavelength) 
        electrons_4um = np.trapz(value_4um,x=wavelength_4um) 
        
        QSE =   5000000/(16**2-1)# wellDepth /(2**bitDepth-1)    #electrons /  DC
        DC = int(electrons/QSE)
        
        data[i,0] = T
        data[i,1] = radiance
        data[i,2] = electrons
        data[i,3] = DC
        data_4um[i,0] = T
        data_4um[i,1] = radiance_4um
        data_4um[i,2] = electrons_4um
        
        i += 1
        print(T)   
        print('Radiance: ' + str(np.round(radiance)))
        print('Electrons: ' + str(np.round(electrons)))
        print('Digital Count: ' + str(np.round(DC)))
        #print('Radiance (4um): ' + str(np.round(radiance_4um)))
        #print('Electrons (4um): ' + str(np.round(electrons_4um)))
        
        MLT_radiance_calc = (radiance + 0.1) / 3.3420E-4 
        print('MLT conversion electrons: ' + str(np.round(MLT_radiance_calc,0)))
        
    #plt.plot(data[:,0], (data[:,2]*3.3420E-4 -0.1 ))
    plt.plot(data_4um[:,0], data_4um[:,1])
    plt.plot(data[:,0],np.zeros([data.shape[0],])+22)
    plt.xlabel('Fire Temperature [K]')
    plt.ylabel('Radiance [W/$m^2$/sr/$\mu$m]')
    plt.legend(('TIRS band 10','7.5 $\mu$m band','Radiance = 22 W/$m^2$/sr/$\mu$m'))
    plt.title('Radiance')
 
#main()   
    
    
    
    
    
    
    
    