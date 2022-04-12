#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:58:49 2021

@author: tkpci
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import pdb
from display_heat_map import display_heat_map

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
    f.write("T   4F   0   0.00000       1.0       1.0 F F F         0.000\n");
    f.write("    1    0    0    3    0    0     0.000     0.000     0.000     0.000  %8.3f\n" % tape5_info['alt']);
    #f.write("   %2.0f    0    0\n" % tape5_info['lev']);

    # write footer information
    f.write("   750.000  %8.3f   180.000     0.000     0.000     0.000    0          0.000\n" % tape5_info['alt']);
    f.write("    1    0  %3.0f    0\n" % tape5_info['iday']);
    f.write("  %8.3f  %8.3f     0.000     0.000  %8.3f     0.000     0.000     0.000\n" % (tape5_info['lat'],tape5_info['lon'],tape5_info['time']));
    f.write("     2.000    15.000     0.020     0.025RM        M  A   \n");
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
    
    surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (wave**(5) * (np.exp((h*c) / (wave * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
    
    radiance_spectral = np.interp(wavelength, tape6['wavelength'], radiance_spectral)
    
    
    appTemp_spectral = inverse_planck(radiance_spectral,wavelength)
    appTemp = np.trapz(list(rsr * appTemp_spectral),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
    
    
    radiance = np.trapz(list(rsr * radiance_spectral),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
    
    if T > 450:
        tape6=read_tape6('tape6_0.98_'+ str(T),filepath='/cis/staff/tkpci/modtran/tape6_predefined/')
        radiance_spectral_pure = np.interp(wavelength, tape6['wavelength'], tape6['radiance'])# In W/m2/sr/um
        radiance_pure = np.trapz(list(rsr * radiance_spectral_pure),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
        appTemp_spectral = inverse_planck(radiance_spectral_pure,wavelength)
        appTemp_pure = np.trapz(list(rsr * appTemp_spectral),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
       
        print(appTemp, appTemp_pure, T)
    
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
    # tape6['CWV'] = np.float64(CWV)
    # tape6['RH'] = RH
    # tape6['skintemp'] = skintemp

    idx = np.argsort(tape6['wavelength'])
    tape6['wavelength'] = tape6['wavelength'][idx]
    tape6['path_thermal'] = tape6['path_thermal'][idx]
    tape6['ground_reflected'] = tape6['ground_reflected'][idx]
    tape6['transmission'] = tape6['transmission'][idx]
    tape6['radiance'] = tape6['radiance'][idx]
    
    # tape6['dewpoint'] = T_profile - ((100-relH)/5)
    # tape6['T_profile'] = T_profile
    # tape6['KM_profile'] = KM_profile
    # tape6['RH_profile'] = relH

    return tape6
   
def plot_stuff(tape6, wavelength,rsr, temp_range, appTemp):
    
    trans = np.interp(wavelength, tape6['wavelength'], tape6['transmission'])
    plt.plot(wavelength, trans)
    plt.plot(wavelength, rsr)
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmission')
    plt.xlim([2,6])
    plt.ylim([0,1])
    plt.legend({'transmission','rsr'})
    
    plt.figure()
    plt.plot(temp_range, appTemp)
    
def get_background_parameters(fire_data):
    
    start_temp = 20 + 273.15
    end_temp = 30 + 273.15
    emis_4 = 0.96
    emis_10 = 0.96
    
    mu, sigma = start_temp + (end_temp - start_temp)/2  , 2.3 # mean and standard deviation (Giglio et Al.)
    s_temp = np.random.normal(mu, sigma, 25*25)
    mu_emis, sigma_emis = emis_4 , 0.009 # mean and standard deviation (Giglio et Al.)
    s_emis_4 = np.random.normal(mu_emis, sigma_emis, 25*25)
    mu_emis, sigma_emis = emis_10 , 0.009 # mean and standard deviation (Giglio et Al.)
    s_emis_10 = np.random.normal(mu_emis, sigma_emis, 25*25)
        
    background = {}
    radiance = np.zeros(s_temp.shape[0])
    appTemp = np.zeros(s_temp.shape[0])
    radiance_10 = np.zeros(s_temp.shape[0])
    appTemp_10 = np.zeros(s_temp.shape[0])
    
    
    
    # calculate background appTemp with random values
    for i in range(s_temp.shape[0]):
        radiance[i], appTemp[i] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr'], s_temp[i],s_emis_4[i], fire_data['wavelength'] )
        radiance_10[i], appTemp_10[i] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr_TIRS10'], s_temp[i], s_emis_10[i], fire_data['wavelength_TIRS10'] )
    
    mean_T4_bg = np.mean(appTemp)
    mean_T10_bg = np.mean(appTemp_10)
    mean_delta_bg = np.mean(appTemp - appTemp_10)
    mad_T4_bg = np.sum(np.abs(appTemp - np.mean(appTemp)))/s_temp.shape[0]
    mad_T10_bg = np.sum(np.abs(appTemp_10 - np.mean(appTemp_10)))/s_temp.shape[0]
    delta_T = appTemp - appTemp_10
    mad_delta_bg = np.sum(np.abs(delta_T - np.mean(delta_T)))/s_temp.shape[0]
    
    s_temp = np.arange(start_temp, end_temp,3)
    
    radiance = np.zeros(s_temp.shape[0])
    appTemp = np.zeros(s_temp.shape[0])
    radiance_10 = np.zeros(s_temp.shape[0])
    appTemp_10 = np.zeros(s_temp.shape[0])
    
    # calculate appTemp of fire pixel for area without fire
    cnt = 0
    for i in range(s_temp.shape[0]):
        radiance[cnt], appTemp[cnt] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr'], s_temp[i], emis_4, fire_data['wavelength'] )
        radiance_10[cnt], appTemp_10[cnt] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr_TIRS10'], s_temp[i], emis_10, fire_data['wavelength_TIRS10'] )
        cnt += 1
    
    background['radiance'] = radiance
    background['appTemp'] = appTemp
    background['radiance_10'] = radiance_10
    background['appTemp_10'] = appTemp_10
    background['mean_T4_bg'] = mean_T4_bg
    background['mean_T10_bg'] = mean_T10_bg
    background['mean_delta_bg'] = mean_delta_bg
    background['mad_T4_bg'] = mad_T4_bg
    background['mad_T10_bg'] = mad_T10_bg
    background['mad_delta_bg'] = mad_delta_bg
    
    return background

def get_fire_parameters(fire_data):
    
    
    emis_model = 0.98    #Giglio et al. 1999
    #emis_model = 1    
    
    temp_range = np.arange(500,1300,100)
    #temp_range = np.asarray([1200])
    radiance = np.zeros(temp_range.shape[0])
    appTemp = np.zeros(temp_range.shape[0])
    radiance_10 = np.zeros(temp_range.shape[0])
    appTemp_10 = np.zeros(temp_range.shape[0])
  
    

    counter2 = -1
    for j in temp_range:
        counter2 += 1
        radiance[counter2], appTemp[counter2] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr'] , j, emis_model, fire_data['wavelength'])
        radiance_10[counter2], appTemp_10[counter2] = calc_TOA_radiance(fire_data['tape6'] , fire_data['rsr_TIRS10'] , j, emis_model, fire_data['wavelength_TIRS10'] )
    

    fire_data['radiance'] = radiance
    fire_data['appTemp'] = appTemp
    fire_data['radiance_10'] = radiance_10
    fire_data['appTemp_10'] = appTemp_10 
    fire_data['temp_range'] = temp_range
    
    return fire_data

def calc_tests(fire_data, background, fire_fraction):
    
    fire_pixels = fire_data['appTemp'].shape[0]
    non_fire_pixels = background['appTemp'].shape[0]
    
    tests = np.zeros([fire_pixels * non_fire_pixels * fire_fraction.shape[0],3])
    
    cnt = 0
    
    for i in range(fire_pixels):
        for j in range(non_fire_pixels):
            for fire_frac in fire_fraction:
            
                T4 = fire_data['appTemp'][i] * fire_frac + background['appTemp'][j] * (1-fire_frac)
                T10 = fire_data['appTemp_10'][i] * fire_frac + background['appTemp_10'][j]  * (1-fire_frac)
                delta_T = T4 - T10 
                
                test1 = 0
                test2 = 0
                test3 = 0
                test4 = 0
                test5 = 0
                test6 = 0
                
                mad_fire = 6
                
                
               
                # test 1
                if T4 > 360:
                    test1 = 1
                    
                # test2
                if delta_T > (background['mean_delta_bg'] + 3.5 * background['mad_delta_bg']):
                    test2 = 1
                    #pdb.set_trace()
                    
                # test 3
                if delta_T > (background['mean_delta_bg'] + 6):
                    test3 = 1
                    
                # test 4
                if T4 > (background['mean_T4_bg'] + 3 * background['mad_T4_bg']):
                    test4 = 1
                    
                # test 5
                if T10 > (background['mean_T10_bg'] + background['mad_T10_bg'] - 4):
                    test5 = 1
                    
                    
                # test 6 (if neighboring pixels are fire)
                if mad_fire > 5:
                    test6 = 1
                
                
                if test1 or ((test2 and test3 and test4) and (test5 or test6)):
                    tests[cnt,0] = 1
                    
                # if test1:
                #     tests[cnt,0] = 1
                
                
                    
                tests[cnt,1] = fire_frac
                tests[cnt,2] = fire_data['temp_range'][i]
                
                cnt += 1
        
    
    return tests


def main():
    
    tape5_name = 'tape5'
    tape6_name = 'tape6'
    FWHM = 0.15
    center = 4
    emis = 1
    T = 400     #skintemp

    
    tape5_info= {}
    tape5_info['emissivity'] = emis   #fixed for fire
    tape5_info['profile'] = 2   # 1 = tropical, 2 = mid-lat summer, 3 = mid-lat winter etc.
    tape5_info['lst'] = T    # fire temperature
    
    fire_data = {}
    
    
    create_tape5(tape5_info,tape5_name)
    runModtran(tape5_name,tape6_name)
    fire_data['tape6'] = read_tape6('tape6',filepath='/cis/staff/tkpci/modtran/tape6_predefined/')
    fire_data['rsr'],fire_data['wavelength']  = create_RSR(center, FWHM)
    
    filepath = '/cis/staff/tkpci/modtran/RSR/'
    filename = filepath + 'TIRS.csv'
    data = np.genfromtxt(filename, delimiter=',')
    fire_data['wavelength_TIRS10'] = data[:,0]
    fire_data['rsr_TIRS10'] = data[:,1]
    
    plt.plot(fire_data['wavelength'],fire_data['rsr'])
    plt.plot(fire_data['wavelength_TIRS10'],fire_data['rsr_TIRS10'])
    

    fire_data = get_fire_parameters(fire_data)
    
    background = get_background_parameters(fire_data)
    
    fire_fraction = np.asarray([10, 10**1.5, 10**2, 10**2.5, 10**3, 10**3.5, 10**4, 10**4.5, 10**5])/10**6
    fire_fraction = np.arange(0.01,0.15,0.01)
    
    
    tests = calc_tests(fire_data, background, fire_fraction)  
    
    display_heat_map(tests)
    
def radiance_to_DC():
    
    A_det = 25E-6 ** 2  #m^2
    f_num = 1.64
    t = 3.49E-3  # sec
    tau_optics = 0.12 # no idea - this is a guess to make values work
    #tau_optics = 0.76177045  # Table from Matt for 10.8 um band -  note Matt also said DC in MTL file is not actual DC - so not sure how to confirm values
    QE = 0.01  # saw note on this stating < 1%
    em = 0
    
    FWHM = 0.15
    center = 4
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
    center = 4
    emis = 1
    T = 400     #skintemp

    
    tape5_info= {}
    tape5_info['emissivity'] = emis   #fixed for fire
    tape5_info['profile'] = 2   # 1 = tropical, 2 = mid-lat summer, 3 = mid-lat winter etc.
    
    temp_range = np.arange(300,1250,50)
    data = np.zeros([temp_range.shape[0],3])
    data_4um = np.zeros([temp_range.shape[0],3])
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
        
        data[i,0] = T
        data[i,1] = radiance
        data[i,2] = electrons
        data_4um[i,0] = T
        data_4um[i,1] = radiance_4um
        data_4um[i,2] = electrons_4um
        
        i += 1
        print(T)   
        print('Radiance: ' + str(np.round(radiance)))
        print('Electrons: ' + str(np.round(electrons)))
        #print('Radiance (4um): ' + str(np.round(radiance_4um)))
        #print('Electrons (4um): ' + str(np.round(electrons_4um)))
        
        MLT_radiance_calc = (radiance + 0.1) / 3.3420E-4 
        print('MLT conversion electrons: ' + str(np.round(MLT_radiance_calc,0)))
        
    plt.plot(data[:,0], (data[:,2]*3.3420E-4 -0.1 ))
    plt.plot(data_4um[:,0], data_4um[:,2]*3.3420E-4 -0.1)
    plt.plot(data[:,0],np.zeros([data.shape[0],])+22)
    plt.xlabel('Fire Temperature [K]')
    plt.ylabel('Radiance [W/$m^2$/sr/$\mu$m]')
    plt.legend(('TIRS band 10','4 $\mu$m band','Radiance = 22 W/$m^2$/sr/$\mu$m'))
    plt.title('Radiance')
    
    fire_fraction = np.arange(0.05,0.55,0.05)
    fire_fraction = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5,1 ])
    saturate = np.zeros([temp_range.shape[0],fire_fraction.shape[0]])
    saturate_4um = np.zeros([temp_range.shape[0],fire_fraction.shape[0]])
    
    for i in range(temp_range.shape[0]):
        for j in range(fire_fraction.shape[0]):
            rad = data[i,1]*fire_fraction[j] + data[1,1]*(1-fire_fraction[j])
            rad4um = data_4um[i,1]*fire_fraction[j] + data_4um[1,1]*(1-fire_fraction[j])
            
            if rad <= 22:
                saturate[i,j] = 1
            if rad4um <= 22:
                saturate_4um[i,j] = 1
                
def plot_saturate(saturate, fire_fraction, temp_range):
    
    fig, ax = plt.subplots()
    im = ax.imshow(saturate, cmap='gray_r')
    
    xlabs_unique = fire_fraction*100
    ylabs_unique = temp_range
    
    str1 = [str(e) for e in list((xlabs_unique.astype(int)))]
    #str1 = ['10', '', '10^2', '', '10^3', '', '10^4', '', '10^5'] #[str(e) for e in list((xlabs_unique))]
    str2 = [str(e) for e in list(ylabs_unique.astype(int))]
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(xlabs_unique.shape[0]))
    ax.set_yticks(np.arange(ylabs_unique.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(str1)
    ax.set_yticklabels(str2)
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    # for i in range(len(str1)):
    #     for j in range(len(str2)):
    #         text = ax.text(i, j, int(saturate[j, i]*100),
    #                         ha="center", va="center", color="g")
    
    ax.set_ylim(len(str2)-0.5, -0.5)
    ax.set_title("Saturation with 300K background temperature")
    plt.xlabel('Percent of pixel with fire')
    plt.ylabel('Temperature of fire [K]')
    fig.tight_layout()
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=[1,1,1], lw=6),
                    Line2D([0], [0], color=[0,0,0], lw=6)]
    
    ax.legend(custom_lines, ['Saturated', 'Detectable'])
    
    plt.show()     

 
#main()   
    
    
    
    
    
    
    
    