#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:12:21 2020

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy as sc
import pdb
import pandas as pd
from multiprocessing import Process, Manager,Pool, cpu_count
import os
from datetime import datetime
import csv

# run MODTRAN    
def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    
    command = 'cp /cis/staff2/tkpci/modtran/tape5/' + tape5_name + ' /cis/staff2/tkpci/modtran/modtran_output/tape5'
    
    os.system(command) 
    
    command = 'cd /cis/staff2/tkpci/modtran/modtran_output\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff2/tkpci/modtran/modtran_output/tape6 /cis/staff2/tkpci/modtran/tape6/' + tape6_name 
    os.system(command) 
    
    command = 'rm /cis/staff2/tkpci/modtran/modtran_output/*'
    os.system(command)


def calc_rad_from_tape6():
    
    tape6 = read_tape6('tape6_1',filepath='/cis/staff/tkpci/modtran/tape6/')
    wavelength = tape6['wavelength']
    
    
    #RSR to use for analysis
    RSR10 = load_RSR(sensor='TIRS10',center=10.9,FWHM=0.59)    #10.9   #eon 10.8
    RSR11 = load_RSR(sensor='TIRS11',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
    # interpolate to wavelength
    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)  
    
    # get UCSB emissivity files and interpolate to wavelenght
    emis10, emis_spectral = emis_interp(wavelength,RSR10_interp, num=0)
    emis11, emis_spectral = emis_interp(wavelength,RSR11_interp, num=0)
    
    # create dictionary to hold data
    dataOut = {}
    dataOut['rad10'] = []
    dataOut['rad11'] = []
    dataOut['emis10'] = []
    dataOut['emis11'] = []
    dataOut['skintemp'] = []
    dataOut['CWV'] = []
    dataOut['RH'] = []
    dataOut['T10'] = []
    dataOut['T11'] = []
    dataOut['trans10'] = []
    dataOut['trans11'] = []    
    
    
    for counter in range(2311):
        print('Busy with tape_' + str(counter))
        
        filename = 'tape6_' + str(counter+1)
        tape6 = read_tape6(filename , filepath='/cis/staff2/tkpci/modtran/tape6/')

        if tape6['RH'] > 90:
            print('Relative Humidity above 90% ... skipping')
            continue
        
        trans10 = np.trapz(list(RSR10_interp * tape6['transmission']),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR10_interp),x=list(tape6['wavelength']),axis=0)
        trans11 = np.trapz(list(RSR11_interp * tape6['transmission']),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR11_interp),x=list(tape6['wavelength']),axis=0)
    
       
        # calculate radiance for both bands
        T = np.asarray([-10,-5,0,5,10,15,20]) + tape6['skintemp'] # all 7 iterations
        
        
        for i in range(T.shape[0]):
             for j in range(emis10.shape[0]):
                dataOut['rad10'].append(calc_TOA_radiance(tape6,RSR10_interp,T[i],emis_spectral[j,:]))
                dataOut['rad11'].append(calc_TOA_radiance(tape6,RSR11_interp,T[i],emis_spectral[j,:]))
                dataOut['emis10'].append( emis10[j])
                dataOut['emis11'].append(emis11[j])
                dataOut['skintemp'].append(T[i])
                dataOut['CWV'].append(tape6['CWV'])
                dataOut['RH'].append(tape6['RH'])
                dataOut['T10'].append(inverse_planck(calc_TOA_radiance(tape6,RSR10_interp,T[i],emis_spectral[j,:]),tape6['wavelength']))
                dataOut['T11'].append(inverse_planck(calc_TOA_radiance(tape6,RSR11_interp,T[i],emis_spectral[j,:]),tape6['wavelength']))
                dataOut['trans10'].append(trans10)
                dataOut['trans11'].append(trans11)                
            

    return dataOut, wavelength

###########################################read in tape6 file ###########################################
    
def read_tape6(filename,filepath='/cis/staff2/tkpci/modtran/tape6/'):

    infile = open(filepath+filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()

    word = 'REL H'
    start_ind = []
    start_ind_T = []
    start_ind_KM= []
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
                k += 1
                relH = lines[i+3+k]
                RH_line = relH[30:33]
                RH_line = RH_line.replace(" ","")
            
    relH = np.asarray(start_ind)
    T_profile = np.asarray(start_ind_T)
    KM_profile = np.asarray(start_ind_KM)
    RH = max(relH)
    #print('RH: ' + str(max(relH)))

    word = 'GM / CM2'
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            CWV_line = lines[i+2]
            CWV = float(CWV_line[16:24])
            break
    
    word = 'USER INPUT DATA:'
    for i in range(0,len(lines)):
        k=0
        if word in lines[i]:
            skintemp_line = lines[i+1]
            skintemp = float(skintemp_line[21:31])
            break
    
    
    
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
    tape6['CWV'] = np.float64(CWV)
    tape6['RH'] = RH
    tape6['skintemp'] = skintemp

    idx = np.argsort(tape6['wavelength'])
    tape6['wavelength'] = tape6['wavelength'][idx]
    tape6['path_thermal'] = tape6['path_thermal'][idx]
    tape6['ground_reflected'] = tape6['ground_reflected'][idx]
    tape6['transmission'] = tape6['transmission'][idx]
    
    tape6['dewpoint'] = T_profile - ((100-relH)/5)
    tape6['T_profile'] = T_profile
    tape6['KM_profile'] = KM_profile
    tape6['RH_profile'] = relH

    return tape6

##################################### calc toa radiance ###############################################

def load_RSR(center, FWHM, sensor='TIRS10'):
    
    # load RSR of specified sensor
    
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    if sensor == 'TIRS10':
        filename = filepath + 'TIRS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'TIRS11':
        filename = filepath + 'TIRS.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,2]
    elif sensor == 'MODIS31':
        filename = filepath + 'MODIS_31.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,2]
        rsr = data[:,3]
    elif sensor == 'MODIS32':
        filename = filepath + 'MODIS_32.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,2]
        rsr = data[:,3]
    elif sensor == 'ASTER13':
        filename = filepath + 'ASTER_13.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'ASTER14':
        filename = filepath + 'ASTER_14.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        rsr = data[:,1]
    elif sensor == 'GAUSS':
        wave = np.arange(7.9,14.1,0.01)
        rsr = norm.pdf(wave,center,FWHM/2.3548)
        rsr = rsr/max(rsr)
        #plt.plot(wave,rsr)
    elif sensor == 'RECT':
        wave = np.arange(7.9,14.1,0.01)
        rsr = np.zeros(wave.shape)
        rsr = np.where((wave > center - FWHM/2) & (wave < center + FWHM/2),1,0)
        #plt.plot(wave,rsr)
    
    RSR = {}
    RSR['wave']= wave
    RSR['rsr'] = rsr
        
    return RSR

def interp_RSR(wavelength,RSR, toplot = 1):
    # interp RSR to tape6 wavelangth
    RSR_interp = np.interp(wavelength, RSR['wave'], RSR['rsr'])

    if toplot == 1:
        plt.plot(wavelength, RSR_interp)
        
    return RSR_interp
        
def calc_TOA_radiance(tape6,RSR_interp,T,emis):
    # calculate TOA radiance
    h = 6.626 * 10**(-34);
    c = 2.998 * 10**8;
    k = 1.381 * 10**(-23);
    
   
    surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (tape6['wavelength']**(5) * (np.exp((h*c) / (tape6['wavelength'] * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
    
    radiance = np.trapz(list(RSR_interp * radiance_spectral),x=list(tape6['wavelength']),axis=0)/np.trapz(list(RSR_interp),x=list(tape6['wavelength']),axis=0)
    
    return radiance


######################################### get emissivity files ############################################
    
def emis_interp(wavelength,RSR_interp, num = 0):
    
    if num != 1:
    
        filename = '/cis/staff2/tkpci/emissivities/SoilsMineralsParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        
        emis_spectral = []
        emis = []
        
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        
        filename = '/cis/staff2/tkpci/emissivities/VegetationParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
    
        filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
        
    else:
        
        emis_spectral = []
        emis = []
        
        filename = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities_1.csv'
        data = np.genfromtxt(filename, delimiter=',')
        data = data[:,0:2]
        wave = data[:,0]
    
        
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
            
        emis=np.asarray(emis)
        emis_spectral = np.asarray(emis_spectral)
    
    return emis, emis_spectral


############################################## calc app temp for B10 and B11

def appTemp(radiance, band='b10'):
    
    if band == 'b10':
        K2 = 1321.0789;
        K1 = 774.8853;
    elif band == 'b11':
        K2 = 1201.1442;
        K1 = 480.8883;
    else:
        print('call function with T = appTemp(radiance, band=\'b10\'')    
        return
    
    temperature = np.divide(K2,(np.log(np.divide(K1,np.array(radiance)) + 1)));
    
    return temperature


def inverse_planck(radiance_spectral,wavelength):
    
    wvl = wavelength #* 1e-6

    L = radiance * 1e6
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    Temp = (2 * h * c * c) / (L * (wvl**5))
    Temp2 = np.log(Temp+1)
    appTemp = (h * c )/ (k * wvl *Temp2)
    
    return appTemp







#dataOut = pd.read_csv('/cis/staff2/tkpci/Code/Python/TIGR/dataOut.csv')
#
#
############################################## calc CWV coefficients ####################################

def calc_SW(dataOut):     
    
    T10 =  np.array(dataOut['T10'])
    T10 = T10.flatten()
    T11 =  np.array(dataOut['T11'])
    T11 = T11.flatten()
    
    e10 = np.array(dataOut['emis10'])
    e10 = e10.flatten()
    e11 = np.array(dataOut['emis11'])
    e11 = e11.flatten()
    
    T_diff = (T10 - T11)/2
    T_plus =  (T10 + T11)/2
    e_mean = (e10 + e11)/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (e10-e11)/(e_mean**2)
    quad = (T10-T11)**2
    
    b0 = np.ones(T_diff.shape)
    
    y = np.array(dataOut['skintemp'])
    y = y.flatten()
    #y = skintemp
    
    x = []
    # b0
    x.append(b0)
    # b1
    x.append(T_plus)
    # b2
    x.append(T_plus*e_diff)
    # b3
    x.append(T_plus*e_change)
    # b4
    x.append(T_diff)
    # b5
    x.append(T_diff*e_diff)
    # b6
    x.append(T_diff*e_change)
    # b7
    x.append(quad)
            
    x = np.array(x)
    
    LST_SW = 2.2925 + 0.9929*x[1,:] + 0.1545*x[2,:] -0.3122*x[3,:] + 3.7186*x[4,:] + 0.3502*x[5,:] -3.5889*x[6,:] + 0.1825*x[7,:]
    #diff = LST_SW-y
    rmse = np.sqrt(np.mean((LST_SW - y)**2))
    
    print('rmse: ' + str(rmse))
    
    dataOut['LST_SW'] = LST_SW   
    return dataOut

# for num in range(2311):
#     tape5_name = 'tape5_'+str(num+1)
#     tape6_name = 'tape6_'+str(num+1)
#     runModtran(tape5_name, tape6_name)
    
    
    
    
# dataOut, wavelength = calc_rad_from_tape6()
# dataOut = calc_SW(dataOut)

# for val in dataOut.values():
#     print(val.shape)
# print(dataOut.keys())

# dframe = pd.DataFrame.from_dict(dataOut)
# dframe.to_csv("/cis/staff2/tkpci/Code/Python/TIGR/dataOut.csv")

temp = pd.read_csv('/cis/staff2/tkpci/Code/Python/TIGR/dataOut.csv')
dataOut = temp.to_dict()
print(dataOut.keys())

T10 = np.array(list(dataOut['T10'].values()))
T11 = np.array(list(dataOut['T11'].values()))
emis10 = np.array(list(dataOut['emis10'].values()))
emis11 = np.array(list(dataOut['emis11'].values()))
skintemp = np.array(list(dataOut['skintemp'].values()))
LST_SW = np.array(list(dataOut['LST_SW'].values()))
trans10 = np.array(list(dataOut['trans10'].values()))
trans11 = np.array(list(dataOut['trans11'].values()))

CWV = np.array(list(dataOut['CWV'].values()))

# dataOut = calc_SW(dataOut)
diff = LST_SW - skintemp

plt.scatter(CWV, LST_SW - skintemp, s=2)
plt.xlabel('CWV - MODTRAN value')
plt.ylabel('TIGR diff (SW - skintemp')

bins = np.zeros([CWV.shape[0],])+6
bins[CWV < 6] = 5
bins[CWV < 5] = 4
bins[CWV < 4] = 3
bins[CWV < 3] = 2
bins[CWV < 2] = 1
bins[CWV < 1] = 0

std = np.zeros([7,2])
std[:,0] = [0,1,2,3,4,5,6]
for i in range(7):
    
    std[i,1] = np.std(diff[bins == i])

val = ['0-1','1-2','2-3','3-4','4-5','5-6','>6']
plt.bar(val, std[:,1])
plt.xlabel('CWV')
plt.ylabel('Standard Deviation (K)')

    



ratio = np.divide(trans11,trans10)
z1 = np.polyfit(ratio[ratio > 0.6], CWV[ratio > 0.6],2)
p1 = np.poly1d(z1)

p1

# z = np.polyfit(np.divide(trans11,trans10), CWV,2, cov=True)
# p = np.poly1d(z)

xp = np.linspace(0.5, 1, 100)

plt.plot(xp, p1(xp), 'k')

plt.scatter(np.divide(trans11,trans10), CWV, s = 1.5)
#plt.xlim([0.5, 1])
#plt.ylim([0, 7])
plt.ylabel('CWV [g/cm$^2$]')
plt.xlabel('transmission ratio [$\u03C4_j/\u03C4_i$]')

# # linear case
# correlation_matrix = np.corrcoef(np.divide(trans11,trans10), CWV)
# correlation_xy = correlation_matrix[0,1]
# r_squared = correlation_xy**2

# ploy case
x = np.divide(trans11,trans10)
y = CWV
coeffs = np.polyfit(x, y, 2)

# r-squared
p = np.poly1d(coeffs)
# fit values, and mean
yhat = p(x)                         # or [p(z) for z in x]
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
R2 = ssreg / sstot



plt.plot(xp, p1(xp))

slope, intercept, r_value, p_value, std_err = sc.stats.linregress(np.divide(trans11,trans10), CWV)

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, 'k')




