#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:34:20 2022

@author: tkpci
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def read_RSR():
    filepath = '/cis/staff2/tkpci/modtran/RSR/'
    filename = filepath + 'thermopiles.csv'
    data = np.genfromtxt(filename, delimiter=',')
    wavelength = data[:,0]
    rsrs = data[:,1:]
    volts = [ 'T1CA(v)', 'T1CB(v)', 'T1CC(v)', 'T1CD(v)',
            'T2CA(v)', 'T2CB(v)', 'T2CC(v)', 'T2CD(v)']
    plt.plot(wavelength,rsrs)
    plt.legend(volts)
    
    return rsrs, wavelength

def read_data(rsrs, wavelength):
    
    filename = '/cis/staff2/tkpci/1312022_roomtemp_10502.csv'
    df = pd.read_csv(filename)
    
    #pdb.set_trace()
    
    data = df.groupby('bbtemp [C]').mean().reset_index()
    
    data['rsr1_rad'] = np.zeros(len(data))
    data['rsr2_rad'] = np.zeros(len(data))
    data['rsr3_rad'] = np.zeros(len(data))
    data['rsr4_rad'] = np.zeros(len(data))
    data['rsr5_rad'] = np.zeros(len(data))
    data['rsr6_rad'] = np.zeros(len(data))
    data['rsr7_rad'] = np.zeros(len(data))
    data['rsr8_rad'] = np.zeros(len(data))
    
    data['rsr1_T'] = np.zeros(len(data))
    data['rsr2_T'] = np.zeros(len(data))
    data['rsr3_T'] = np.zeros(len(data))
    data['rsr4_T'] = np.zeros(len(data))
    data['rsr5_T'] = np.zeros(len(data))
    data['rsr6_T'] = np.zeros(len(data))
    data['rsr7_T'] = np.zeros(len(data))
    data['rsr8_T'] = np.zeros(len(data))
    
    tempC = data['bbtemp [C]']
    tempK = tempC.values + 273.15 
    
    count = 0
    
    
    
    for temperature in tempK:
        for i in range(rsrs.shape[1]):
            rsr = rsrs[:,i]
            L = planck(temperature, wavelength)
            temp_rsr = np.trapz(list(L * rsr),x=list(wavelength),axis=0)/np.trapz(list(rsr),x=list(wavelength),axis=0)
            name = 'rsr'+str(i+1)+'_rad'
            data[name][count] = temp_rsr
        count += 1
        
    return data
            

def planck(temperature, wavelength):
    
    wave = wavelength * 1e-6
    T = temperature 
    
    c = 2.99792458e8
    h = 6.6260755e-34
    k = 1.380658e-23
    L = (2 * h * c**2 / wave**5) * 1 / (np.exp((h*c)/(wave*k*T))-1)*1e-6
    
    #plt.plot(wavelength,L)
    
    return L

def main():
    
    # read in rsrs
    rsrs, wavelength = read_RSR()
    # calculate band effective radiance values for each temperature and rsr
    data = read_data(rsrs, wavelength)
    # calc associated radiance per band from LUT's for each rsr
    for i in range(8):
        rad = data['rsr'+str(i+1)+"_rad"]
        LUT = np.loadtxt("LUT_radiometer0"+str(i+1)+".csv", delimiter=',')
        data['rsr'+str(i+1)+"_T"] = np.interp(rad,LUT[:,0],LUT[:,1])
        
    # for i in range(len(data)):
    #     plt.plot(data['bbtemp [C]'], data['T'+str(i+1)+'CA(v)'])
        
    # plt.plot(data['bbtemp [C]'], data['T1Therm(ohm)'])
    volts = [ 'T1CA(v)', 'T1CB(v)', 'T1CC(v)', 'T1CD(v)',
            'T2CA(v)', 'T2CB(v)', 'T2CC(v)', 'T2CD(v)']
    
    v = np.zeros([len(data),8])
    i=0
    for volt in volts:
        v[:,i] = data[volt]
        i+=1
    
    plot_two_axis(data['bbtemp [C]'], v,data['T1Therm(ohm)'])
    #plt.figure()
    #plt.plot(data['bbtemp [C]'], v)
    #plt.legend(volts)
    #plt.xlabel('BB Temperature [C]')
    #plt.ylabel('Measured volts')
    
    return data
    

def plot_two_axis(x,y,y2):
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('BB temperature [C]')
    ax1.set_ylabel('volt')#, color=color)
    ax1.plot(x, y)#, color=color)
    ax1.tick_params(axis='y')#, labelcolor=color)
    volts = [ 'T1CA(v)', 'T1CB(v)', 'T1CC(v)', 'T1CD(v)',
            'T2CA(v)', 'T2CB(v)', 'T2CC(v)', 'T2CD(v)']
    ax1.legend(volts)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('ohm', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__=="__main__":
    data = main()
       
