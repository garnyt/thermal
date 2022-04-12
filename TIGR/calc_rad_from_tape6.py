#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:12:21 2020

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pdb
import SW_regression
import pandas as pd


def calc_rad_from_tape6(shift10 = 0, shift11 = 0, widen10 = 0, widen11 = 0):
    
    # fun tape6 once to get wavelength that RSR and emis ban be done outside of loop
    tape6 = read_tape6('tape6_1',filepath='/cis/staff/tkpci/modtran/tape6/')
    wavelength = tape6['wavelength']
    
    # download RSR for specific sensor - nominal
    #RSR10 = load_RSR(sensor='TIRS10')
    #RSR11 = load_RSR(sensor='TIRS11')
    
    #RSR to use for analysis
    RSR10 = load_RSR(sensor='RECT',center=10.9,FWHM=0.59)    #10.9   #eon 10.8
    RSR11 = load_RSR(sensor='RECT',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
    # interpolate to wavelength
    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
    
#    shift10 = 0
#    shift11 = 0
#    widen10 = 0
#    widen11 = 0
    
    # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
    RSR10_interp = shift_widen_bands(RSR10_interp, wavelength, shift10, widen10)
    RSR11_interp = shift_widen_bands(RSR11_interp, wavelength, shift11, widen11)
    
    
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
    
    cntr = 0
    
    reduced_TIGR = np.genfromtxt('tigr_200_profiles_index.csv', delimiter=',')
    reduced_TIGR = reduced_TIGR.astype(int)
    
    #for counter in range(2):    
    #    counter += 1
    for counter in reduced_TIGR:
        cntr +=1
        
        filename = 'tape6_' + str(counter+1)
        tape6 = read_tape6(filename , filepath='/cis/staff/tkpci/modtran/tape6/')

        if tape6['RH'] > 90:
            print('Relative Humidity above 90% ... skipping')
            continue
            
                
        
        # calculate radiance for both bands
        T = np.asarray([-10,-5,0,5,10,15,20]) + tape6['skintemp'] # all 7 iterations
        
#        
#        dataOut['rad10'].append(calc_TOA_radiance(tape6,RSR10_interp,T,emis_spectral))
#        dataOut['rad11'].append(calc_TOA_radiance(tape6,RSR11_interp,T,emis_spectral))
#        dataOut['emis10'].append(np.tile(emis10,T.shape[0]))
#        dataOut['emis11'].append(np.tile(emis11,T.shape[0]))
#        dataOut['skintemp'].append(np.repeat(T,emis_spectral.shape[0],axis=0))
#        
#        dataOut['CWV'].append(np.repeat(tape6['CWV'][np.newaxis],emis_spectral.shape[0]*T.shape[0],axis=0)) 
#        dataOut['RH'].append(np.repeat(tape6['RH'][np.newaxis],emis_spectral.shape[0]*T.shape[0],axis=0))    
#        
#        dataOut['T10'].append(appTemp(calc_TOA_radiance(tape6,RSR10_interp,T,emis_spectral), band='b10'))
#        dataOut['T11'].append(appTemp(calc_TOA_radiance(tape6,RSR11_interp,T,emis_spectral), band='b11'))
         
        for i in range(T.shape[0]):
             for j in range(emis10.shape[0]):
                dataOut['rad10'].append(calc_TOA_radiance(tape6,RSR10_interp,T[i],emis_spectral[j,:]))
                dataOut['rad11'].append(calc_TOA_radiance(tape6,RSR11_interp,T[i],emis_spectral[j,:]))
                dataOut['emis10'].append( emis10[j])
                dataOut['emis11'].append(emis11[j])
                dataOut['skintemp'].append(T[i])
                dataOut['CWV'].append(tape6['CWV'])
                dataOut['RH'].append(tape6['RH'])
                dataOut['T10'].append(appTemp(calc_TOA_radiance(tape6,RSR10_interp,T[i],emis_spectral[j,:]), band='b10'))
                dataOut['T11'].append(appTemp(calc_TOA_radiance(tape6,RSR11_interp,T[i],emis_spectral[j,:]), band='b11'))
                
            

    return dataOut, wavelength

###########################################shift or widen bands #########################################
    
def shift_widen_bands(RSR_interp, wavelength, shift = 0, widen = 0):
    
    if (shift == 0) and (widen == 0):
        new_RSR_interp = RSR_interp
    else:
        min_wave = min(wavelength)
        max_wave = max(wavelength)
        size_wave = wavelength.shape[0]
        delta_wave = (max_wave - min_wave)/(size_wave)
        
        if shift != 0:
            to_move = int(shift/delta_wave)
            to_move = -to_move
            new_RSR_interp = np.hstack([RSR_interp[to_move:],RSR_interp[:to_move]])
           
        if widen != 0:
            rsr = np.where(RSR_interp >0.01,1,0)
            rsr_wave = wavelength*rsr
            minval = np.min(rsr_wave[np.nonzero(rsr_wave)])
            maxval = np.max(rsr_wave[np.nonzero(rsr_wave)])
            centerval = (maxval-minval)/2+minval
            
            min_idx = np.where(wavelength == minval)
            max_idx = np.where(wavelength == maxval)
            
            new_RSR = np.hstack([0,RSR_interp[min_idx[0][0]:max_idx[0][0]+1],0])
            
            cnt = sum(rsr)+2
            to_widen = int((widen/delta_wave))
            new_min = minval - widen/2
            new_max = maxval + widen/2
            step = (new_max - new_min)/cnt
            wave = np.arange(new_min,new_max,step)
            diff = wave.shape[0] - new_RSR.shape[0]
            if diff > 0:
                 new_RSR = np.hstack([new_RSR,0])
            if diff < 0:
                new_RSR = new_RSR[:diff]
                
            new_RSR_interp = np.interp(wavelength, wave, new_RSR)
        
    
    return new_RSR_interp



###########################################read in tape6 file ###########################################
    
def read_tape6(filename,filepath='/cis/staff/tkpci/modtran/tape6/'):

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

def load_RSR(sensor='TIRS10',center=10.9,FWHM=0.5):
    
    # load RSR of specified sensor
    
    filepath = '/cis/staff/tkpci/modtran/RSR/'
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
    elif sensor == 'GAUSS':
        wave = np.arange(7.9,14.1,0.05)
        rsr = norm.pdf(wave,center,FWHM/2.3548)
        rsr = rsr/max(rsr)
        #plt.plot(wave,rsr)
    elif sensor == 'RECT':
        wave = np.arange(7.9,14.1,0.05)
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
    
#    emis = np.transpose(emis)
#    emis_num = emis.shape[1]
#    T_num = T.shape[0]
#    
    wavelength = tape6['wavelength'] * 10**(-6)
#    wavelength = np.repeat(wavelength[:,np.newaxis],emis_num*T_num,axis=1)
#    
#    RSR_interp = np.repeat(RSR_interp[:,np.newaxis],emis_num*T_num,axis=1)
#    
#    emis = np.tile(emis,T_num)
#    T = np.repeat(T,emis_num,axis=0)
#    T = np.repeat(T[:,np.newaxis],emis.shape[0],axis=1)
#    T = np.transpose(T)
#    
#    trans = np.repeat(tape6['transmission'][:,np.newaxis],emis_num*T_num,axis=1)
#    path = np.repeat(tape6['path_thermal'][:,np.newaxis],emis_num*T_num,axis=1)
#    ground = np.repeat(tape6['ground_reflected'][:,np.newaxis],emis_num*T_num,axis=1)
#    
#    surfEmis = emis * trans * 2*h*c**(2) / (wavelength**(5) * (np.exp((h*c) / (wavelength * T * k))-1)) * 10**(-6)
#    radiance_spectral = surfEmis + path + ground*(1-emis)
    
    surfEmis = emis * tape6['transmission'] * 2*h*c**(2) / (wavelength**(5) * (np.exp((h*c) / (wavelength * T * k))-1)) * 10**(-6)
    radiance_spectral = surfEmis + tape6['path_thermal'] + tape6['ground_reflected']*(1-emis)
    
    radiance = np.trapz(list(RSR_interp * radiance_spectral),x=list(wavelength),axis=0)/np.trapz(list(RSR_interp),x=list(wavelength),axis=0)
    
    return radiance


######################################### get emissivity files ############################################
    
def emis_interp(wavelength,RSR_interp, num = 0):
    
    if num != 1:
    
        filename = '/cis/staff/tkpci/emissivities/SoilsMineralsParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
        
        emis_spectral = []
        emis = []
        
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
            
        
        filename = '/cis/staff/tkpci/emissivities/VegetationParsedEmissivities_10.csv'
        data = np.genfromtxt(filename, delimiter=',')
        wave = data[:,0]
    
        for i in range(data.shape[1]-1):
            temp = np.interp(wavelength, wave, data[:,i+1])
            emis_spectral.append(temp)
            emis.append(np.trapz(list(RSR_interp * temp),x=list(wavelength))/np.trapz(list(RSR_interp),x=list(wavelength)))
    
        filename = '/cis/staff/tkpci/emissivities/WaterIceSnowParsedEmissivities_10.csv'
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
        
        filename = '/cis/staff/tkpci/emissivities/WaterIceSnowParsedEmissivities_1.csv'
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


def print_output_to_file(output): 

    
    w = csv.writer(open("output.csv", "w"))
    for key, val in dict.items():
        w.writerow([key, val])
 

def loop_though_options():       
    # loop through various optiosn

    output = {} 
    output['band10'] = [] 
    output['band11'] = []
    output['rmse_new_coeff'] = []
    output['stddev_new_coeff'] = []
    output['mean_new_coeff'] = []
    output['rmse_orig_coeff'] = []
    output['stddev_orig_coeff'] = []
    output['mean_orig_coeff'] = []
    
    
    #shift = list(np.arange(-0.5,0.5,0.1))
    #total = len(shift)**2
    #shift = list(np.arange(-0.5,0,0.3))
    move = np.arange(-0.5,0.6,0.1)
    move = np.round(np.where(abs(move) < 0.001,0,move),1)
    move = list(move)
    
    total = len(move)**2
    
    cnt = 0
        
    for i in move:
        for j in move:
            output['band10'].append(i)
            output['band11'].append(j)
        
            dataOut, wavelength = calc_rad_from_tape6(shift10 = 0, shift11 = 0, widen10 = i, widen11 = j)
            
            output = SW_regression.calc_SW_coeff(dataOut, output)
            cnt += 1

            df = pd.DataFrame(output)
            df.to_csv('SW_RSR_RECT_widen.csv')
            print('Saving line %d of %d to file' % (cnt, total))
             
    return output


output = loop_though_options()


def plot_RSR():
    
    tape6 = read_tape6('tape6_1214',filepath='/cis/staff/tkpci/modtran/tape6/')
    wavelength = tape6['wavelength']
    trans = tape6['transmission']
    trans = trans/np.max(trans)
    # download RSR for specific sensor - nominal
    RSR10 = load_RSR(sensor='TIRS10')
    RSR11 = load_RSR(sensor='TIRS11')
    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
    plt.plot(wavelength, RSR10_interp,'k', label = 'TIRS 10')
    plt.plot(wavelength, RSR11_interp,'k', label = 'TIRS 11')
    plt.plot(wavelength, trans, linewidth=0.5, c='0.55')
    
    #RSR to use for analysis
    RSR10 = load_RSR(sensor='TIRS10',center=10.9,FWHM=0.59)    #10.9   #eon 10.8
    RSR11 = load_RSR(sensor='TIRS11',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
    # interpolate to wavelength
    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
    
    shift10 = -0.5
    shift11 = -0.5
    widen10 = 0
    widen11 = 0
    
    # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
    RSR10_interp = shift_widen_bands(RSR10_interp, wavelength, shift10, widen10)
    RSR11_interp = shift_widen_bands(RSR11_interp, wavelength, shift11, widen11)
    
    plt.plot(wavelength, RSR10_interp,'b--', label = 'TIRS 10')
    plt.plot(wavelength, RSR11_interp,'r--', label = 'TIRS 11')
    
    shift10 = 0.5
    shift11 = 0.5
    widen10 = 0
    widen11 = 0
    
    RSR10 = load_RSR(sensor='TIRS10',center=10.9,FWHM=0.59)    #10.9   #eon 10.8
    RSR11 = load_RSR(sensor='TIRS11',center=12.0,FWHM=1.01)    #12.0   #eon 12.1
    # interpolate to wavelength
    RSR10_interp = interp_RSR(wavelength, RSR10, toplot = 0)
    RSR11_interp = interp_RSR(wavelength, RSR11, toplot = 0)
    # shift or widen bands (add value to shift / widen in micron - neg values work) Widen - value with wich to widen FWHM
    RSR10_interp = shift_widen_bands(RSR10_interp, wavelength, shift10, widen10)
    RSR11_interp = shift_widen_bands(RSR11_interp, wavelength, shift11, widen11)
    
    plt.plot(wavelength, RSR10_interp,'b--', label = 'TIRS 10')
    plt.plot(wavelength, RSR11_interp,'r--', label = 'TIRS 11')
    
    
    plt.ylim([0,1.05])
    plt.title('Landsat band 10 and 11 RSRs')
    plt.xlabel('Wavelength [um]')
    plt.xlim([8,14])
    #plt.legend()
    plt.draw()
    plt.show()
       


#print(len(dataOut['rad10']))

#np.save('dataOut_90.npy',dataOut)


#def get_dewpoint_profile_from_tape6():
#    
#    T = []
#    DP = []
#    KM = []
#    
#    tape6 = read_tape6('tape6_1',filepath='/cis/staff/tkpci/modtran/tape6/')
#    T = tape6['T_profile']
#    DP = tape6['dewpoint']
#    KM = tape6['KM_profile']
#    
#    index_90 = np.zeros(1386)
#    j=0
#    
#    for i in range(2311-1):
#        
#        name = 'tape6_'+str(i+2)
#        print(name)
#        tape6 = read_tape6(name,filepath='/cis/staff/tkpci/modtran/tape6/')  
#        if tape6['RH'] < 90:  
#            j +=1
#            T = np.dstack((T,tape6['T_profile']))
#            DP = np.dstack((DP,tape6['dewpoint']))
#            index_90[j] = i+1
#            
#    T = np.squeeze(T)
#    DP = np.squeeze(DP)
#    
#    data = np.vstack((T,DP))
#    
#    idx = np.squeeze(endmembers_index)
#    idx = idx.astype(int)
#    index_90 = index_90.astype(int)
#    
#    reduced_TIGR = index_90[idx]
#    
#    savefile = 'tigr_200_profiles_index.csv'
#
#    np.savetxt(savefile, reduced_TIGR, delimiter=',', fmt='%d')
#    
#    for i in idx:
#        plt.plot(np.transpose(T[:,i]),KM)
#    
#    # TIGR profiles from pdf
#    tropical = np.arange(1,872,2)
#    midlat1 = np.arange(873,1260,2)
#    midlat2 = np.arange(1261,1614,2)
#    polar1 = np.arange(1615,1718,2)
#    polar2 = np.arange(1719,2311,2)
#    
#    tigr_profiles = [1,873,1261,1615,1719]
#    
#    reduced_TIGR = np.genfromtxt(savefile, delimiter=',')
#    reduced_TIGR = reduced_TIGR.astype(int)
#    
#    for i in tropical:
#        plt.subplot(2,3,1)
#        if DP[0,i] != 0:
#            plt.plot(np.transpose(T[:,i]),KM)
#            plt.title('Tropical')
#    for i in midlat1:
#        plt.subplot(2,3,2)
#        if DP[0,i] != 0:
#            plt.plot(np.transpose(T[:,i]),KM)
#            plt.title('midlat1')
#    for i in midlat2:
#        plt.subplot(2,3,3)
#        if DP[0,i] != 0:
#            plt.plot(np.transpose(T[:,i]),KM)
#            plt.title('midlat2')
#    for i in polar1:
#        plt.subplot(2,3,4)
#        if DP[0,i] != 0:
#            plt.plot(np.transpose(T[:,i]),KM)
#            plt.title('polar1')
#    for i in polar2:
#        plt.subplot(2,3,5)
#        if DP[0,i] != 0:
#            plt.plot(np.transpose(T[:,i]),KM)
#            plt.title('polar2')
#            
#    



