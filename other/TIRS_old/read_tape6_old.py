"""
Created on Fri Nov  8 11:33:20 2019

@author: Tania Kleynhans

Retrieve spectral data from MODTRAN tape6
"""
import numpy as np
import pdb

def read_tape6(RH = 0, filename='D:/Users/tkpci/Documents/DIRS_Research/MODTRAN/tape6/tape6'):

    infile = open(filename, 'r', encoding='UTF-8')   # python3
    lines = infile.readlines()  # .strip()
    infile.close()
    RHcheck = 1
    
    if RH == 1:
        word = 'REL H'
        start_ind = []
        for i in range(0,len(lines)):
            k=0
            if word in lines[i]:
                relH = lines[i+3]
                while str.isnumeric(relH[34:36]):
                    start_ind.append(float(relH[31:36]))
                    k += 1
                    relH = lines[i+3+k]
                
        relH = np.asarray(start_ind)
        
        if max(relH) > 90:
            RHcheck = 0
    
    if RHcheck == 0:
        tape6 = 0
        return tape6
    else:
    
        tape6 = {}
        word = 'WAVLEN'
        start_ind = []
        for i in range(0,len(lines)):
            k=0
            if word in lines[i]:
                wave = lines[i+4]
                while not str.isnumeric(wave[4:6]):
                    k += 1
                    wave = lines[i+4+k]
                start_ind.append(i+4+k)
    
        data = []
        for i in range(len(start_ind)):
            data.append(lines[start_ind[i]:start_ind[i]+50])
    
        
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
    #            temp = results[i]
    #            try:
    #                float(temp[0])
    #            except:
    #                pass
    #            else:
                 ind.append(i)
        
        #pdb.set_trace()            
        output_data = np.asarray(results[0:ind[-1]]).astype(float)
    
        tape6['wavelength'] = output_data[:,1] # In Microns
        tape6['path_thermal'] = output_data[:,3] * 10**4 # In W/m2/sr/um
        tape6['ground_reflected'] = output_data[:,9] * 10**4 # In W/m2/sr/um
        tape6['transmission'] = output_data[:,14]
    
        idx = np.argsort(tape6['wavelength'])
        tape6['wavelength'] = tape6['wavelength'][idx]
        tape6['path_thermal'] = tape6['path_thermal'][idx]
        tape6['ground_reflected'] = tape6['ground_reflected'][idx]
        tape6['transmission'] = tape6['transmission'][idx]
    
        return tape6


