#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:07:04 2022

@author: tkpci
"""

# run TIGR profiles with RTTOV
import numpy as np
import run_RTTOV
import pandas as pd
import os
import sys
import csv
sys.path.append('/cis/staff2/tkpci/Code/Python/TIGR/')
import read_tape6
from multiprocessing import Process, Manager,Pool, cpu_count
import pdb

def RTTOV_output(num):
    
    emis10 = 0.00000001
    emis11 = 0.00000001
    
    out = pd.DataFrame(columns =('skintemp',
                                 'upwell10',
                                 'down10',
                                 'trans10',
                                 'upwell11',
                                 'down11',
                                 'trans11',
                                 'RH'))

    cnt = num + 1 # TIGR profiles starts at 0 and tape6 files at 1
    filename = 'tape6_' + str(cnt)  
    tape6 = read_tape6.read_tape6(filename , filepath='/cis/staff/tkpci/modtran/tape6/')
    
    print(num)
    dataRTTOV = run_RTTOV.run_RTTOV(emis10, emis11, num) # reads TIGR data from original csv 
        
    out.loc[0] = [tape6['skintemp'],
                  dataRTTOV['upwell10'],
                  dataRTTOV['down10'],
                  dataRTTOV['trans10'],
                  dataRTTOV['upwell11'],
                  dataRTTOV['down11'],
                  dataRTTOV['trans11'],
                  tape6['RH']]

    write_data_line_by_line(out)


def write_data_line_by_line(out):
    
 
    headers = ",".join(out.keys())
    values = ",".join(str(e) for e in out.iloc[0])     
    
    
    # write data to txt file
    filename_out = '/dirs/data/tirs/RTTOV/analysis/TIGR_RTTOV_data3.csv'
    
    if not os.path.isfile(filename_out):
        
        with open(filename_out, mode='w') as file_out:
            csv.excel.delimiter=';'
            file_writer = csv.writer(file_out, dialect=csv.excel)
        
            file_writer.writerow([headers])
            file_writer.writerow([values])
            
    else:
        with open(filename_out, mode='a') as file_out:
            csv.excel.delimiter=';'
            file_writer = csv.writer(file_out, dialect=csv.excel)
        
            file_writer.writerow([values])


def main():
    
       
    myProcesses = []
    profiles = 2311
    
    for val in range(profiles):        
        myProcesses.append(Process(target=RTTOV_output, args=(val,)))

        
    print(str(val+1) + " instances created")
    
    cores = 18
    
    iter = int(profiles/cores)
    print("Running " + str(cores) + " processes at a time")
    
    for i in range(iter+1):
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > profiles:
            end_cnt = profiles
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
            
        process.close()
            
            

if __name__=="__main__":
    data = main()
    
   
    
    
    
    
    
    
    
    
    
    