#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:00:00 2021

@author: tkpci
"""


import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pdb
from multiprocessing import Process, Manager,Pool, cpu_count


# run MODTRAN    
def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    #pdb.set_trace()
    
    command = 'cp /cis/staff/tkpci/modtran/tape5/'+tape5_name +' /cis/staff/tkpci/modtran/tape5/tape5' 
    os.system(command) 
   
    command = 'cd /cis/staff/tkpci/modtran/tape5/'+'\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff2/tkpci/modtran/tape5/tape6 /cis/staff2/tkpci/modtran/tape6_user_defined_profiles/' + tape6_name 
    os.system(command) 
    
    command = 'rm -r /cis/staff/tkpci/modtran/tape5/'+tape5_name
    os.system(command)
    
def main():
    
    dir_files = os.listdir('/cis/staff/tkpci/modtran/tape5_1m/')
    # counter = 0
        
    # for tape5_file in dir_files:
    #     print('counter: ',counter)
    #     counter +=1
    #     tape6_name = 'tape6_' + tape5_file
        
    #     file_there = '/cis/staff2/tkpci/modtran/tape6_TIGR_uncertainty/' + tape6_name
    #     if not os.path.isfile(file_there):
    #         runModtran(tape5_file,tape6_name)
    #     #runModtran(dir_files[counter],tape6_name)
    
        
    myProcesses = []
    cnt = 0
    
    for tape5_file in dir_files:  
        
        if tape5_file[0:5] == 'tape5':
            
            print(tape5_file)
            tape6_name = 'tape6_' + tape5_file[6:]
            
            file_there = '/cis/staff2/tkpci/modtran/tape6_1m/' + tape6_name
            if not os.path.isfile(file_there):
                myProcesses.append(Process(target=runModtran, args=(tape5_file, tape6_name,)))
                cnt += 1
        


    
    cores = 18
    iter = int(cnt/cores)
    print("Running " + str(cores) + " processes at a time")
    
    for i in range(iter+1):
        print('Range busy with: ', i)
       
        start_cnt = (i+1)*cores - cores
        print("Start count = " , start_cnt)
        
        end_cnt = start_cnt + cores
        
        if end_cnt > cnt:
            end_cnt = cnt
            
        print('Start Process: ',i)
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.start()
            
        print('Join Process: ',i)
                               
        for process in myProcesses[start_cnt: end_cnt]:
            process.join()
            
        for process in myProcesses[start_cnt: end_cnt]:
            process.terminate()
            process.close()
            
        
if __name__ == "__main__":
    main()
