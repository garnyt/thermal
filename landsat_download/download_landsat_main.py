#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:10:52 2021

@author: tkpci
"""

import numpy as np
import csv
import download_landsat_sceneid
from tkinter import *   
from tkinter import filedialog 
import warnings
import os
import pdb
from multiprocessing import Process, Manager,Pool, cpu_count

    
def product2entityid(product_id):
    """ convert product landsat ID to entity ID

    Ex:
    LC08_L1TP_017030_20131129_20170307_01_T1 ->
    LC80170302013333LGN01
    """
    
    collection = '00'
    
    if len(product_id) == 21:
        return product_id
    else:
        path = product_id[10:13]
        row = product_id[13:16]
    
        date = datetime.datetime.strptime(product_id[17:25], '%Y%m%d')
    
    return 'LC8{path}{row}{date}LGN{coll}'.format(path=path, row=row, date=date.strftime('%Y%j'), coll=collection)

#check if data downloaded completely and remove if not
def check_downloaded_data(filepath, num_files = 24):
    
    scene_id_downloaded = os.listdir(filepath) 
    
    cnt = 1
    
    for scene in scene_id_downloaded:
        files = os.listdir(filepath + scene)
        if len(files) < num_files:
            os.system("rm -R "+filepath + scene)
            print('Scene ', scene, ' removed')
            cnt += 1
    
    print(cnt)
        
        
        
        
    

def main(csv_file, filepath, datasetName = "landsat_ot_c2_l1"):
    
    warnings.simplefilter('ignore')
    
  
    try:
        # if you have csv file with scene ID's
        f = open(csv_file, "r")
        scene_id = f.read().split("\n")
        scene_id_needed = scene_id
        
        # get list of landst scenes already downloaded that it does not download again
        # NOTE: currently this will only work if the file names have the same format i.e. scene ID vs product ID
        scene_id_already_downloaded = os.listdir(filepath) 
        
        # filter out scenes already downloaded
        scene_id = list(set(scene_id_needed) - set(scene_id_already_downloaded))

    except:
        # single scene input
        scene_id = list(csv_file)
                
    # myProcesses = []
    # profiles = len(scene_id)
    
    # for val in range(profiles):        
    #     myProcesses.append(Process(target=download_landsat_sceneid.main, args=(scene_id,filepath,datasetName,)))

        
    # print(str(val+1) + " instances created")
    
    # cores = 20
    
    # iter = int(profiles/cores)
    # print("Running " + str(cores) + " processes at a time")
    
    # for i in range(iter+1):
       
    #     start_cnt = (i+1)*cores - cores
    #     print("Start count = " , start_cnt)
        
    #     end_cnt = start_cnt + cores
        
    #     if end_cnt > profiles:
    #         end_cnt = profiles
            
    #     for process in myProcesses[start_cnt: end_cnt]:
    #         process.start()
                               
    #     for process in myProcesses[start_cnt: end_cnt]:
    #         process.join()
            
    #     process.close() 
    
    # call download code if not multithreading
    download_landsat_sceneid.main(scene_id,filepath,datasetName)
    

    
    
########################### change parameters for this ##############################################

# filepath where downloads should be saved
filepath = '/dirs/data/tirs/Landsat9/L9_C2L2_2021_202203/'

# choose level 1 or 2 data
# Collection 2 level 1 (Landsat data)
# datasetName = "landsat_ot_c2_l1"
# Collection 2 level 2 (SC Product)
datasetName = "landsat_ot_c2_l2"

# add path to csv file OR just add scene ID (both types work)
csv_file = '/dirs/data/tirs/Landsat9/L9Buoy.csv'
#or
#csv_file = ['LC80370372021319LGN00']
#csv_file = ['LC80370372021319LGN00','LC80140352021318LGN00','LC90370402021319LGN04']

# remove all files incompletely downloaded (usually whith large sets that does not run continiously)
# check_downloaded_data(filepath)

#download data
main(csv_file, filepath, datasetName)