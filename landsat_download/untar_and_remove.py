#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 07:30:05 2022

@author: tkpci
"""

import os

files = os.listdir('/cis/staff/tkpci/Code/Python/landsat_download/')
filepath = '/dirs/data/tirs/L8L9_C2L1_202004_202203/data/'

for file in files:
    if file[-3:] == ".gz":
        try:
            scene = file[:-7]
            print(scene)
    
            os.mkdir(filepath+scene)
    
            # untar files
            os.system("tar -xvf ./"+ file+ " -C " +filepath+scene)
            os.system("rm ./" + file)
        except:
            print(scene, " already there")
            
            os.system("rm ./" + file)