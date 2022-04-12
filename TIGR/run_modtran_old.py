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


# run MODTRAN    
def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    
    #command = 'cp /cis/staff2/tkpci/modtran/tape5_geos/' + tape5_name + ' /cis/staff2/tkpci/modtran/tape5/tape5'
    
    #os.system(command) 
    
    command = 'cd /cis/staff2/tkpci/modtran/tape5_predefined\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff2/tkpci/modtran/tape5_predefined/tape6 /cis/staff2/tkpci/modtran/tape6_predefined/' + tape6_name 
    os.system(command) 
    
    command = 'rm /cis/staff2/tkpci/modtran/tape5_predefined/*'
    os.system(command)