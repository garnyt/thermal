"""
Created on Tue Nov  5 11:29:01 2019

@author: Tania Kleynhans

Connect to server, Run MODTRAN, and get tape6 back

"""


import os
import numpy as np


def runModtran(tape5_name, tape6_name):
    
    print('Modtran run: ',tape5_name)
    
    command = 'cp /cis/staff2/tkpci/modtran/tape5_TES/' + tape5_name + ' /cis/staff2/tkpci/modtran/tape5/tape5'
    
    os.system(command) 
    
    command = 'cd /cis/staff2/tkpci/modtran/tape5\n' \
              'ln -s /dirs/pkg/Mod4v3r1/DATA\n' \
              '/dirs/pkg/Mod4v3r1/Mod4v3r1.exe'
    
    os.system(command) 

    command = 'cp /cis/staff2/tkpci/modtran/tape5/tape6 /cis/staff2/tkpci/modtran/tape6_TES/' + tape6_name 
    os.system(command) 
    
    command = 'rm /cis/staff2/tkpci/modtran/tape5/*'
    os.system(command)
