#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:46:08 2020

@author: tkpci
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pdb


def create_heat_map():
    
    filepath = '/cis/staff2/tkpci/Code/Python/TIGR/multiprocess_out/'
    
    rsr_shape = 'TIRS'
    move = 'Shift'
    #move = 'Widen'
    name_all = ['orig_coeff','orig_coeff_error', 'new_coeff', 'new_coeff_error']
    fsize = 15
    tsize = 10
    
    #name_all = ['orig_coeff', 'new_coeff']

    for ii in range(len(name_all)):
            
        name = name_all[ii]
        
        b10_widen = []
        b11_widen = []
        rmse = []
        
        for file in listdir(filepath):
            if file[-3:] == 'csv':
                data = pd.read_csv(filepath + file)
                
                b10_widen.append(data['band10'][0])
                b11_widen.append(data['band11'][0])
                rmse.append(data['rmse_'+name][0])
    
        ranges = b10_widen
        range_unique = np.unique(ranges) 
           
        
        output = np.zeros([range_unique.shape[0],range_unique.shape[0]])
           
        
        for i in range(range_unique.shape[0]):
            for j in range(range_unique.shape[0]):
                b10_val = range_unique[i]
                b11_val = range_unique[j]
                
                idx = np.where((b10_widen==b10_val) & (b11_widen==b11_val))
                
                #pdb.set_trace()
                
                output[i,j] = rmse[int(idx[0])]
    
        if 'orig' in name:
            colorbar_max = 10
        else:
            if 'error' in name:
                colorbar_max = 3
            else:
                colorbar_max = 2
                
        fig, ax = plt.subplots()
        #im = ax.imshow(output, vmin = 1, vmax = 10) #, cmap = 'coolwarm'
        im = ax.imshow(output, vmin = 0.67, vmax = colorbar_max)
        fig.colorbar(im)
    
        if move == 'Shift':
            str10 = [str(e) for e in list(np.round(range_unique,1))]
            str11 = [str(e) for e in list(np.round(range_unique,1))]
    #        str10 = [str(e) for e in list(np.round(range_unique+ 0.75,2))]
    #        str11 = [str(e) for e in list(np.round(range_unique+ 0.75,2))]
        else:
            str10 = [str(e) for e in list(np.round(range_unique,2))]
            str11 = [str(e) for e in list(np.round(range_unique,2))]
            #str10 = [str(e) for e in list(np.round(range_unique+ 0.59,2))]
            #str11 = [str(e) for e in list(np.round(range_unique+ 1.01,2))]
        
        # We want to show all ticks...
        ax.set_xticks(np.arange(range_unique.shape[0]))
        ax.set_yticks(np.arange(range_unique.shape[0]))
        # ... and label them with the respective list entries

        ax.set_xticklabels(str10, fontsize = tsize)
        ax.set_yticklabels(str11, fontsize = tsize)
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(str10)):
            for j in range(len(str11)):
                text = ax.text(j, i, output[i, j],
                               ha="center", va="center", color="w")
        
        ax.set_ylim(len(str10)-0.5, -0.5)

        #ax.set_title(move + ' ' + rsr_shape + ' bands ('+ name + ') RMSE [K]')
        plt.xlabel(move +' band at 9.75 [$\mu$m]', fontsize = fsize)
        plt.ylabel(move +' band 10 [$\mu$m]', fontsize = fsize)    
    
        #plt.xlabel('Widen band 11 [um]')
        #plt.ylabel('Shift band 11 [um]')
        
    #    if move == 'Shift':
    #        plt.xlabel('Band 10 bandcenter [um]')
    #        plt.ylabel('Band 11 bandcenter [um]')
    #    else:
    #        plt.xlabel('Band 10 FWHM [um]')
    #        plt.ylabel('Band 11 FWHM [um]')
        
        fig.tight_layout()
        plt.show()     


create_heat_map()
    
    
           