#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:46:08 2020

@author: tkpci
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_heat_map():
    
    data = pd.read_csv('/cis/staff2/tkpci/Code/Python/TIGR/SW_RSR_RECT_shift_multiprocess.csv')
    
    b10_widen = pd.Series.to_numpy(data['band10'])
    b11_widen = pd.Series.to_numpy(data['band11'])
    rmse = pd.Series.to_numpy(data['rmse_new_coeff'])
    rmse = pd.Series.to_numpy(data['rmse_orig_coeff'])
    ranges = pd.Series.to_numpy(data['band10'])
    range_unique = np.unique(ranges)
    
    output = np.zeros([range_unique.shape[0],range_unique.shape[0]])
       
    
    for i in range(range_unique.shape[0]):
        for j in range(range_unique.shape[0]):
            b10_val = range_unique[i]
            b11_val = range_unique[j]
            
            idx = data.index[(data['band10']==b10_val) & (data['band11']==b11_val)]
            
            output[i,j] = rmse[idx[0]]

            
    fig, ax = plt.subplots()
    im = ax.imshow(output)
    
    str1 = [str(e) for e in list(np.round(range_unique,1))]
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(range_unique.shape[0]))
    ax.set_yticks(np.arange(range_unique.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(str1)
    ax.set_yticklabels(str1)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(str1)):
        for j in range(len(str1)):
            text = ax.text(j, i, output[i, j],
                           ha="center", va="center", color="w")
    
    ax.set_ylim(len(str1)-0.5, -0.5)
    #ax.set_title("Shift TIRS bands (new coeff) - RMSE [K]")
    ax.set_title("Shift TIRS bands - RMSE [K]")
    plt.xlabel('Shift band 11 [um]')
    plt.ylabel('Shift band 10 [um]')
    fig.tight_layout()
    plt.show()     


create_heat_map()            