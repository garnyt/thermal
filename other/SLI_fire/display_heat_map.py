#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:46:08 2020

@author: tkpci
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display_heat_map(tests):
    

    xlabs = tests[:,1]
    xlabs_unique = np.unique(xlabs)
    ylabs = tests[:,2]
    ylabs_unique = np.unique(ylabs)
    ylabs_unique = np.flip(ylabs_unique)
    
    output = np.zeros([ylabs_unique.shape[0],xlabs_unique.shape[0]])
       
    
    for i in range(xlabs_unique.shape[0]):
        for j in range(ylabs_unique.shape[0]):
            x_val = xlabs_unique[i]
            y_val = ylabs_unique[j]
            
            idx = np.where((tests[:,1] == x_val) & (tests[:,2] == y_val))
            
            output[j,i] = np.mean(tests[idx,0])

            
    fig, ax = plt.subplots()
    im = ax.imshow(output, cmap='gray_r')
    
    str1 = [str(e) for e in list((xlabs_unique))]
    #str1 = ['10', '', '10^2', '', '10^3', '', '10^4', '', '10^5'] #[str(e) for e in list((xlabs_unique))]
    str2 = [str(e) for e in list(ylabs_unique.astype(int))]
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(xlabs_unique.shape[0]))
    ax.set_yticks(np.arange(ylabs_unique.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(str1)
    ax.set_yticklabels(str2)
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(str1)):
        for j in range(len(str2)):
            text = ax.text(i, j, int(output[j, i]*100),
                            ha="center", va="center", color="g")
    
    ax.set_ylim(len(str2)-0.5, -0.5)
    ax.set_title("Probability of detection")
    plt.xlabel('Percent of pixel with fire')
    plt.ylabel('Temperature of fire [K]')
    fig.tight_layout()
    
    plt.show()     

         