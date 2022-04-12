#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:10:54 2020

@author: tkpci
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

def open_csv(filepath):
    
    
    rd = open(filepath)
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    # remove first column and space at end (not sure why its reading that)
    date = data[0,0]
    data = data[:,1:-1]
    # change to float
    data = data.astype(float)
        
    return data, date

def plot_csv_data(data, date, side, stat):
    
    x_val = data[:,0]/70
    
    y_val_mean = np.zeros([data.shape[0],6])
    y_val_std = np.zeros([data.shape[0],6])
    
    increase = 640
    num_means = 11520
    num_array = 1920
    legend = list()
    
    if side == 'A':
        # for array A
        start_col_mean = 2
        start_col_std = 2 + num_means
        
    elif side == 'B':
        # for array B
        start_col_mean = 2 + increase 
        start_col_std = 2 + num_means + increase
        
    elif side == 'C':    
        # for array C
        start_col_mean = 2 + 2* increase 
        start_col_std = 2 + num_means + 2* increase
   
    
    for i in range(6):
        temp = data[:,start_col_mean:start_col_mean+increase]
        temp = np.mean(temp, axis=1)
        y_val_mean[:,i] = temp
        start_col_mean += num_array
        
        temp = data[:,start_col_std:start_col_std+increase]
        temp = np.mean(temp, axis=1)
        y_val_std[:,i] = temp
        start_col_std += num_array
        
        legend.append('row ' + str(i+1))
        
    
    if stat == 'mean':
        y_val = y_val_mean
    elif stat == 'std':
        y_val = y_val_std
    
    plt.plot(x_val,y_val) 
    # not sure what you wanted as title, so just added the date 
    plt.title('Side: ' + side + ' ' + stat + ': ' + date)
    plt.xlabel('seconds')
    plt.ylabel('DN')
    plt.legend(legend , loc=1)
    
    plt.show()
    
    
    
def main():    
  
    filepath = '/dirs/data/tirs/code//mdf_stats.csv'
    data, date = open_csv(filepath) 
    plot_csv_data(data, date, side = 'C', stat = 'std')   # or stat = 'std'
  
    
if __name__ == '__main__':
    main()
    
