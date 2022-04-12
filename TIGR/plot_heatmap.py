"""
Created on Mon Apr  6 09:46:08 2020

@author: tkpci
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_heat_map():
    
    # read in text file
    filepath = '/cis/staff2/tkpci/data/'
    filename = 'ResizedSDShiftArray10-12.txt'
    data = np.loadtxt(filepath + filename)
    
    # reshape data
    data = np.transpose(data)
    data = np.reshape(data, [26,9,7])
    
    # create labels for axis
    y_range = np.round(np.arange(0.2,1.1,0.1),1) 
    x_range = np.round(np.arange(10,12.6,0.1),1)
    
    output = np.transpose(data[:,:,0])
    
    # create figure                
    fig, ax = plt.subplots()
    # plot axis labels
    plt.xlabel('Band Center [um]', fontsize=10)
    plt.ylabel('FWHM [um]', fontsize=10)
    
    # display figure - set display limits
    im = ax.imshow(output, vmin = np.min(output), vmax = np.max(output))
    
    # create an axes on the right side of ax for colorbar. The width of cax will be 2%
    # of ax and the padding between cax and ax will be fixed at 0.1 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(im, cax=cax)
    plt.ylabel('Root Mean Squared Error [K]', fontsize=10)
    plt.tick_params(labelsize=8)
    
    # create tick labels
    strx = [str(e) for e in list(x_range)]
    stry = [str(e) for e in list(np.round(y_range,1))]
  
    # To show all ticks...
    ax.set_xticks(np.arange(x_range.shape[0]))
    ax.set_yticks(np.arange(y_range.shape[0]))
    
    # and label them with the respective list entries
    ax.set_xticks(np.arange(x_range.shape[0]))
    ax.set_xticklabels(strx)
    ax.set_yticklabels(stry)
    ax.tick_params(labelsize=8)
    
    
    # Loop over data dimensions and create text annotations.
    # uncomment for values within each block
#    for i in range(len(stry)):
#        for j in range(len(strx)):
#            text = ax.text(j, i, np.round(output[i, j],1),
#                           ha="center", va="center", color="w")
#    
    
    
    # set axis limits that it does not start in half blocks
    ax.set_ylim(len(stry)-0.5, -0.4)  # for 9 blocks
  
    fig.tight_layout()
    plt.show()     


create_heat_map()            