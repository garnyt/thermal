#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:56:09 2022

@author: tkpci
"""
import numpy as np
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(diff)

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

    return coords

cid = fig.canvas.mpl_connect('button_press_event', onclick)


ax.scatter(coords[0][0],coords[0][1], s=20,c='red')
ax.scatter(coords[1][0],coords[1][1], s=20,c='red')