#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 08:05:47 2022

@author: tkpci
"""

import os
from osgeo import gdal, gdalconst
import numpy as np
import matplotlib.pyplot as plt
import time


def open_landsat(filename):
      
    gdal.UseExceptions()
    rd=gdal.Open(filename)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    img = data.ReadAsArray()
    
    return img

def georegisterImages(img1_filename, img2_filename, output_filename):
    
    #img1_filename = geotif image which resolution you want to keep
    #img2_filename = geotif image to be resampled and matched to img1 - this will be the output image
    
    # Source
    src_filename = img2_filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    #src.GetMetadata()
    bands = src.RasterCount
    
    
    
    # We want a section of source that matches this:
    match_filename = img1_filename
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    
    try:
        
        dst_filename = output_filename
              
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, bands, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection( match_proj)
        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Cubic)
        
        del dst # Flush
        print('  File stacked and saved')
        
    except:
        print('  Oops, something went wrong wile stacking the images. Make sure input images are GEOtifs, and that they overlap on a map')
        
# img1_filename ='/dirs/data/tirs/KBR/LC08_L1TP_014035_20211114_20211125_02_T1/LC08_L1TP_014035_20211114_20211125_02_T1_SW_LST.TIF'       
# img2_filename = '/dirs/data/tirs/KBR/resampled_to_30m/LC80140352021318LGN00_SW_LST_90m_to_30m.TIF'
# output_filename = '/dirs/data/tirs/KBR/resampled_to_30m/LC80140352021318LGN00_SW_LST_90m_to_30m_resampled.TIF'
# georegisterImages(img1_filename, img2_filename, output_filename)   



file30 =  'LC09_L1TP_014035_20211114_20220119_02_T1'#'LC08_L1TP_014035_20211114_20211125_02_T1', 'LC09_L1TP_037037_20211115_20220119_02_T1'
filename =  'LC90140352021318LGN03' #'LC80140352021318LGN00', 'LC90370372021319LGN04'
filename_sc =  'LC09_L2SP_014035_20211114_20220119_02_T1_ST_B10.TIF' #LC08_L2SP_014035_20211114_20211125_02_T1_ST_B10.TIF'

dist = ['90m','100m','120m','150m']
filepath = '/dirs/data/tirs/KBR/resampled_to_30m/'
# sw nominal
filepath_30m = '/dirs/data/tirs/KBR/SW/'
SW = open_landsat(filepath_30m  + file30 + '/' + file30 + '_SW_LST.TIF')
SW_no_filter = open_landsat(filepath_30m + file30 + '/' + file30 + '_SW_LST_no_filter.TIF')

filepath_sc = '/dirs/data/tirs/KBR/SC/'

SC = open_landsat(filepath_sc + filename + '/' + filename_sc)
SC = SC.astype(float)* 0.00341802 + 149
SC[SC == 149] = np.nan

plt.imshow(SW_no_filter, vmin = 275, vmax = 295, cmap = 'jet')
plt.colorbar()
plt.axis('off')
plt.title('SW without filter applied')



y1 = 5180 #5113
x1 = 6579 #5868
y2 = y1+100#5210
x2 = x1+100#5980

# y1 = 0
# x1 = 0
# y2 = SC.shape[0]
# x2 = SC.shape[1]

vmin_ = 0
vmax_ = 1.5


i=0
for val in dist:
    i +=1
    file = filename + '_SW_LST_'+ val + '_to_30m_resampled.TIF'
    img = open_landsat(filepath+file)
    plt.subplot(2,3,i)
    diff = SC - img
    # remove outliers
    diff[diff>5] = np.nan
    diff[diff< -5] = np.nan
    #plt.imshow(img, vmin = 270, vmax = 340)
    plt.imshow(diff[y1:y2,x1:x2], vmin = vmin_, vmax=vmax_, cmap = 'jet')
    cbar = plt.colorbar()
    cbar.set_label('Kelvin difference', rotation=90)
    stat =  np.nanstd(diff[y1:y2,x1:x2].flatten())
    plt.title(val + ': stdev: '+str(np.round(stat,2))+' [K]')
    plt.tight_layout
    plt.axis('off')
    

plt.subplot(2,3,5)
diff = SC-SW
# remove outliers
diff[diff>5] = np.nan
diff[diff< -5] = np.nan
#plt.imshow(img, vmin = 270, vmax = 340)
plt.imshow(diff[y1:y2,x1:x2], vmin = vmin_, vmax=vmax_, cmap = 'jet')
cbar = plt.colorbar()
cbar.set_label('Kelvin difference', rotation=90)
stat =  np.nanstd(diff[y1:y2,x1:x2].flatten())
plt.title('SW: stdev: '+str(np.round(stat,2))+' [K]')
plt.tight_layout
plt.axis('off')

plt.subplot(2,3,6)
diff = SC-SW_no_filter
# remove outliers
diff[diff>5] = np.nan
diff[diff< -5] = np.nan
#plt.imshow(img, vmin = 270, vmax = 340)
plt.imshow(diff[y1:y2,x1:x2], vmin = vmin_, vmax=vmax_, cmap = 'jet')
cbar = plt.colorbar()
cbar.set_label('Kelvin difference', rotation=90)
stat =  np.nanstd(diff[y1:y2,x1:x2].flatten())
plt.title('SW (no filter): stdev: '+str(np.round(stat,2))+' [K]')
plt.tight_layout
plt.axis('off')



def get_point(img):
   
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
    #coords = []
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.waitforbuttonpress()
    
    ax.scatter(ix,iy, s=20,c='red')
    
    return ix,iy


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))

    # global coords
    # coords[ix, iy]

    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)

    return ix,iy

diff = SW
ix,iy = get_point(diff[y1:y2,x1:x2])


#ax.scatter(coords[1][0],coords[1][1], s=20,c='red')






