#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 07:19:39 2020

@author: tkpci
"""

import gdal
from dataLandsatASTER import DataLandsatASTER
import numpy as np
import pdb
import createSiteInfo
import subprocess
import re 
import os
import matplotlib.pyplot as plt



B10_L1_name = '/dirs/data/tirs/downloads/pathrow_022036/LC08_L1GT_022036_20131031_20170308_01_A2/LC08_L1GT_022036_20131031_20170308_01_T2_B10.TIF'
B10_L2_name = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_P022_R036/LC80220362013304LGN01/LC08_L1GT_022036_20131031_20200708_02_T2_B10.TIF'

B10_L1_name = '/dirs/data/tirs/downloads/pathrow_022036/LC08_L1TP_022036_20130329_20170310_01_A1/LC08_L1TP_022036_20130329_20170310_01_T1_B10.TIF'
B10_L2_name = '/dirs/data/tirs/downloads/Surfrad_level2_downloads/Surfrad_P022_R036/LC80220362013088LGN02/LC08_L1TP_022036_20130329_20200708_02_T1_B10.TIF'

B10_L1_name = '/dirs/data/tirs/downloads/pathrow_035026/LC08_L1TP_035026_20130326_20170310_01_A1/LC08_L1TP_035026_20130326_20170310_01_T1_B10.TIF'
B10_L2_name = '/dirs/data/tirs/downloads/Surfrad_P035_R026/LC80350262013085LGN03/LC08_L1TP_035026_20130326_20200715_02_T1_B10.TIF'

B10_L1_name = '/dirs/data/tirs/downloads/pathrow_035026/LC08_L1TP_035026_20130924_20170308_01_A1/LC08_L1TP_035026_20130924_20170308_01_T1_B10.TIF'
B10_L2_name = '/dirs/data/tirs/downloads/Surfrad_P035_R026/LC80350262013267LGN01/LC08_L1TP_035026_20130924_20200715_02_T1_B10.TIF'

B10_L1_name = '/dirs/data/tirs/downloads/Buoy_level2_downloads/Lake_Tahoe/LC80430332018161LGN00/LC08_L2SP_043033_20180610_20200718_02_T1_ST_CDIST.TIF'
B10_L2_name = '/dirs/data/tirs/downloads/Buoy_level2_downloads/Lake_Tahoe/LC80430332018161LGN00/LC08_L2SP_043033_20180610_20200718_02_T1_ST_QA.TIF'
bqaName = '/dirs/data/tirs/downloads/Buoy_level2_downloads/Lake_Tahoe/LC80430332018161LGN00/LC08_L1TP_043033_20180610_20180615_01_T1_BQA.TIF'


# L1_x = [3563,3624]
# L1_y = [5413,5474]
# L2_x = [3563,3624]
# L2_y = [5413,5474]

gdal.UseExceptions()
rd=gdal.Open(B10_L1_name)
b10_L1 = rd.GetRasterBand(1)
b10_L1_data = b10_L1.ReadAsArray()

# where_are_NaNs = np.isnan(b10_L1_data)
# b10_L1_data[where_are_NaNs] = 0
# b10_L1_data = b10_L1_data*0.00033420+0.1

gdal.UseExceptions()
rd=gdal.Open(B10_L2_name)
b10_L2= rd.GetRasterBand(1)
b10_L2_data = b10_L2.ReadAsArray()

# where_are_NaNs = np.isnan(b10_L2_data)
# b10_L2_data[where_are_NaNs] = 0
# b10_L2_data=b10_L2_data*0.00033420+0.1


gdal.UseExceptions()
rd=gdal.Open(bqaName)
bqa= rd.GetRasterBand(1)
bqa_data = bqa.ReadAsArray()
bqa_data = bqa_data.astype(float)
  
    # https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con 
    
cloud = [2800,2804,2808,2812,6896,6900,6904,6908]

# note cloud = 0, clear = 1 >> this is easier as a mask to apply
for val in cloud:
    bqa_data[bqa_data == val] = -10

bqa_data[bqa_data == 1] = -10    
bqa_data[bqa_data >= 0] = 1
bqa_data[bqa_data < 1] = 0


# add gain and bias

#b10_L1_data_GB = b10_L1_data * 1.0151 - 0.14774

plt.subplot(1,3,1)
plt.imshow(b10_L1_data*0.01, vmin = 0, vmax = 10, cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('Single Channel Distance to Cloud [Km]')



plt.subplot(1,3,2)
plt.imshow(b10_L2_data*0.01, vmin = 0,cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('Single Channel QA [K]')

plt.subplot(1,3,3)
plt.imshow(bqa_data,cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('Cloud Mask (confident cloud)')


plt.subplot(2,3,1)
plt.imshow(b10_L1_data,  vmin = 7, vmax = 10, cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('B10 Level 1 radiance')

plt.subplot(2,3,2)
plt.imshow(b10_L2_data, vmin =7,vmax = 10, cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('B10 Level 2 radiance')

plt.subplot(2,3,3)
plt.imshow(b10_L1_data_GB,  vmin =7,vmax = 10, cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('B10 Level 1 radiance with gain and bias')

plt.subplot(2,3,4)
plt.imshow(b10_L1_data -b10_L2_data, vmin =0,vmax = 0.04, cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('Difference: Level 1 - Level 2')

plt.subplot(2,3,5)
plt.imshow(b10_L1_data_GB -b10_L2_data,vmin =-0.01,vmax = 0.04, cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('Difference: Level 1 (GB) - Level 2')

plt.subplot(2,3,6)
plt.imshow(b10_L1_data -b10_L1_data_GB,cmap = "gray")
plt.colorbar()
plt.axis("off")
plt.title('Difference: Level 1 - Level 1 (GB)')



diff_12 = b10_L1_data -b10_L2_data
mean = np.mean(diff_12)
std = np.std(diff_12)

diff_12 = b10_L1_data_GB -b10_L2_data
mean = np.mean(diff_12)
std = np.std(diff_12)

plt.hist(diff_12.flatten(), bins = 50)


mean_L1 = np.mean(b10_L1_data[L1_y[0]:L1_y[1],L1_x[0]:L1_x[1]])
mean_L1_GB = np.mean(b10_L1_data_GB[L1_y[0]:L1_y[1],L1_x[0]:L1_x[1]])

mean_L2 = np.mean(b10_L2_data[L2_y[0]:L2_y[1],L2_x[0]:L2_x[1]])

std_L1 = np.std(b10_L1_data[L1_y[0]:L1_y[1],L1_x[0]:L1_x[1]])
std_L1_GB = np.std(b10_L1_data_GB[L1_y[0]:L1_y[1],L1_x[0]:L1_x[1]])

std_L2 = np.std(b10_L2_data[L2_y[0]:L2_y[1],L2_x[0]:L2_x[1]])










