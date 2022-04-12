#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:13:08 2021

@author: tkpci
"""

# calculate SW with various emis coefficients


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = '/cis/staff/tkpci/Code/Python/All3_data_collection2_raw.csv'
data = pd.read_csv(filename)
coeff = [2.2925,0.9929,0.1545,-0.3122,3.7186,0.3502,-3.5889,0.1825]

#data.set_index("SR_sitename", inplace = True)

data['emis10_fixed'] = np.zeros(len(data['L8_emis10']))
data['emis11_fixed'] = np.zeros(len(data['L8_emis10']))

# plt.hist(data['L8_emis10'] - data['L8_emis11'], bins=20)
# plt.xlabel('Difference between band 10 and band 11 emissivity')
# plt.title('Surfrad site emissivity differences (B10/B11)')
# plt.ylabel('# of datapoints')

veg_emis_10 = 0.9580
veg_emis_11 = 0.9584
soil_emis_10 = 0.9562
soil_emis_11 = 0.9680

cnt = np.arange(-0.002,0.03,0.01)

out = pd.DataFrame()
out['Cloud Bins'] = ['0 km','0 to 1 km','1 to 5 km','5 to 10 km','10+ km ']
out.set_index("Cloud Bins", inplace = True)

temp = data.groupby(['cloud_bin'])['SRlessL8_LST'].std()
kwargs = {'nominal_emis' : lambda x: temp}
out = out.assign(**kwargs)

for i in range(cnt.shape[0]):

    veg_emis_11 = veg_emis_10 + cnt[i]
    soil_emis_11 = soil_emis_10 + cnt[i]
    
    data['emis10_fixed'] = np.where(data['SR_sitename'] == 'Fort_Peck_MT', veg_emis_10,data['emis10_fixed'] )
    data['emis11_fixed'] = np.where(data['SR_sitename'] == 'Fort_Peck_MT', veg_emis_11,data['emis11_fixed'] )
    
    data['emis10_fixed'] = np.where(data['SR_sitename'] == 'Desert_Rock_NV', soil_emis_10,data['emis10_fixed'] )
    data['emis11_fixed'] = np.where(data['SR_sitename'] == 'Desert_Rock_NV', soil_emis_11,data['emis11_fixed'] )
    
    data['emis10_fixed'] = np.where(data['SR_sitename'] == 'Goodwin_Creek_MS', veg_emis_10,data['emis10_fixed'] )
    data['emis11_fixed'] = np.where(data['SR_sitename'] == 'Goodwin_Creek_MS', veg_emis_11,data['emis10_fixed'] )
        
    # calculate terms for split window algorithm
    T_diff = (data['L8_AppTemp10'] - data['L8_AppTemp11'])/2
    T_plus =  (data['L8_AppTemp10'] + data['L8_AppTemp11'])/2
    e_mean = (data['emis10_fixed'] + data['emis11_fixed'])/2
    e_diff = (1-e_mean)/(e_mean)
    e_change = (data['emis10_fixed']-data['emis11_fixed'])/(e_mean**2)
    quad = (data['L8_AppTemp10']-data['L8_AppTemp11'])**2
    
    # calculate split window LST  
    splitwindowLST = coeff[0] + coeff[1]*T_plus+ coeff[2]*T_plus*e_diff + coeff[3]*T_plus*e_change + \
        coeff[4]*T_diff + coeff[5]*T_diff*e_diff + coeff[6]*T_diff*e_change + coeff[7]*quad
        
    data['SW_fixed_emis'] = splitwindowLST
    
    data['diff']  = data['Surfrad_LST'] - data['SW_fixed_emis']
    
    temp = data.groupby(['cloud_bin'])['diff'].std()
    
    name = '(e10+' + str(np.round(cnt[i],3))+')'
    kwargs = {name : lambda x: temp}
    out = out.assign(**kwargs)
    #out.assign(B10_emis= lambda x: temp)



#out.to_numpy()
#data.groupby(['cloud_bin']).size()



#data.to_csv('/cis/staff/tkpci/Code/Python/All data_collection2_raw_fixed_emis_.csv')