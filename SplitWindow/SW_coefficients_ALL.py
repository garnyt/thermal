#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:01:26 2021

@author: tkpci


Create json file for SW coefficients, emissivity parameters and uncertainty values

"""

import json

def SW_TIRS_coefficients_json():
    
    data = {}
    
    data['aster_filepath'] = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/' # directory where downloaded aster tiles are kept
    data['initialdir'] = "/dirs/data/tirs/downloads/test/"  # directory where popup folder will start to look for landsat files
    data['autodownloads_Aster'] = '/dirs/data/tirs/downloads/aster/AutoDownloads/' # folder to temporarily download ASTER tiles
    data['username'] = 'Cisthermal' # login details for Earthdata (https://search.earthdata.nasa.gov/search)
    data['password'] = 'C15Th3rmal'
  
    data['TIRS_10'] = {}
    data['TIRS_11'] = {}
    data['TIRS2_10'] = {}
    data['TIRS2_11'] = {}
    data['L8'] = {}
    data['L9'] = {}
    
    
    # Landsat 8
    # derived SW coefficients and associated regression error
    data['L8']['SW_coefficients'] = [2.2925,0.9929,0.1545,-0.3122,3.7186,0.3502,-3.5889,0.1825]
    data['L8']['SW_regfit'] = 0.7212
    
    # uncertainty coefficients
    # correlation coefficient for apparent temperature and emissivity
    # this was calculated using the 113 MODIS emissivities (spectrally sampled)
    # the appTemp correlation was calculated using the TIGR simulation data
    data['L8']['corr_emis'] = 0.7
    data['L8']['corr_appTemp'] = 1
    
    
    # TIRS_10
    # RIT derived ASTER to TIRS emissivity conversion coefficients based on UCSB emissivity dataset
    data['TIRS_10']['estimated_coeff_1'] = 0.56789
    data['TIRS_10']['estimated_coeff_2'] = 0.42249
    data['TIRS_10']['estimated_coeff_3'] = 0.00990    
    # snow emissivity and vegetation coefficient values - Glynn derived
    data['TIRS_10']['snow_emis_value'] = 0.9904
    data['TIRS_10']['vegetation_coeff'] = 0.9885    
    data['TIRS_10']['water_emissivity'] = 0.988
    # bare soil - as per SC code
    data['TIRS_10']['bare_soil_variable'] = 0.5
    # uncertainty value for ASTER to landsat emis (RIT derived)
    data['TIRS_10']['emis_regfit_10'] = 0.000972     
    # landsat uncertainty in apparent temperature - based on paper by Montanaro et. al.
    data['TIRS_10']['landsat_uncertainty_10_K'] = 0.15
    
    
   
    
    
    
    # TIRS_11
    # RIT derived ASTER to TIRS emissivity conversion coefficients based on UCSB emissivity dataset
    data['TIRS_11']['estimated_coeff_1'] = -0.54148252 
    data['TIRS_11']['estimated_coeff_2'] = 1.4305094
    data['TIRS_11']['estimated_coeff_3'] = 0.10916213    
    # snow emissivity and vegetation coefficient values - RIT derived
    data['TIRS_11']['snow_emis_value'] = 0.9724
    data['TIRS_11']['vegetation_coeff'] = 0.9700   
    data['TIRS_11']['water_emissivity'] = 0.986448696154591
    # bare soil - as per SC code
    data['TIRS_11']['bare_soil_variable'] = 0.5
    # uncertainty value for ASTER to landsat emis (RIT derived)
    data['TIRS_11']['emis_regfit_11'] = 0.00545     
    # landsat uncertainty in apparent temperature - based on paper by Montanaro et. al.
    data['TIRS_11']['landsat_uncertainty_11_K'] = 0.20 
    
    # uncertainty coefficients
    
    
    # Landsat 9
    # derived SW coefficients and associated regression error
    data['L9']['SW_coefficients'] = [2.14116823,0.99391182,0.15328625,-0.27561351,3.32215426,0.32960457,-2.93095085,0.15652295]
    data['L9']['SW_regfit'] = 0.74161
    
    
    # TIRS2_10
    # RIT derived ASTER to TIRS emissivity conversion coefficients based on UCSB emissivity dataset
    data['TIRS2_10']['estimated_coeff_1'] = 0.6805076
    data['TIRS2_10']['estimated_coeff_2'] = 0.31538733
    data['TIRS2_10']['estimated_coeff_3'] = 0.00439255
    # snow emissivity and vegetation coefficient values - RIT derived
    data['TIRS2_10']['snow_emis_value'] = 0.991146
    data['TIRS2_10']['vegetation_coeff'] = 0.9889 # Ray value based on TIRS
    data['TIRS2_10']['water_emissivity'] = 0.988 #0.992921523229335
    # bare soil - as per SC code
    data['TIRS2_10']['bare_soil_variable'] = 0.5
    # uncertainty value for ASTER to landsat emis (RIT derived)
    data['TIRS2_10']['emis_regfit_10'] = 0.000594
    # landsat uncertainty in apparent temperature - based on paper byt Montanaro et. al.
    data['TIRS2_10']['landsat_uncertainty_10_K'] = 0.1  #estimated - waiting for more info
    
    
    # TIRS2_11
    # RIT derived ASTER to TIRS emissivity conversion coefficients based on UCSB emissivity dataset
    data['TIRS2_11']['estimated_coeff_1'] = -0.58257351
    data['TIRS2_11']['estimated_coeff_2'] = 1.46527441
    data['TIRS2_11']['estimated_coeff_3'] = 0.11562234
    # snow emissivity and vegetation coefficient values - RIT derived
    data['TIRS2_11']['snow_emis_value'] = 0.972072
    data['TIRS2_11']['vegetation_coeff'] = 0.9700# same as TIRS for test      0.9909 # Ray value
    data['TIRS2_11']['water_emissivity'] = 0.9863197349099047
    # bare soil - as per SC code
    data['TIRS2_11']['bare_soil_variable'] = 0.5
    # uncertainty value for ASTER to landsat emis (RIT derived)
    data['TIRS2_11']['emis_regfit_11'] = 0.005532
    # landsat uncertainty in apparent temperature 
    data['TIRS2_11']['landsat_uncertainty_11_K'] = 0.1  #estimated - waiting for more info
    
    return data

def main():
    
    # create dictionary with data coefficients
    data = SW_TIRS_coefficients_json()
    
    # write to json file
    # Serializing json 
    json_object = json.dumps(data, indent = 4)
      
    # Writing to sample.json
    
    
    with open("SW_coefficients_ALL.json", "w") as outfile:
        outfile.write(json_object)
    
    
    # load data
    #with open("SW_coefficients_ALL.json", 'r') as openfile:
        # Reading from json file into dictionary
     #   data = json.load(openfile)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    