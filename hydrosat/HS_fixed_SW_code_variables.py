#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:01:26 2021

@author: tkpci


Create json file for SW coefficients, emissivity parameters and uncertainty values

"""

import json

def SW_TIRS_coefficients_json():
    
    
    variables = {}
    # change variables below to help with work flow
    variables['aster_filepath'] = '/dirs/data/tirs/downloads/aster/E13_E14_NDVI_files/' # directory where downloaded aster tiles are kept
    variables['initialdir'] = "/dirs/data/tirs/downloads/test/"  # directory where popup folder will start to look for landsat files
    variables['autodownloads_Aster'] = '/dirs/data/tirs/downloads/aster/AutoDownloads/' # folder to temporarily download ASTER tiles
    variables['username'] = 'Cisthermal' # login details for Earthdata (https://search.earthdata.nasa.gov/search)
    variables['password'] = 'C15Th3rmal'
    
    
    # dont change the variables below - unless instructed
    variables['TIRS_10'] = {}
    variables['TIRS_11'] = {}
    variables['L8'] = {}
    variables['L9'] = {}
    
    
    # Landsat 8
    # derived SW coefficients and associated regression error
    variables['L8']['SW_coefficients'] = [2.2925,0.9929,0.1545,-0.3122,3.7186,0.3502,-3.5889,0.1825]
    variables['L8']['SW_regfit'] = 0.7212
    
    
    # TIRS_10
    # RIT derived ASTER to TIRS emissivity conversion coefficients based on UCSB emissivity dataset
    variables['TIRS_10']['estimated_coeff_1'] = 0.56789
    variables['TIRS_10']['estimated_coeff_2'] = 0.42249
    variables['TIRS_10']['estimated_coeff_3'] = 0.00990    
    # snow emissivity and vegetation coefficient values - Glynn derived
    variables['TIRS_10']['snow_emis_value'] = 0.9904
    variables['TIRS_10']['vegetation_coeff'] = 0.9885    
    variables['TIRS_10']['water_emissivity'] = 0.988
    # bare soil - as per SC code
    variables['TIRS_10']['bare_soil_variable'] = 0.5
    # uncertainty value for ASTER to landsat emis (RIT derived)
    variables['TIRS_10']['emis_regfit_10'] = 0.000972     
    # landsat uncertainty in apparent temperature - based on paper byt Montanaro et. al.
    variables['TIRS_10']['landsat_uncertainty_10_K'] = 0.15
    
    
    # TIRS_11
    # RIT derived ASTER to TIRS emissivity conversion coefficients based on UCSB emissivity dataset
    variables['TIRS_11']['estimated_coeff_1'] = -0.54148252 
    variables['TIRS_11']['estimated_coeff_2'] = 1.4305094
    variables['TIRS_11']['estimated_coeff_3'] = 0.10916213    
    # snow emissivity and vegetation coefficient values - RIT derived
    variables['TIRS_11']['snow_emis_value'] = 0.9724
    variables['TIRS_11']['vegetation_coeff'] = 0.9700   
    variables['TIRS_11']['water_emissivity'] = 0.986448696154591
    # bare soil - based on papers
    variables['TIRS_11']['bare_soil_variable'] = 0.5
    # uncertainty value for ASTER to landsat emis (RIT derived)
    variables['TIRS_11']['emis_regfit_11'] = 0.00545     
    # landsat uncertainty in apparent temperature - based on paper byt Montanaro et. al.
    variables['TIRS_11']['landsat_uncertainty_11_K'] = 0.20 
    
    
    # Landsat 9
    # derived SW coefficients and associated regression error
    variables['L9']['SW_coefficients'] = [2.14116823,0.99391182,0.15328625,-0.27561351,3.32215426,0.32960457,-2.93095085,0.15652295]
    variables['L9']['SW_regfit'] = 0.74161
    

    
    return variables

def main():
    
    # create dictionary with data coefficients
    variables = SW_TIRS_coefficients_json()
    
    # write to json file
    # Serializing json 
    json_object = json.dumps(variables, indent = 4)
      
    # Writing to sample.json
    filepath = '/cis/staff/tkpci/Code/Python/hydrosat/'
    
    with open(filepath + "fixed_SW_code_variables.json", "w") as outfile:
        outfile.write(json_object)
    
    
    # load data
    with open(filepath + "fixed_SW_code_variables.json", 'r') as openfile:
        # Reading from json file into dictionary
        variables = json.load(openfile)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    