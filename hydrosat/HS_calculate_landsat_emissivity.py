"""
Created on Oct 12 2021

@author: Tania Kleynhans

PURPOSE: Estimates a Landsat Emissivity product from ASTER Emissivity and
         NDVI.  The results are meant to be used for generation of a
         Surface Temperature product.
"""

import numpy as np
import os
import warnings
import pdb


def main(dataOut):
    
    warnings.simplefilter('ignore')
    
    # create dictionary to hold temp files for emis calculation
    temp_dataOut = {}

    # create mask 
    temp_dataOut['mask'] = np.where(dataOut['rad10'] == 0)
    # add gain and bias to all data
    dataOut, temp_dataOut = convertGainBias(dataOut,temp_dataOut)

    # calculate NDVI and NDSI
    temp_dataOut  = generate_landsat_NDVI_NDSI(dataOut,temp_dataOut)
    
    # convert ASTER emissivity to Landsat emissivity
    temp_dataOut['ls_emis_data'] = convert_emis_from_ASTER(dataOut, 'TIRS_10')
    # calculate emssivity
    dataOut['emis10'] = calculate_emissivity(dataOut, temp_dataOut, 'TIRS_10')
    dataOut['emis10'][temp_dataOut['mask']] = 0
    
    
    temp_dataOut['ls_emis_data'] = convert_emis_from_ASTER(dataOut, 'TIRS_11')
    dataOut['emis11'] = calculate_emissivity(dataOut, temp_dataOut, 'TIRS_11')
    dataOut['emis11'][temp_dataOut['mask']] = 0
    
    return dataOut


       
# calculate NDVi and NDSI for sensor to adjust ASTER emissivity
def generate_landsat_NDVI_NDSI(dataOut, temp_dataOut):
    
    ndvi = (dataOut['rad5'] - dataOut['rad4'])/(dataOut['rad5'] + dataOut['rad4'])
    
    where_are_NaNs = np.isnan(ndvi)
    ndvi[where_are_NaNs] = 0
    
    ndvi[temp_dataOut['mask']]=0
    ndvi[ndvi < 0.0000001] = 0
    ndvi[ndvi > 1] = 1
    # note, NDVI data is normalized but not stretched as paper suggests. Awaiting feedback 
    temp_dataOut['ndvi'] = ndvi/np.max(ndvi)
    
    temp_dataOut['ndsi'] = (dataOut['rad3'] - dataOut['rad6'])/(dataOut['rad3'] + dataOut['rad6'])
    temp_dataOut['ndsi'][temp_dataOut['mask']]=0
    temp_dataOut['snow_locations'] = np.where(temp_dataOut['ndsi'] > 0.4)
    
    return temp_dataOut
    

# convert ASTER emissivity to sensor emissivity    
def convert_emis_from_ASTER(dataOut, sensor):
    
    ls_emis_data = (dataOut['variables'][sensor]['estimated_coeff_1'] * dataOut['emis13'] +
                 dataOut['variables'][sensor]['estimated_coeff_2'] * dataOut['emis14'] +
                 dataOut['variables'][sensor]['estimated_coeff_3'])
    
    return ls_emis_data


# convert DC to radiance and scale ASTER data
def convertGainBias(dataOut, temp_dataOut):
    

    if dataOut['SR']: # Collection 2 level 2 data
        dataOut['rad3'] = dataOut['rad3'] * 0.0000275-0.2      # convert SR_DC to reflectance
        dataOut['rad4'] = dataOut['rad4'] * 0.0000275-0.2        # convert to reflectance
        dataOut['rad5'] = dataOut['rad5'] * 0.0000275-0.2        # convert to reflectance
        dataOut['rad6'] = dataOut['rad6'] * 0.0000275-0.2        # convert to reflectance
        SR_check = 1
           
    else:
        # convert to reflectance
        dataOut['rad3'] = dataOut['rad3']* 0.00002-0.1      # convert to reflectance
        dataOut['rad4'] = dataOut['rad4']* 0.00002-0.1      # convert to reflectance
        dataOut['rad5'] = dataOut['rad5']* 0.00002-0.1      # convert to reflectance
        dataOut['rad6'] = dataOut['rad6']* 0.00002-0.1      # convert to reflectance
    
    # make some aster NDVI adjustments
    temp = dataOut['aster_ndvi']
    temp[temp_dataOut['mask']]=0
    temp[temp < 0.0000001] = 0
    temp[temp > 1] = 1
    where_are_NaNs = np.isnan(temp)
    temp[where_are_NaNs] = 0
    temp = temp/np.max(temp)    # ASTER data normalized since ASTER NDVI was calculated from uncorrected toa reflectances (Glynn email)
    
    dataOut['aster_ndvi'] = temp
    
    return dataOut, temp_dataOut
    
    
def calculate_emissivity(dataOut, temp_dataOut, sensor):
    
    water_locations = np.where(temp_dataOut['ls_emis_data'] > dataOut['variables'][sensor]['water_emissivity'])
    
    # Get pixels with significant bare soil component 
    bare_locations = np.where(dataOut['aster_ndvi'] < dataOut['variables'][sensor]['bare_soil_variable'])
    
    # note 0.975 is the ASTER spectral response for bands 13/14 for vegetation (Glynn email)                    
    ls_emis_bare = (temp_dataOut['ls_emis_data'][bare_locations] \
        - 0.975 * dataOut['aster_ndvi'][bare_locations]) \
        / (1 - dataOut['aster_ndvi'][bare_locations])
    
    ls_emis_final = (dataOut['variables'][sensor]['vegetation_coeff'] * temp_dataOut['ndvi'] +
                     temp_dataOut['ls_emis_data'] * (1.0 - temp_dataOut['ndvi']))
    
    ls_emis_final[bare_locations] = ls_emis_bare
    ls_emis_final[temp_dataOut['snow_locations']] = dataOut['variables'][sensor]['snow_emis_value']
    
    ls_emis_final[ls_emis_final > 1] = dataOut['variables'][sensor]['water_emissivity']
    ls_emis_final[water_locations] = dataOut['variables'][sensor]['water_emissivity']
    
    return ls_emis_final
    
    
    
    
    
    
    
    
    
