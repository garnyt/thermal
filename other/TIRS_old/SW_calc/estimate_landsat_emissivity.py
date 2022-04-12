"""
Created on Fri Nov 15 11:52:01 2019

@author: Tania Kleynhans

PURPOSE: Estimates a Landsat Emissivity product from ASTER Emissivity and
         NDVI.  The results are meant to be used for generation of a
         Surface Temperature product.
"""

import numpy as np
import pdb

def estimate_landsat_emissivity(dataLandsatASTER,MTL,sensor, convert):

    # add gain and bias to all data
    mask = np.where(dataLandsatASTER.getRad('rad10') == 0)
    if convert == 'yes':
        dataLandsatASTER = convertGainBias(dataLandsatASTER,mask,MTL)

    # calculate NDVI and NDSI
    ndvi, ndsi, snow_locations = generate_landsat_NDVI_NDSI(dataLandsatASTER,mask)
    
    # convert ASTER emissivity to Landsat emissivity
    ls_emis_data = convert_emis_from_ASTER(dataLandsatASTER, sensor)
    
    pdb.set_trace()
    
    # calculate emssivity
    emis = calculate_emissivity(dataLandsatASTER, ls_emis_data, ndvi, ndsi, snow_locations, sensor)
    
    emis[mask] = 0
    
    return emis



def sensor_coefficients(sensor):
    
    # upload sensor coefficients
    if sensor == 'TIRS10':
        coefficients={'estimated_1':0.56472,                         #ASTER13:   Tania values   Aaron coeff = 0.56789
                               'estimated_2':0.42545,                #ASTER14:   Tania values   Aaron Coeff = 0.42249
                               'estimated_3':00.010103,                #intercept: Tania values   Aaron Coeff = 0.00990
                               'snow_emissivity':0.9904,            #USGS code values - get same value when spectrally sampling
                               'vegetation_coeff':0.9885,           #USGS code values
                               'aster_water':0.988,                 #USGS code values - Tania value 0.993050066587395
                               'bare_soil_variable':0.1}            #USGS uses 0.5 - according to literature it should be 0.1
    elif sensor == 'TIRS11':
        coefficients={'estimated_1':-0.55987,                     #ASTER13: Tania values      Aaron coeff = -0.54148252 
                               'estimated_2':1.4465,             #ASTER14: Tania values       Aaron Coeff = 1.4305094
                               'estimated_3':0.1117,            #intercept: Tania values        Aaron Coeff = 0.10916213
                               'snow_emissivity':0.9724,            #Spectrally sampled to B11 RSR (Tania)
                               'vegetation_coeff':0.9700,           #Value randomly selected
                               'aster_water':0.986448696154591,   #Tania value spectrally sampled   
                               'bare_soil_variable':0.1}            #USGS uses 0.5 - according to literature it should be 0.1
    else:
        print('Sensor not in list. Please enter correct sensor. Currently only TIRS10 or TIRS11 works.')
        return
    
    return coefficients
 
       
# calculate NDVi and NDSI for sensor to adjust ASTER emissivity
def generate_landsat_NDVI_NDSI(dataLandsatASTER,mask):
    
    ndvi = (dataLandsatASTER.getRad('rad5') - dataLandsatASTER.getRad('rad4'))/(dataLandsatASTER.getRad('rad5') + dataLandsatASTER.getRad('rad4'))
    
    where_are_NaNs = np.isnan(ndvi)
    ndvi[where_are_NaNs] = 0
    
    ndvi[mask]=0
    ndvi[ndvi < 0.0000001] = 0
    ndvi[ndvi > 1] = 1
    ndvi = ndvi/np.max(ndvi)
    
    ndsi = (dataLandsatASTER.getRad('rad3') - dataLandsatASTER.getRad('rad6'))/(dataLandsatASTER.getRad('rad3') + dataLandsatASTER.getRad('rad6'))
    ndsi[mask]=0
    snow_locations = np.where(ndsi > 0.4)
    
    return ndvi, ndsi, snow_locations
    

# convert ASTER emissivity to sensor emissivity    
def convert_emis_from_ASTER(dataLandsatASTER, sensor):
    
    coefficients = sensor_coefficients(sensor)
    ls_emis_data = (coefficients['estimated_1'] * dataLandsatASTER.getAster('emis13') +
                 coefficients['estimated_2'] * dataLandsatASTER.getAster('emis14') +
                 coefficients['estimated_3'])
    
    return ls_emis_data


# convert DC to radiance and scale ASTER data
def convertGainBias(dataLandsatASTER,mask,MTL):
    
    f = open(MTL,"r")
    content = f.read()
                    
    rad3_mult_index = content.index('RADIANCE_MULT_BAND_3')
    rad4_mult_index = content.index('RADIANCE_MULT_BAND_4')
    rad5_mult_index = content.index('RADIANCE_MULT_BAND_5')
    rad6_mult_index = content.index('RADIANCE_MULT_BAND_6')
    
    rad3_add_index = content.index('RADIANCE_ADD_BAND_3')
    rad4_add_index = content.index('RADIANCE_ADD_BAND_4')
    rad5_add_index = content.index('RADIANCE_ADD_BAND_5')
    rad6_add_index = content.index('RADIANCE_ADD_BAND_6')
    
    rad3_mult = float(content[rad3_mult_index+23:rad3_mult_index+37].rstrip())
    rad4_mult = float(content[rad4_mult_index+23:rad4_mult_index+37].rstrip())
    rad5_mult = float(content[rad5_mult_index+23:rad5_mult_index+37].rstrip())
    rad6_mult = float(content[rad6_mult_index+23:rad6_mult_index+37].rstrip())
    
    rad3_add = float(content[rad3_add_index+22:rad3_add_index+24+9].rstrip())
    rad4_add = float(content[rad4_add_index+22:rad4_add_index+24+9].rstrip())
    rad5_add = float(content[rad5_add_index+22:rad5_add_index+24+9].rstrip())
    rad6_add = float(content[rad6_add_index+22:rad6_add_index+24+9].rstrip())
    
    # convert to radiance, and then to reflectance
    dataLandsatASTER.setRad('rad3',(dataLandsatASTER.getRad('rad3')*rad3_mult + rad3_add) * 0.00002-0.1)      # convert to reflectance
    dataLandsatASTER.setRad('rad4',(dataLandsatASTER.getRad('rad4')*rad4_mult + rad4_add) * 0.00002-0.1)       # convert to reflectance
    dataLandsatASTER.setRad('rad5',(dataLandsatASTER.getRad('rad5')*rad5_mult + rad5_add) * 0.00002-0.1)       # convert to reflectance
    dataLandsatASTER.setRad('rad6',(dataLandsatASTER.getRad('rad6')*rad6_mult + rad6_add)* 0.00002-0.1)       # convert to reflectance
    
    #do not convert here - this happens during the SW calculations
    #dataLandsatASTER.setRad('rad10',dataLandsatASTER.getRad('rad10')*0.00033420+0.1)  # convert to radiance
    #dataLandsatASTER.setRad('rad11',dataLandsatASTER.getRad('rad11')*0.00033420+0.1)  # convert to radiance
    
    # ASTER data converted when downloaded and stacked
    #dataLandsatASTER.setAster('emis13',dataLandsatASTER.getAster('emis13')*0.001)
    #dataLandsatASTER.setAster('emis14',dataLandsatASTER.getAster('emis14')*0.001)
    #dataLandsatASTER.setAster('ndvi', dataLandsatASTER.getAster('ndvi')*0.01)
    
    # convert SC LST
    # try:
    #     ST_mult_index = content.index('TEMPERATURE_MULT_BAND_ST_B10')
    #     ST_add_index = content.index('TEMPERATURE_ADD_BAND_ST_B10') 
    #     ST_mult = float(content[ST_mult_index+30:ST_mult_index+42].rstrip())
    #     ST_add = float(content[ST_add_index+30:ST_add_index+35].rstrip())
    #     dataLandsatASTER.setRad('SC_LST',(dataLandsatASTER.getRad('SC_LST')*ST_mult + ST_add))       # convert to ST
    # except:
    #     print('Could not convert Single Channel ST to Kelvin from MTL file')
            
    
    temp = dataLandsatASTER.getAster('ndvi')
    temp[mask]=0
    temp[temp < 0.0000001] = 0
    temp[temp > 1] = 1
    where_are_NaNs = np.isnan(temp)
    temp[where_are_NaNs] = 0
    temp = temp/np.max(temp)
    
    dataLandsatASTER.setAster('ndvi',temp)
    
    return dataLandsatASTER
    
    
def calculate_emissivity(dataLandsatASTER, ls_emis_data, ndvi, ndsi, snow_locations, sensor):
    
    coefficients = sensor_coefficients(sensor)
    water_locations = np.where(ls_emis_data > coefficients['aster_water'])
    
    # Get pixels with significant bare soil component 
    bare_locations = np.where(dataLandsatASTER.getAster('ndvi') < coefficients['bare_soil_variable'])
    
    
    ls_emis_bare = (ls_emis_data[bare_locations] \
        - 0.975 * dataLandsatASTER.getAster('ndvi')[bare_locations]) \
        / (1 - dataLandsatASTER.getAster('ndvi')[bare_locations])
    
    ls_emis_final = (coefficients['vegetation_coeff'] * ndvi +
                     ls_emis_data * (1.0 - ndvi))
    
    ls_emis_final[bare_locations] = ls_emis_bare
    ls_emis_final[snow_locations] = coefficients['snow_emissivity']
    
    ls_emis_final[ls_emis_final > 1] = coefficients['aster_water']
    ls_emis_final[water_locations] = coefficients['aster_water']
    
    return ls_emis_final
    
    
    
    
    
    
    
    
    
