#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:08:16 2022

@author: tkpci
"""

import numpy as np
#from example_data_tk import *
from read_TIGR_RTTOV import read_TIGR
from rttov_wrapper_f2py import *
import sys
import pdb


def run_RTTOV(emis10, emis11, num = 1000):
    
    # read TIGR profiles into dictionary (set to read only 200 selected profiles for test)
    profile = read_TIGR(num)  # read each profiles as TIGR_data[i] - where i is a sting from 0 to 200
    
    #pdb.set_trace()
    # read in pressure levels [hPa] surface last
    p_ex = np.flip(np.array(profile['preshPa'], dtype=np.float64))
    # read in air temperature levels [K]
    t_ex = np.flip(np.array(profile['tempK'], dtype=np.float64))
    # read in water vapor
    q_ex = np.flip(np.array(profile['cwv'], dtype=np.float64)) 
    # See wrapper user guide for gas IDs (page 48) - Id 1 = water vapor
    gas_id_q = 1
    # Set profile gas_units: 0=>ppmv over dry air; 1=>kg/kg; 2=>ppmv over moist air
    gas_units = 1  # kg/kg
    # gas_units = 2  # ppmv over moist air
    
    
    # Define number of profiles and number of levels
    nprofiles = 1
    nlevels = len(p_ex)

    # The gas ID array tells the wrapper which gases, aerosol and cloud profiles are being supplied:
    # it must specify water vapour in all cases plus any other optional items;
    gas_id = np.array([gas_id_q], dtype=np.int32)


    # Define arrays for pressure, temperature and gases/clouds/aerosols;
    # specify Fortran ('F') order for array storage to be more efficient
    p = np.empty((nlevels, nprofiles), order='F', dtype=np.float64)
    t = np.empty((nlevels, nprofiles), order='F', dtype=np.float64)
    gases = np.empty((nlevels, nprofiles, len(gas_id)), order='F', dtype=np.float64)

    # Populate the pressure, temperature, q and co2 arrays: these are the same for both profiles
    for i in range(nprofiles):
        p[:, i] = p_ex[:]
        t[:, i] = t_ex[:]
        #q_ex = q_ex/621.9907 * 10**6  # convert from g/kg to ppmv
        q_ex = q_ex/1000 # convert from g/kg to kg/kg
        gases[:, i, 0] = q_ex[:]      # index 0 in gas_id array above is water vapour
 
   
    
    # Initialise cloud inputs to zero
    #gases[:, :, 2:] = 0.

    # Logical flag to set cloud and aerosol units: => true kg/kg (cld+aer);
    mmr_cldaer = 1
    
    # approxiamtion of date since year is not known
    iday = profile['iday']
    month = int(iday/30) +1
    day = iday-30*(month-1)
    # datetimes[6][nprofiles]: yy, mm, dd, hh, mm, ss
    datetimes = np.array([[2015, month, day, 0, 0, 0]], dtype=np.int32)

    # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi (for those - thinking this is Japanese, no, its for satellite zenith, sun azimuth etc)
    angles = np.array([[0.,  0., 45., 180.]], dtype=np.float64)

    # surftype[2][nprofiles]: surftype (0=land,1=sea,2=sea-ice), watertype % used for surface solar BRDF model only
    surftype = np.array([[0, 0]],dtype=np.int32)

    # surfgeom[3][nprofiles]: lat, lon, elev
    surfgeom = np.array([profile['lat'], profile['lon'],profile['alt']], dtype=np.float64)

    # s2m[6][nprofiles]: 2m p, 2m t, 2m q, 10m wind u, v, wind fetch
    #s2m = np.array([[1013., 0.263178E+03, 0.236131E+04, 4., 2., 100000.]], dtype=np.float64)
    s2m = np.array([[p_ex[-1], t_ex[-1], q_ex[-1], 0., 0., 0.]], dtype=np.float64)
    
    #pdb.set_trace()

    # skin[9][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5 (RTTOV default for land 3. 0,5.0, 15.0, 0.1, 0.3)
    skin = np.array([[np.round(profile['lst'],2), 0., 0., 0., 3.0, 5.0, 15.0, 0.1, 0.3]], dtype=np.float64)
    

    # simplecloud[2][nprofiles]: ctp, cfraction
    simplecloud = np.array([[0., 0.]], dtype=np.float64)

    # clwscheme[nprofiles]: clw_scheme, clwde_param
    clwscheme = np.array([[0, 0]], dtype=np.int32)

    # icecloud[2][nprofiles]: ice_scheme, icede_param
    icecloud = np.array([[0, 0]], dtype=np.int32)

    # zeeman[2][nprofiles]: be, cosbk
    zeeman = np.array([[0., 0.]], dtype=np.float64)
    
    # The remaining profile data is specified in example_data.py
    # Note that most arrays in example_data.py need to be transposed
    # for use with the direct wrapper interface rather than pyrttov:
    datetimes   = datetimes.transpose()
    angles      = angles.transpose()
    surftype    = surftype.transpose()
    surfgeom    = surfgeom.transpose()
    s2m         = s2m.transpose()
    skin        = skin.transpose()
    simplecloud = simplecloud.transpose()
    clwscheme   = clwscheme.transpose()
    icecloud    = icecloud.transpose()
    zeeman      = zeeman.transpose()

    # =================================================================
    
    # =================================================================
    # Load the instrument

    # Specify RTTOV and wrapper options. In this case:
    # - turn interpolation on
    # - provide access to the full radiance structure after calling RTTOV
    # - turn on the verbose wrapper option
    # NB the spaces in the string between option names and values are important!
    # set store_rad2 to get down and upwelling radiance as well (not just total rad)
    opts_str = 'opts%interpolation%addinterp 1 ' \
               'store_trans 1 '                  \
               'store_rad 1 '                    \
               'store_rad2 1 '                   \
               'verbose_wrapper 0 '

    # Specify instrument and channel list and add coefficient files to the options string
    rtcoef_dir = '/dirs/data/tirs/RTTOV/rttov131.common/rtcoef_rttov13/'

    #################################################################
    rtcoef_file = rtcoef_dir + 'rttov13pred54L/rtcoef_landsat_8_tirs_o3co2.dat'
    sccldcoef_file = rtcoef_dir + 'cldaer_visir/sccldcoef_landsat_8_tirs.dat'
    
    nchannels = 2
    channel_list = np.arange(1, nchannels+1, 1, dtype=np.int32)
    
    # rtcoef_file = rtcoef_dir + 'rttov13pred54L/rtcoef_eos_1_modis_o3co2_ironly.dat'
    # sccldcoef_file = rtcoef_dir + 'cldaer_ir/sccldcoef_eos_1_modis_ironly.dat'

    # nchannels = 16
    # channel_list = np.arange(1, nchannels+1, 1, dtype=np.int32)

    opts_str += ' file_coef ' + rtcoef_file + \
                ' file_sccld ' + sccldcoef_file


    # Call the wrapper subroutine to load the instrument and check we obtained a valid instrument ID
    inst_id = rttov_load_inst(opts_str, channel_list)
    if inst_id < 1:
        print('Error loading instrument')
        sys.exit(1)
    # =================================================================
    
        
    # =================================================================
    # Initialise emissivity atlas
    
    # emis_atlas_path = '/dirs/data/tirs/RTTOV/rttov131.common/emis_data/'
    # month = datetimes[1, 0]            # Month is taken from the profile date
    
    # # Call the wrapper subroutine to set up the IR emissivity atlas
    # # NB we specify inst_id here so the atlas is initialised for this specific instrument for faster access;
    # #    to initialise the atlas for use with multiple instruments pass 0 as the inst_id
    # #    (see wrapper user guide for more information)
    # atlas_wrap_id = rttov_load_ir_emis_atlas(emis_atlas_path, month, -1, inst_id, 0)
    # if atlas_wrap_id < 1: print('Error loading IR emissivity atlas: atlas will not be used')
    # =================================================================
    
    
    # =================================================================
    # Declare arrays for other inputs and outputs
    
    # Define array for input/output surface emissivity and BRDF
    surfemisrefl = np.empty((nchannels, nprofiles, 4), order='F', dtype=np.float64)
    
    # # Define direct model outputs
    btrefl  = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
    rad     = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
   
    # # =================================================================
    # # specify emissivity for each channel
    surfemisrefl[:,:,:] = -1.
    surfemisrefl[0,:,0] = emis10  # first band (10)
    surfemisrefl[1,:,0] = emis11  # second band (11)
    
    
  
    # =================================================================
    # Call RTTOV
    
    # # Initialise the surface emissivity and reflectance before every call to RTTOV:
    # # in this case we specify a negative number to use the IR atlas over land
    # # (because we initialised it above) and to use RTTOV's emissivity models over sea surfaces
    # # surfemisrefl[:,:,:] = -1.
    
    # # Use atlas
    # if atlas_wrap_id > 0:
    #     err = rttov_get_emisbrdf(atlas_wrap_id, surfgeom[0], surfgeom[1], surftype[0], surftype[1], \
    #                               angles[0,:], angles[1,:], angles[2,:], angles[3,:], skin[2,:], \
    #                               inst_id, channel_list, surfemisrefl[:,:,0])
    #     if err != 0:
    #         print('Error returning atlas emissivities: not using atlas')
    #         surfemisrefl[:,:,:] = -1.
    err = 0
    # Call the wrapper subroutine to run RTTOV direct
    err = rttov_call_direct(inst_id, channel_list, datetimes, angles, surfgeom, surftype, skin, s2m, \
                            simplecloud, clwscheme, icecloud, zeeman, p, t, gas_units, mmr_cldaer, \
                            gas_id, gases, surfemisrefl, btrefl, rad)
    if err != 0:
        print('Error running RTTOV direct')
        sys.exit(1)
    # =================================================================
    
    
    # =================================================================
    # Examine outputs
    
    # Outputs available are:
    # - surfemisrefl array contains surface emissivities (and reflectances) used by RTTOV
    # - rad array contains RTTOV radiance%total array
    # - btrefl array contains RTTOV radiance%bt and radiance%refl arrays (depending on channel wavelength)
    # - it is also possible to access the whole radiance structure because we set the store_rad option above
    
    dataRTTOV = {}
    
    dataRTTOV['emis10'] = surfemisrefl[0,0,0]
    dataRTTOV['emis11'] = surfemisrefl[1,0,0]
    
    btclear = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
    err = rttov_get_bt_clear(inst_id, btclear) # get brightness temperature
    
    dataRTTOV['T10'] = np.round(btclear[0,0],3)
    dataRTTOV['T11'] = np.round(btclear[1,0],3)
    
    rad = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
    err = rttov_get_rad_clear(inst_id, rad)  #mW/cm-1/sr/m2
    
    # convert to W/m^2/sr/um from mW/cm-1/sr/m^2 (wavenumber)
    # convert to W/m^2.sr/cm-1 = /1000
    # convert wavenumber to micron - um = 10000/cm-1 => cm-1 = 10000/um
    # central wavenumber => band 10 = 0.9184305420E+03
    # central wavenumber => band 11 = 0.8346752399E+03
    
    dataRTTOV['rad10'] = np.round(rad[0,0]/1000*0.9184305420E+03**2/10000,3) # convert to W/m^2/sr/um from mW/cm-1/sr/m^2 (wavenumber)
    dataRTTOV['rad11'] = np.round(rad[1,0]/1000*0.8346752399E+03**2/10000,3) # convert to W/m^2/sr/um
    
    trans = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
    err = rttov_get_tau_total(inst_id, trans)
    
    dataRTTOV['trans10'] = np.round(trans[0,0],3)
    dataRTTOV['trans11'] = np.round(trans[1,0],3)
    
    upwell = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
    err = rttov_get_rad2_upclear(inst_id, upwell) #mW/cm-1/sr/m2
    
    dataRTTOV['upwell10'] = np.round(upwell[0,0]/1000*0.9184305420E+03**2/10000,3) # convert to W/m^2/sr/um
    dataRTTOV['upwell11'] = np.round(upwell[1,0]/1000*0.8346752399E+03**2/10000,3) # convert to W/m^2/sr/um
    
    downwell = np.empty((nchannels, nprofiles), order='F', dtype=np.float64)
    err = rttov_get_rad2_dnclear(inst_id, downwell) #mW/cm-1/sr/m2
    
    dataRTTOV['down10'] = np.round(downwell[0,0]/1000*0.9184305420E+03**2/10000,3) # convert to W/m^2/sr/um
    dataRTTOV['down11'] = np.round(downwell[1,0]/1000*0.8346752399E+03**2/10000,3) # convert to W/m^2/sr/um
    
    dataRTTOV['LST'] = np.round(profile['lst'],1)  # rounding to 1 decimal since MODTRAN rounds the profiles to that    
   
    return dataRTTOV
    
    # =================================================================
    
    
    # =================================================================
    # Deallocate memory for all instruments and atlases
    
    #err = rttov_drop_all()
    #if err != 0: print('Error deallocating wrapper')
    # =================================================================
        
        
       
        
        
        
        
        
        
    