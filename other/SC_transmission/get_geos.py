#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:47:17 2020

@author: eoncis
"""
import os
import numpy
import pdb
from netCDF4 import Dataset
import pdb
import urllib.error
import urllib.parse
import urllib.request

import requests

CHUNK = 1024 * 1024 * 8   # 1 MB
class RemoteFileException(Exception):
    pass

def open_netcdf4(filename):
    """
    Open data file in netCDF4 format.

    Args:
        filename: file to open

    Returns:
        rootgrp: data reference to variables stored in data file.

    Raises:
        IOError: if file does not exist at the expected path
    """
    if not os.path.isfile(filename):
        raise IOError('Data file not at path: {0}'.format(filename))

    rootgrp = Dataset(filename, "r", format="NETCDF4")
    return rootgrp

def choose_points(lat, lon, buoy_lat, buoy_lon, flat=False):
    """
    Choose the four closest NARR or MERRA points to a lat/lon position.

    Args:
        lat, lon: numpy arrays, 2d
        buoy_lat, buoy_lon: these is the point to get close to

    Returns:
        chosen indices, coordinates of the 4 closest points (euclidean)
    """
    
    distances = (lat - buoy_lat)**2 + (lon - buoy_lon)**2
    dist_args_sorted = numpy.argsort(distances.flatten())

    chosen_idxs = dist_args_sorted[0:4]
    chosen_idxs = numpy.unravel_index(chosen_idxs, lat.shape)

    coordinates = list(zip(lat[chosen_idxs], lon[chosen_idxs]))

    return chosen_idxs, coordinates

def distance(p1, p2):
	""" Euclidean Distance of 2 iterables """
	x1, y1 = p1
	x2, y2 = p2
	return numpy.sqrt((x1-x2)**2+(y1-y2)**2)

def idw(samples, locations, point, power=2):
	""" Shepard's Method (inverse distance weighting interpolation)
	
	Args::
		samples: data to be interpolated, of length n
		locations: locations of the data, shape 2, n
		point: point to interpolate too, shape 2
		power: integer, arbitary

	Returns:
		weighted_samples: samples, weighted by the weights

	Notes::
		from A NOVEL CONFIDENCE METRIC APPROACH FOR A LANDSAT LAND SURFACE
		TEMPERATURE PRODUCT, Monica J. Cook and Dr. John R. Schott
	"""
	distances = numpy.asarray([distance(i, point) for i in locations])
	
	weights = distances ** -power
	weights /= weights.sum()   # normalize to 1

	weighted_samples = [weights[i] * samples[i] for i in range(len(locations))]

	return sum(weighted_samples)

def download_ftp(url, filepath):
    """ download an FTP resource. """
    
    total_size = 0
    
    try:
        request = urllib.request.urlopen(url)
        total_size = int(request.getheader('Content-Length').strip())
    except urllib.error.URLError as e:
        print(url)
        error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                    
        raise RemoteFileException(error_string)
        
    downloaded = 0
    filename = filepath[len(filepath) - filepath[::-1].index('/'):]

    with open(filepath, 'wb') as fileobj:        
        while True:
            output_string = "    Downloading %s - %.1fMB of %.1fMB\r" % (filename, (downloaded / 1000000), (total_size / 1000000))
            
            print(output_string)
                        
            chunk = request.read(CHUNK)
            if not chunk:
                break
            fileobj.write(chunk)
            downloaded += len(chunk)

        output_string = "\n     Download completed..."
        print(output_string)
        

    return filepath

def download_http(url, filepath, auth=None):
    """ download a http or https resource using requests. """
        
    with requests.Session() as session:
        output_string = "\n    Opening session to %s" % (url[:url.find('/', 9, len(url))])
        print(output_string)
            
        req = session.request('get', url)

        if auth:
            resource = session.get(req.url, auth=auth)
    
            if resource.status_code != 200:
                error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                    
                raise RemoteFileException(error_string)
                
            else:
                output_string = "\n     Session opened successfully"
                print(output_string)
        
        else:
            resource = session.get(req.url)
            
            if resource.status_code != 200:
                error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                    
                raise RemoteFileException(error_string)
                
            else:
                output_string = "\n     Session opened successfully"
                print(output_string)
                
        with open(filepath, 'wb') as f:
            output_string = "\n    Downloading %s " % (filepath[filepath.rfind('/') + 1:])            
            print(output_string)
            
            f.write(resource.content)
            
            output_string = "\n     Download completed..."
            print(output_string)

    return filepath

def url_download(url, out_dir, _filename=None, auth=None):
    """ download a file (ftp or http), optional auth in (user, pass) format """

    out_dir = 'output_data/' + out_dir + '/geos'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = _filename if _filename else url.split('/')[-1]
    filepath = os.path.join(out_dir, filename)

    if os.path.isfile(filepath):
        return filepath
    
    if url[0:3] == 'ftp':
        download_ftp(url, filepath)
    else:
        download_http(url, filepath, auth)

    return filepath

def geos_closest_time(date):
    
    d_int = date.hour + date.minute/60
    d = numpy.asarray([0, 3, 6, 9, 12, 15, 18, 21]).astype(int)
    t1, t2 = sorted(abs(d - d_int).argsort()[:2])
    d1 = d[t1]
    d2 = d[t2]
    
    return d1,d2

def geos_interp_time(date, a1, a2, t1, t2):
    """ linear interp.
    Args:
        date: Python datetime object
        a1, a2: 2 numpy arrays, same dimensions as each other and output
    """
    hour = date.hour
    minute = date.minute
    second = date.second

    # round to nearest minute
    if second > 30: minute = minute + 1

    # convert hour-min acquisition time to decimal time
    time = hour + minute / 60.0

    # interpolate in time
    a = a1 + (time - t1) * ((a2 - a1)/(t2 - t1))

    return a

def calculate_dew_point(RH,T):
    """
    convert RH to Dew Point(Approx of Clausius - Clapeyron equation)
    :param RH: relative humidity
    :param T: air temperature
    :return: DP: the dew point
    """
    B = (numpy.log(RH) + ((17.27 * (T - 273.15)) / (237.3 + T - 273.15))) / 17.27
    DP = (237.3 * B) / (1 - B) + 273.15
    return DP

def geos_download(date,d1,d2,out_dir):
    """
    Download GEOS data via ftp.

    Args:
        cc: CalibrationController object

    Returns:
        None
    """
        
    geos_url = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y%s/M%s/D%s/GEOS.fp.asm.inst3_3d_asm_Np.%s_%02d00.V01.nc4'
    
    url1 = geos_url % (date.strftime('%Y'),
                      date.strftime('%m'),
                      date.strftime('%d'),
                      date.strftime('%Y%m%d'),
                      d1)
    
    url2 = geos_url % (date.strftime('%Y'),
                      date.strftime('%m'),
                      date.strftime('%d'),
                      date.strftime('%Y%m%d'),
                      d2)
    
    filename1 = url_download(url1,out_dir)
    filename2 = url_download(url2,out_dir)
    
    return filename1,filename2

def geos_process(date, lat_oi, lon_oi, out_dir, grnd_alt = 0, max_alt = 100):
    """
    process atmospheric data, yield an atmosphere
    """
    
    d1,d2 = geos_closest_time(date)
    filename1,filename2 = geos_download(date,d1,d2,out_dir)

    atmo_data1 = open_netcdf4(filename1)
    atmo_data2 = open_netcdf4(filename2)
    press = numpy.array(atmo_data1.variables['lev'][:])
    
    # choose points
    lat = atmo_data1.variables['lat'][:]
    lon = atmo_data1.variables['lon'][:]
    lat = numpy.stack([lat]*lon.shape[0], axis=0)
    lon = numpy.stack([lon]*lat.shape[1], axis=1)
    chosen_idxs, data_coor = choose_points(lat, lon, lat_oi, lon_oi)

    latidx = tuple(chosen_idxs[0])
    lonidx = tuple(chosen_idxs[1])
    index1 = (0, slice(None), latidx, lonidx)
    
    temp1 = numpy.empty
    temp2 = numpy.empty
    
    temp1 = numpy.diagonal(atmo_data1.variables['T'][index1], axis1=1, axis2=2).T
    temp2 = numpy.diagonal(atmo_data2.variables['T'][index1], axis1=1, axis2=2).T

    rhum1 = numpy.diagonal(atmo_data1.variables['RH'][index1], axis1=1, axis2=2).T   # relative humidity
    rhum2 = numpy.diagonal(atmo_data2.variables['RH'][index1], axis1=1, axis2=2).T

    height1 = numpy.diagonal(atmo_data1.variables['H'][index1], axis1=1, axis2=2).T / 1000.0   # height
    height2 = numpy.diagonal(atmo_data2.variables['H'][index1], axis1=1, axis2=2).T / 1000.0
    

    # interpolate in time, now they are shape (4, N)
    t = geos_interp_time(date, temp1, temp2, d1,d2)
    h = geos_interp_time(date, height1, height2,d1,d2)
    rh = geos_interp_time(date, rhum1, rhum2,d1,d2)
    
    # interpolate in space, now they are shape (1, N)
    height = idw(h, data_coor, [lat_oi, lon_oi])
    temp = idw(t, data_coor, [lat_oi, lon_oi])
    relhum = idw(rh, data_coor, [lat_oi, lon_oi])
    
    # calculate the number of nan and zero values in the array and remove them, reducing the size of the array accordingly
    nr_of_nans1 = numpy.sum(temp1[0].mask)
    nr_of_nans2 = numpy.sum(temp2[0].mask)
    nr_of_nans = max([nr_of_nans1,nr_of_nans2,])
    
    
    height = height[nr_of_nans:]
    mask = numpy.where((numpy.array(height) >= grnd_alt) & (numpy.array(height) <= max_alt))
    height = height[mask]
    temp = temp[nr_of_nans:]
    temp = temp[mask]
    relhum = relhum[nr_of_nans:]
    relhum = relhum[mask]
    press = press[nr_of_nans:]
    press = press[mask]
    dewtemp = calculate_dew_point(relhum,temp)
    
    profile = list([height,press,temp,dewtemp])
                
    return profile