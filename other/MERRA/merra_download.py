import numpy
from netCDF4 import num2date
import pdb
import datetime
import os
#import re
#import sys
#import shutil
#import urllib.error
#import urllib.parse
#import urllib.request
#import warnings
#import time
import requests
import netCDF4




def download(date):
    """
    Download MERRA data via ftp.
    """

    MERRA_LOGIN = ('cisthermal','C15Th3rmal')           # https://disc.gsfc.nasa.gov/
    MERRA_DIR = '/dirs/data/tirs/downloads/merra2/'
    MERRA_URL = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NPASM.5.12.4/%s/%s/MERRA2_400.inst3_3d_asm_Np.%s.nc4'
    
    # year with century, zero padded month, then full date
    url = MERRA_URL % (date.strftime('%Y'), date.strftime('%m'),
                                date.strftime('%Y%m%d'))

    filename = url_download(url, MERRA_DIR, auth=MERRA_LOGIN)
    return filename

def url_download(url, out_dir, _filename=None, auth=None):
    """ download a file (ftp or http), optional auth in (user, pass) format """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = _filename if _filename else url.split('/')[-1]
    filepath = os.path.join(out_dir, filename)

    if os.path.isfile(filepath):
        return filepath

    download_http(url, filepath, auth)

    return filepath


def download_http(url, filepath, auth=None):
    """ download a http or https resource using requests. """
        
    with requests.Session() as session:
            
        req = session.request('get', url)

        if auth:
            output_string = "     Opening session..."
            print(output_string)
            
            resource = session.get(req.url, auth=auth)
    
            if resource.status_code != 200:
                error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                print(error_string)
            else:
                output_string = "     Session opened successfully"
                print(output_string)
                        
        else:
            resource = session.get(req.url)
            
            if resource.status_code != 200:
                error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                print(error_string)
                
            else:
                output_string = "     Session opened successfully"
                print(output_string)
                
        with open(filepath, 'wb') as f:
            output_string = "     Downloading %s " % (filepath[filepath.rfind('/') + 1:])            
            print(output_string)
            
            f.write(resource.content)
            
            output_string = "     Download completed..."
            print(output_string)
                
    return filepath


def process(date, scene_lat, scene_lon, shared_args, verbose=False):
    """
    process atmospheric data, yield an atmosphere
    """
    date = datetime.datetime(2020,2,28,10)
    
    filepath = download(date)

    atmo_data = netCDF4.Dataset(filepath, "r", format="NETCDF4")

    # choose points
    lat = atmo_data.variables['lat'][:]
    lon = atmo_data.variables['lon'][:]
    lat = numpy.stack([lat]*lon.shape[0], axis=0)
    lon = numpy.stack([lon]*lat.shape[1], axis=1)
    
    # find closest points to lat lon
    distances = (lat - scene_lat)**2 + (lon - scene_lon)**2
    dist_args_sorted = numpy.argsort(distances.flatten())
    chosen_idxs = dist_args_sorted[0:4]
    chosen_idxs = numpy.unravel_index(chosen_idxs, lat.shape)
    data_coor = list(zip(lat[chosen_idxs], lon[chosen_idxs]))

    latidx = tuple(chosen_idxs[0])
    lonidx = tuple(chosen_idxs[1])
    
    # find closest times
    dates = num2date(atmo_data.variables['time'][:].data, atmo_data.variables['time'].units)
    t1, t2 = sorted(abs(dates - date).argsort()[:2])
    t1_dt = num2date(atmo_data.variables['time'][t1], atmo_data.variables['time'].units)
    t2_dt = num2date(atmo_data.variables['time'][t2], atmo_data.variables['time'].units)

    index1 = (t1, slice(None), latidx, lonidx)
    index2 = (t2, slice(None), latidx, lonidx)

    press = numpy.array(atmo_data.variables['lev'][:])

    temp1 = numpy.empty
    temp2 = numpy.empty
    
    temp1 = numpy.diagonal(atmo_data.variables['T'][index1], axis1=1, axis2=2).T
    temp2 = numpy.diagonal(atmo_data.variables['T'][index2], axis1=1, axis2=2).T

    rhum1 = numpy.diagonal(atmo_data.variables['RH'][index1], axis1=1, axis2=2).T   # relative humidity
    rhum2 = numpy.diagonal(atmo_data.variables['RH'][index2], axis1=1, axis2=2).T

    height1 = numpy.diagonal(atmo_data.variables['H'][index1], axis1=1, axis2=2).T / 1000.0   # height
    height2 = numpy.diagonal(atmo_data.variables['H'][index2], axis1=1, axis2=2).T / 1000.0

    # interpolate in time, now they are shape (4, N)
    t = interp_time(date, temp1, temp2, t1_dt, t2_dt)
    h = interp_time(date, height1, height2, t1_dt, t2_dt)
    rh = interp_time(date, rhum1, rhum2, t1_dt, t2_dt)
    
    # interpolate in space, now they are shape (1, N)
    height = interp_idw(h, data_coor, [scene_lat, scene_lon])
    temp = interp_idw(t, data_coor, [scene_lat, scene_lon])
    relhum = interp_idw(rh, data_coor, [scene_lat, scene_lon])*100  # to get to % for MODTRAN
    
    # calculate the number of nan and zero values in the array and remove them, reducing the size of the array accordingly
    nr_of_nans1 = numpy.sum(temp1[0].mask)
    nr_of_nans2 = numpy.sum(temp2[0].mask)
    nr_of_nans = max([nr_of_nans1,nr_of_nans2])
    
    height = height[nr_of_nans:]
    temp = temp[nr_of_nans:]
    relhum = relhum[nr_of_nans:]
    press = press[nr_of_nans:]

    # load standard atmosphere for mid-lat summer
    stan_atmo = numpy.loadtxt('/dirs/data/tirs/downloads/modtran/stanAtm.txt', unpack=True)
    stan_height, stan_press, stan_temp, stan_relhum = stan_atmo
    # add standard atmo above cutoff index
    
    cutoff_idx = numpy.abs(stan_press - press[-1]).argmin()
    height = numpy.append(height, stan_height[cutoff_idx:])
    press = numpy.append(press, stan_press[cutoff_idx:])
    temp = numpy.append(temp, stan_temp[cutoff_idx:])
    relhum = numpy.append(relhum, stan_relhum[cutoff_idx:])
    
    if height1.shape[0] == 4: 
    
        height1 = height1[:,nr_of_nans:]
        height2 = height2[:,nr_of_nans:]
        temp1 = temp1[:,nr_of_nans:]
        temp2 = temp2[:,nr_of_nans:]
        rhum1 = rhum1[:,nr_of_nans:]
        rhum2 = rhum2[:,nr_of_nans:]
        press = press[nr_of_nans:]
        
    else:
        height1 = height1[nr_of_nans:]
        height2 = height2[nr_of_nans:]
        temp1 = temp1[nr_of_nans:]
        temp2 = temp2[nr_of_nans:]
        rhum1 = rhum1[nr_of_nans:]
        rhum2 = rhum2[nr_of_nans:]
        press = press[nr_of_nans:]

    return height, press, temp, relhum


def interp_time(date, a1, a2, t1, t2):
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
    a = a1 + (time - t1.hour) * ((a2 - a1)/(t2.hour - t1.hour))

    return a

def interp_idw(samples, locations, point, power=2):
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

def distance(p1, p2):
	""" Euclidean Distance of 2 iterables """
	x1, y1 = p1
	x2, y2 = p2
	return numpy.sqrt((x1-x2)**2+(y1-y2)**2)

#def error_bar_atmos(date, lat_oi, lon_oi, shared_args, verbose=False):
#    
#    filename = download(date, shared_args)
#    atmo_data = data.open_netcdf4(filename)
#
#    # choose points
#    lat = atmo_data.variables['lat'][:]
#    lon = atmo_data.variables['lon'][:]
#    lat = numpy.stack([lat]*lon.shape[0], axis=0)
#    lon = numpy.stack([lon]*lat.shape[1], axis=1)
#    chosen_idxs, data_coor = funcs.choose_points(lat, lon, lat_oi, lon_oi)
#
#    latidx = tuple(chosen_idxs[0])
#    lonidx = tuple(chosen_idxs[1])
#
#    t1, t2 = data.closest_hours(atmo_data.variables['time'][:].data,
#                                atmo_data.variables['time'].units, date)
#    t1_dt = num2date(atmo_data.variables['time'][t1], atmo_data.variables['time'].units)
#    t2_dt = num2date(atmo_data.variables['time'][t2], atmo_data.variables['time'].units)
#
#    index1 = (t1, slice(None), latidx, lonidx)
#    index2 = (t2, slice(None), latidx, lonidx)
#
#    # shape (4, N) each except for pressure
#    press = numpy.array(atmo_data.variables['lev'][:])
#
#    # the .T on the end is a transpose
#    temp1 = numpy.diagonal(atmo_data.variables['T'][index1], axis1=1, axis2=2).T
#    temp2 = numpy.diagonal(atmo_data.variables['T'][index2], axis1=1, axis2=2).T
#
#    rhum1 = numpy.diagonal(atmo_data.variables['RH'][index1], axis1=1, axis2=2).T   # relative humidity
#    rhum2 = numpy.diagonal(atmo_data.variables['RH'][index2], axis1=1, axis2=2).T
#
#    height1 = numpy.diagonal(atmo_data.variables['H'][index1], axis1=1, axis2=2).T / 1000.0   # height
#    height2 = numpy.diagonal(atmo_data.variables['H'][index2], axis1=1, axis2=2).T / 1000.0
#    
#    # calculate the number of nan and zero values in the array and remove them, reducing the size of the array accordingly
#    nr_of_nans1 = numpy.sum(temp1[0].mask)
#    nr_of_nans2 = numpy.sum(temp2[0].mask)
#    nr_of_nans = max([nr_of_nans1,nr_of_nans2])
        
    if height1.shape[0] == 4: 
    
        height1 = height1[:,nr_of_nans:]
        height2 = height2[:,nr_of_nans:]
        temp1 = temp1[:,nr_of_nans:]
        temp2 = temp2[:,nr_of_nans:]
        rhum1 = rhum1[:,nr_of_nans:]
        rhum2 = rhum2[:,nr_of_nans:]
        press = press[nr_of_nans:]
        
    else:
        height1 = height1[nr_of_nans:]
        height2 = height2[nr_of_nans:]
        temp1 = temp1[nr_of_nans:]
        temp2 = temp2[nr_of_nans:]
        rhum1 = rhum1[nr_of_nans:]
        rhum2 = rhum2[nr_of_nans:]
        press = press[nr_of_nans:]

    atmos = []

    for i in range(4):
        atmos.append(append_standard_atmo(height1[i], press, temp1[i], rhum1[i]))
        atmos.append(append_standard_atmo(height2[i], press, temp2[i], rhum2[i]))

    
    return atmos


def append_standard_atmo(height, press, temp, relhum):
    # load standard atmosphere for mid-lat summer
    # TODO evaluate standard atmo validity, add different ones for different TOY?
    
    stan_atmo = numpy.loadtxt(settings.STAN_ATMO, unpack=True)
    stan_height, stan_press, stan_temp, stan_relhum = stan_atmo
    # add standard atmo above cutoff index
    cutoff_idx = numpy.abs(stan_press - press[-1]).argmin()
    height = numpy.append(height, stan_height[cutoff_idx:])
    press = numpy.append(press, stan_press[cutoff_idx:])
    temp = numpy.append(temp, stan_temp[cutoff_idx:])
    relhum = numpy.append(relhum, stan_relhum[cutoff_idx:])

    return height, press, temp, relhum
