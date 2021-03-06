import numpy
from netCDF4 import num2date
import pdb

from . import (data, funcs)
from .. import (settings, interp)
from ..download import url_download


def download(date, shared_args):
    """
    Download MERRA data via ftp.

    Args:
        cc: CalibrationController object

    Returns:
        None
    """
    # year with century, zero padded month, then full date
    # TODO fix merra url to include new format strings
    url = settings.MERRA_URL % (date.strftime('%Y'), date.strftime('%m'),
                                date.strftime('%Y%m%d'))

    filename = url_download(url, settings.MERRA_DIR, shared_args, auth=settings.MERRA_LOGIN)
    return filename


def process(date, lat_oi, lon_oi, shared_args, verbose=False):
    """
    process atmospheric data, yield an atmosphere
    """
    
    filename = download(date, shared_args)

    atmo_data = data.open_netcdf4(filename)

    # choose points
    lat = atmo_data.variables['lat'][:]
    lon = atmo_data.variables['lon'][:]
    lat = numpy.stack([lat]*lon.shape[0], axis=0)
    lon = numpy.stack([lon]*lat.shape[1], axis=1)
    chosen_idxs, data_coor = funcs.choose_points(lat, lon, lat_oi, lon_oi)

    latidx = tuple(chosen_idxs[0])
    lonidx = tuple(chosen_idxs[1])
    
    t1, t2 = data.closest_hours(atmo_data.variables['time'][:].data,
                                atmo_data.variables['time'].units, date)
    t1_dt = num2date(atmo_data.variables['time'][t1], atmo_data.variables['time'].units)
    t2_dt = num2date(atmo_data.variables['time'][t2], atmo_data.variables['time'].units)

    index1 = (t1, slice(None), latidx, lonidx)
    index2 = (t2, slice(None), latidx, lonidx)

    press = numpy.array(atmo_data.variables['lev'][:])
    
    pdb.set_trace()

    temp1 = numpy.empty
    temp2 = numpy.empty
    
    temp1 = numpy.diagonal(atmo_data.variables['T'][index1], axis1=1, axis2=2).T
    temp2 = numpy.diagonal(atmo_data.variables['T'][index2], axis1=1, axis2=2).T

    rhum1 = numpy.diagonal(atmo_data.variables['RH'][index1], axis1=1, axis2=2).T   # relative humidity
    rhum2 = numpy.diagonal(atmo_data.variables['RH'][index2], axis1=1, axis2=2).T

    height1 = numpy.diagonal(atmo_data.variables['H'][index1], axis1=1, axis2=2).T / 1000.0   # height
    height2 = numpy.diagonal(atmo_data.variables['H'][index2], axis1=1, axis2=2).T / 1000.0

    # interpolate in time, now they are shape (4, N)
    t = interp.interp_time(date, temp1, temp2, t1_dt, t2_dt)
    h = interp.interp_time(date, height1, height2, t1_dt, t2_dt)
    rh = interp.interp_time(date, rhum1, rhum2, t1_dt, t2_dt)
    
    # interpolate in space, now they are shape (1, N)
    height = interp.idw(h, data_coor, [lat_oi, lon_oi])
    temp = interp.idw(t, data_coor, [lat_oi, lon_oi])
    relhum = interp.idw(rh, data_coor, [lat_oi, lon_oi])
    
    # calculate the number of nan and zero values in the array and remove them, reducing the size of the array accordingly
    nr_of_nans1 = numpy.sum(temp1[0].mask)
    nr_of_nans2 = numpy.sum(temp2[0].mask)
    nr_of_nans = max([nr_of_nans1,nr_of_nans2])
    
    height = height[nr_of_nans:]
    temp = temp[nr_of_nans:]
    relhum = relhum[nr_of_nans:]
    press = press[nr_of_nans:]

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

    # TODO add buoy stuff to bottom of atmosphere

    if verbose:
        # send out plots and stuff
        stuff = numpy.asarray([height, press, temp, relhum]).T
        h = 'Height [km], Pressure[kPa], Temperature[k], Relative_Humidity[0-100]' + '\nCoordinates: {0} Buoy:{1}'.format(data_coor, buoy)
        
        numpy.savetxt('atmosphere_{0}_{1}_{2}.txt'.format('merra', date.strftime('%Y%m%d'), buoy.id), stuff, fmt='%7.2f, %7.2f, %7.2f, %7.2f', header=h)

    return height, press, temp, relhum
    

def error_bar_atmos(date, lat_oi, lon_oi, shared_args, verbose=False):
    
    filename = download(date, shared_args)
    atmo_data = data.open_netcdf4(filename)

    # choose points
    lat = atmo_data.variables['lat'][:]
    lon = atmo_data.variables['lon'][:]
    lat = numpy.stack([lat]*lon.shape[0], axis=0)
    lon = numpy.stack([lon]*lat.shape[1], axis=1)
    chosen_idxs, data_coor = funcs.choose_points(lat, lon, lat_oi, lon_oi)

    latidx = tuple(chosen_idxs[0])
    lonidx = tuple(chosen_idxs[1])

    t1, t2 = data.closest_hours(atmo_data.variables['time'][:].data,
                                atmo_data.variables['time'].units, date)
    t1_dt = num2date(atmo_data.variables['time'][t1], atmo_data.variables['time'].units)
    t2_dt = num2date(atmo_data.variables['time'][t2], atmo_data.variables['time'].units)

    index1 = (t1, slice(None), latidx, lonidx)
    index2 = (t2, slice(None), latidx, lonidx)

    # shape (4, N) each except for pressure
    press = numpy.array(atmo_data.variables['lev'][:])

    # the .T on the end is a transpose
    temp1 = numpy.diagonal(atmo_data.variables['T'][index1], axis1=1, axis2=2).T
    temp2 = numpy.diagonal(atmo_data.variables['T'][index2], axis1=1, axis2=2).T

    rhum1 = numpy.diagonal(atmo_data.variables['RH'][index1], axis1=1, axis2=2).T   # relative humidity
    rhum2 = numpy.diagonal(atmo_data.variables['RH'][index2], axis1=1, axis2=2).T

    height1 = numpy.diagonal(atmo_data.variables['H'][index1], axis1=1, axis2=2).T / 1000.0   # height
    height2 = numpy.diagonal(atmo_data.variables['H'][index2], axis1=1, axis2=2).T / 1000.0
    
    # calculate the number of nan and zero values in the array and remove them, reducing the size of the array accordingly
    nr_of_nans1 = numpy.sum(temp1[0].mask)
    nr_of_nans2 = numpy.sum(temp2[0].mask)
    nr_of_nans = max([nr_of_nans1,nr_of_nans2])
        
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
