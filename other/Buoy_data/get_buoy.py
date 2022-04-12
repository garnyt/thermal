#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022

@author: ben, eon, tania
"""
import os
from os.path import join
import numpy
import datetime
import urllib.error
import urllib.parse
import urllib.request
import shutil
import requests
import gzip
import warnings
import pdb

CHUNK = 1024 * 1024 * 8   # 1 MB

class RemoteFileException(Exception):
    pass

class BuoyDataException(Exception):
    pass


class Buoy(object):
    
    # Variables used for calculations
    _Ts = None          # skin temperature
    _T0 = None          # average skin temperature
    _a = None           # 
    _b = None           # dampening constant
    _c = None           # phase constant
    _Um = None          # 24-hour wind speed at height 10m above sea level
    _U1 = None          # 24-hour wind speed at height H1 (if not 10m)
    _f_tcz = None       # magnitude of diurnal surface variation at time t
    _T_zt = None        # Temperature at depth z at time t
    
    def __init__(self, _id, lat, lon, thermometer_depth, height, skin_temp=None,
                 surf_press=None, surf_airtemp=None, surf_rh=None, url=None,
                 filename=None, bulk_temp=None):
         self.id = _id
         self.lat = lat
         self.lon = lon
         self.thermometer_depth = thermometer_depth
         self.height = height
         self.skin_temp = skin_temp
         self.bulk_temp = bulk_temp
         self.surf_press = surf_press
         self.surf_airtemp = surf_airtemp
         self.surf_rh = surf_rh

         self.url = url
         self.filename = filename

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Buoy ID: {0} Lat: {1} Lon: {2}'.format(self.id, self.lat, self.lon)

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

    out_dir =  out_dir + '/noaa'
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

def ungzip(filepath):
    """ un-gzip a file (equivalent `gzip -d filepath`) """
    
    new_filepath = filepath.replace('.gz', '')

    with open(new_filepath, 'wb') as f_out, gzip.open(filepath, 'rb') as f_in:
        try:
           shutil.copyfileobj(f_in, f_out)
        except OSError as e:
            warnings.warn(str(e) + filepath, RuntimeWarning)
            shutil.copyfile(filepath, new_filepath)
            os.remove(filepath)

    return new_filepath

def create_dict(indexes, values):
    
    dictionary = dict(zip(indexes, values))
        
    return dictionary

def read_buoy_heights():
    
    BUOY_TXT = join('misc_files', 'buoyht.txt') 
    buoy_ids, buoy_heights, anemometer_heights = numpy.genfromtxt(BUOY_TXT, skip_header=7,
                                  usecols=(0, 1, 3), unpack=True)
    
    return buoy_ids, buoy_heights, anemometer_heights

def all_datasets():
	# TODO memoize
    """
    Get list of all NOAA buoy datasets.

    Return:
        [[Buoy_ID, lat, lon, thermometer_depth, height], [ ... ]]

    """
    #buoys, heights, anemometer_height = numpy.genfromtxt(settings.BUOY_TXT, skip_header=7,
    #                                  usecols=(0, 1, 3), unpack=True)
    #buoy_heights = dict(zip(buoys, heights))
    buoy_ids, buoy_heights, anemometer_heights = read_buoy_heights()
    buoy_heights = create_dict(buoy_ids, buoy_heights)

    buoy_stations = {}

    STATION_TXT = join('misc_files', 'station_table.txt')     
    with open(STATION_TXT, 'r') as f:
        f.readline()
        f.readline()

        for line in f:
            info = line.split('|')
            sid = info[0]   # 1st column, Station ID
            if not sid.isdigit():  # TODO check if is buoy or ground station
                continue
            payload = info[5]   # 6th column, buoy payload type

            lat_lon = info[6].split(' (')[0]   # 7th column, discard part
            lat_lon = lat_lon.split()

            if lat_lon[1] == 'S':
                lat = float(lat_lon[0]) * (-1)
            else:
                lat = float(lat_lon[0])

            if lat_lon[3] == 'W':
                lon = float(lat_lon[2]) * (-1)
            else:
                lon = float(lat_lon[2])

            ######## --> This implementation is incorrect.  Get details from buoy data file
            # and if there is no hull type specified, ignore buoy.  Direction from Matt Montanaro
            ########
            # TODO research and add more payload options
            if payload == 'ARES payload':
                depth = 1.0
            elif payload == 'AMPS payload':
                depth = 0.6
            else:
                depth = 0.8

            buoy_stations[sid] = Buoy(sid, lat, lon, depth, buoy_heights.get(sid, 0))

    return buoy_stations


def get_station_id(stations,buoy_lat,buoy_lon):
    
    lat = []
    lon = []
    st_id = []
    
    for stat in stations:
        lat.append(stations[stat].lat)
        lon.append(stations[stat].lon)
        st_id.append(stations[stat].id)
    lat = numpy.asarray(lat).astype(float)
    lon = numpy.asarray(lon).astype(float)
    st_id = numpy.asarray(st_id)

    distances = (lat - buoy_lat)**2 + (lon - buoy_lon)**2
    dist_args_sorted = numpy.argsort(distances.flatten())

    chosen_idxs = int(dist_args_sorted[0])
    station_id = st_id[chosen_idxs]
    return station_id

def wind_speed_height_correction(wspd, h1, h2, n=0.1):
    # equation 2.9 in padula, simpolified wind speed correction
    return wspd * (h2 / h1) ** n

def calc_bulk_temp(w_temp_slice):
    
    bulk_temp = (float(0))
    
    if not (len(w_temp_slice) == numpy.count_nonzero(numpy.isnan(w_temp_slice))):
        bulk_temp = numpy.nanmean(w_temp_slice)
    
    return bulk_temp

def calc_avg_skin_temp(avg_bulk_temp_celsius, a, b, c, z, u_m):
    
    az = a * z
    e_bz = numpy.exp(b * z)
    cz = c * z
    
    #pdb.set_trace()
    
    if (u_m < 1.5):
        raise BuoyDataException('Insufficient Water Mixing - Wind Speed Too Low')
    else:
        if ((az > -1.1 and az < 0) and
            ((e_bz > 1) and (e_bz < 6)) and
            (cz > 0) and (cz < 4)):
            if (u_m >= 1.5):
                avg_skin_temp = avg_bulk_temp_celsius - az - 0.2
        elif (u_m > 7.6):
            avg_skin_temp = avg_bulk_temp_celsius - 0.17
        else:
            raise BuoyDataException('Water mixing prerequisites not met')
        
    return avg_skin_temp

def dt_to_dec_hour(dt):
    return dt.hour + dt.minute / 60

def to_kelvin(celsius_value):
    kelvin_value = celsius_value + 273.15
    return kelvin_value

def calc_skin_temp(buoy_id, data, dates, headers, overpass_date, buoy_depth):
    
    
    date_range = []
    date_range.append([])
    date_range.append([])
    
    # Build date_range array with all dates that fall within 12 hours (positive and negative) of overpass date
    
    #dt
    date_range = [(i, d) for i, d in enumerate(dates) if abs(d - overpass_date) < datetime.timedelta(hours=12)]
    
    if len(date_range) == 0:
        raise BuoyDataException('No Buoy Data')

    # Split array into index (line number) and date
    dt_line_numbers_24h, dt_times_24h = zip(*date_range)
    # Build array of water temperatures
    w_temp_24h = data[dt_line_numbers_24h, headers.index('WTMP')]
    # Build array of wind speeds
    wind_spd_24h = data[dt_line_numbers_24h, headers.index('WSPD')]
    
    t_zt = list(zip(dt_times_24h, w_temp_24h))
    # Calculate Temperature at depth z at time t
    
    buoy_ids, buoy_heights, anemometer_heights = read_buoy_heights()
    anemometer_height_dict = create_dict(buoy_ids, anemometer_heights)
    
    if (int(buoy_id)) in anemometer_height_dict:
        anemometer_height = anemometer_height_dict[int(buoy_id)]
    else:
        anemometer_height = 5 # If the buoy id does not exist in the heights file
    
    if anemometer_height == 'N/A':
        anemometer_height = 5 # If there is no value in the heights file, use a default of 5
    
    #pdb.set_trace()
    
    # 24 hour average wind Speed at 10 meters (measured at 5 meters) 
    u_m = wind_speed_height_correction(numpy.nanmean(wind_spd_24h), anemometer_height, 10) # Equasion 2.9 Frank Pedula's thesis, page 23
    
    avg_bulk_temp_celsius_24h = calc_bulk_temp(w_temp_24h)

    if numpy.isnan(avg_bulk_temp_celsius_24h):
        raise BuoyDataException('No Water Temperature Data')

    a = 0.05 - (0.6 / u_m) + (0.03 * numpy.log(u_m))   # thermal gradient
    z = buoy_depth   # depth in meters
    
    # part 2
    b = 0.35 + (0.018 * numpy.exp(0.4 * u_m))   # damping constant
    c = 1.32 - (0.64 * numpy.log(u_m))          # phase constant
    
    avg_skin_temp_no_diurnal = calc_avg_skin_temp(avg_bulk_temp_celsius_24h, a, b, c, z, u_m)    

    if numpy.isnan(c):
        raise BuoyDataException('No Wind Speed Data')

    f_tcz_time = []
    f_tcz_temp = []

    #pdb.set_trace()

    for time, temperature in t_zt:
        f_tcz_time.append(dt_to_dec_hour(time) - (c * z))  # calculate adjusted time as described in Equasion 2.10 Frank Pedula's thesis, page 23n
        f_tcz_temp.append((temperature - avg_bulk_temp_celsius_24h) / numpy.exp(-b * z))
    
    # Find closest date to overpass date)
    closest_dt = min([(i, abs(overpass_date - d)) for i, d in enumerate(dates)], key=lambda i: i[1]) # timedelta
    closest_dt = dt_to_dec_hour(dates[closest_dt[0]])
            
    f_t = numpy.interp(dt_to_dec_hour(overpass_date), f_tcz_time, f_tcz_temp, period=100)
    
    #if numpy.isnan(f_t):
    #    raise BuoyDataException('No Water Temperature Data')
    
    skin_temp = avg_skin_temp_no_diurnal + f_t

    # combine
    skin_temp = to_kelvin(skin_temp) # + 273.15   # [K]
    bulk_temp = to_kelvin(avg_bulk_temp_celsius_24h)

    return skin_temp, bulk_temp

def download(b_id, date, out_dir):
    """
    Download and unzip appripriate buoy data from url.

    Args:
        url: url to download data from
    """
    
    # for historical data
    
    NOAA_URLS = ['https://www.ndbc.noaa.gov/data/historical/stdmet/%sh%s.txt.gz',
                 'https://www.ndbc.noaa.gov/data/stdmet/%s/%s%s%s.txt.gz',
                 'https://www.ndbc.noaa.gov/data/realtime2/%s.txt']
    
    
    #pdb.set_trace()
    
    diff = date.date() - datetime.date.today()
    
    url1 = NOAA_URLS[0] % (b_id, date.year)
    url2 = NOAA_URLS[1] % (date.strftime('%b'), b_id, date.strftime('%-m'), datetime.datetime.now().strftime('%Y'))
    url3 = NOAA_URLS[2] % ( b_id)
    
    # if abs(diff.days) > 45:
    
    # #if date.year < datetime.date.today().year:
    #     url = NOAA_URLS[0] % (b_id, date.year)        
    # else:
    #     #url = NOAA_URLS[1] % (date.strftime('%b'), b_id, date.strftime('%-m'), datetime.datetime.now().strftime('%Y'))
    #     url = NOAA_URLS[2] % (date.strftime('%b'), b_id)

    try:
        filename = url_download(url1,out_dir)
    except:
        try:
            filename = url_download(url2,out_dir)
        except:
            filename = url_download(url3,out_dir)
        
    
    if '.gz' in filename:
        filename = ungzip(filename)
        
    #pdb.set_trace()

    return filename

def calculate_buoy_information(scene):  #, buoy_id=''):
    """
    Pick buoy dataset, download, and calculate skin_temp.

    Args: None

    Returns: None
    """
    ur_lat = scene['CORNER_UR_LAT_PRODUCT']
    ur_lon = scene['CORNER_UR_LON_PRODUCT']
    ll_lat = scene['CORNER_LL_LAT_PRODUCT']
    ll_lon = scene['CORNER_LL_LON_PRODUCT']
    corners = ur_lat, ll_lat, ur_lon, ll_lon

    datasets = datasets_in_corners(corners)
    # try:
    #     if buoy_id and buoy_id in datasets:
    #         buoy = datasets[buoy_id]
    #         buoy.calc_info(scene.date)
    #         return buoy

    #     for ds in datasets:
    #         datasets[ds].calc_info(scene.date)
    #         return datasets[ds]

    # except RemoteFileException as e:
    #     print(e)
    # except BuoyDataException as e:
    #     print(e)

    # raise BuoyDataException('No suitable buoy found.')
    return datasets


def datasets_in_corners(corners):
    """
    Get list of all NOAA buoy datasets that fall within a landsat scene.

    Args:
        corners: tuple of: (ur_lat, ll_lat, ur_lon, ll_lon)

    Return:
        [[Buoy_ID, lat, lon, thermometer_depth], [ ... ]]

    """
    
    stations = all_datasets()
    inside = {}

    # keep buoy stations and coordinates that fall within the corners
    for stat in stations:
        # check for latitude and longitude
        if point_in_corners(corners, (stations[stat].lat, stations[stat].lon)):
            inside[stat] = stations[stat]

    return inside


def point_in_corners(corners, point):
    ur_lat, ll_lat, ur_lon, ll_lon = corners
    lat, lon = point

    if ur_lat > 0 and not (ll_lat < lat < ur_lat):
        return False
    elif ur_lat <= 0 and not (ll_lat > lat > ur_lat):
        return False

    if ur_lon > 0 and not (ll_lon > lon > ur_lon):
        return False
    elif ur_lon <= 0 and not (ll_lon < lon < ur_lon):
        return False

    return True

def buoy_process(buoy_id,overpass_date,out_dir):
    

    buoy_file = download(buoy_id, overpass_date, out_dir) # Don't know why file is downloaded again, because the file is being passed from
    data, headers, dates, units = load(buoy_file)
    b = all_datasets()[buoy_id]
    buoy_depth = b.thermometer_depth
    
    #data, headers = load(file)
    dt_slice = [i for i, d in enumerate(dates) if abs(d - overpass_date) < datetime.timedelta(hours=24)]
    closest_dt = min([(i, abs(overpass_date - d)) for i, d in enumerate(dates)], key=lambda i: i[1])

    w_temp = data[dt_slice, headers.index('WTMP')]
    wind_spd = data[dt_slice, headers.index('WSPD')]

    try:
        surf_airtemp = data[closest_dt[0], headers.index('ATMP')]
    except IndexError:
        raise BuoyDataException('Index out of range, no data available')

    try:
        surf_press = data[closest_dt[0], headers.index('BAR')]
    except ValueError:
        surf_press = data[closest_dt[0], headers.index('PRES')]

    surf_dewpnt = data[closest_dt[0], headers.index('DEWP')]
        
    bulk_temp = data[closest_dt[0], headers.index('WTMP')] + 273.15
    
    
    try:
        skin_temp, bulk_temp = calc_skin_temp(buoy_id, data, dates, headers, overpass_date, buoy_depth)
    # except BuoyDataException as e:
    #     print(e)
    
    # except:
    #     
        
    except BuoyDataException as e:
        raise BuoyDataException(str(e))
        print(str(e))
        
    buoy_data = {
             'buoy_lat': b.lat,
             'buoy_lon': b.lon,
             'buoy_depth': b.thermometer_depth,
             'bulk_temp': bulk_temp,
             'skin_temp': skin_temp,
             'lower_atmo': [surf_press, surf_airtemp, surf_dewpnt]
         }
    
        
    return buoy_data
    

def load(filename):
    """
    Open a downloaded buoy data file and extract data from it.

    Args:
        date: datetime object
        filename: buoy file to open

    Returns:
        data: from file, trimmed to date

    Raises:
        Exception: if no data is found in file
    """
    def _filter(iter):
        # NOAA NDBC uses 99.0 and 999.0 as a placeholder for no data
        new = []
        for item in iter:
            i = float(item)
            if i == 99 or i == 999:
                new.append(numpy.nan)
            else:
                new.append(i)
        return new
    
    dates = []
    lines = []
        
    with open(filename, 'r') as f:
        header = f.readline()
        unit = f.readline()

        for line in f:
            date_str = ' '.join(line.split()[:5])

            if len(line.split()[0]) == 4:
                date_dt = datetime.datetime.strptime(date_str, '%Y %m %d %H %M')
            elif len(line.split()[0]) == 2:
                date_dt = datetime.datetime.strptime(date_str, '%y %m %d %H %M')

            try:
                data = _filter(line.split()[5:])
            except ValueError:
                print(line)
            
            lines.append(data)
            dates.append(date_dt)

    headers = header.split()[5:]
    units = unit.split()[5:]
    lines = numpy.asarray(lines)

    return lines, headers, dates, units


def get_corner_coordinates_from_MTL(scene_filepath):
    
    files_in_folder = os.listdir(scene_filepath)
    
    for ele in files_in_folder:
        if 'MTL.txt' in ele:
            MTL_file = ele
            sceneID = MTL_file[0:len(MTL_file)-8]
            landsat = sceneID[2:4]                          # identifier for L8 or L9
            MTL_file = scene_filepath + '/' + MTL_file
    
    
    f = open(MTL_file,"r")
    content = f.read()
    
    UR_lat = content.index('CORNER_UR_LAT_PRODUCT')
    UR_lon = content.index('CORNER_UR_LON_PRODUCT')
    LL_lat = content.index('CORNER_LL_LAT_PRODUCT')
    LL_lon = content.index('CORNER_LL_LON_PRODUCT')
    
    date_acquired = content.index('DATE_ACQUIRED')
    time_aquired = content.index('SCENE_CENTER_TIME')
    
    CORNER_UR_LAT_PRODUCT = float(content[UR_lat+24:UR_lat+24+9].rstrip())
    CORNER_UR_LON_PRODUCT = float(content[UR_lon+24:UR_lon+24+9].rstrip())
    CORNER_LL_LAT_PRODUCT = float(content[LL_lat+24:LL_lat+24+9].rstrip())
    CORNER_LL_LON_PRODUCT = float(content[LL_lon+24:LL_lon+24+9].rstrip())
    
    
    fulldate = content[date_acquired+16:date_acquired+16+11].rstrip()+' ' + content[time_aquired+21:time_aquired+21+14].rstrip()
    overpass_date = datetime.datetime.strptime(fulldate, '%Y-%m-%d %H:%M:%S.%f')
   
    
    scene = {}
    scene['CORNER_UR_LAT_PRODUCT'] = CORNER_UR_LAT_PRODUCT
    scene['CORNER_UR_LON_PRODUCT'] = CORNER_UR_LON_PRODUCT
    scene['CORNER_LL_LAT_PRODUCT'] = CORNER_LL_LAT_PRODUCT
    scene['CORNER_LL_LON_PRODUCT'] = CORNER_LL_LON_PRODUCT
    
    return scene, overpass_date
    
    
def main(scene_filepath):
    
    out_dir = '/dirs/data/tirs/tania-dir'
    
    # Uncomment below three lines for one buoy only
    # stations = all_datasets()
    # buoy_id = get_station_id(stations,buoy_lat,buoy_lon)
    # overpass_date = datetime.datetime(2018,1,5)
    
    # use below if you have landsat scene
    scene_filepath = '/dirs/data/tirs/Landsat9/LC09_L1TP_014035_20211114_20211209_02_T1/'
    scene, overpass_date = get_corner_coordinates_from_MTL(scene_filepath)
    datasets = calculate_buoy_information(scene)
    
    for ds in datasets:
        #print(datasets[ds])
        buoy_id = datasets[ds].id
        buoy_process(buoy_id,overpass_date,out_dir)
    
    

# test code

# buoy_lat = 39.584
# buoy_lon = -72.6
# out_dir = '/dirs/data/tirs/tania-dir'
# overpass_date = datetime.datetime(2017,5,1)
# buoy_data = buoy_process(overpass_date,buoy_lat,buoy_lon,out_dir)