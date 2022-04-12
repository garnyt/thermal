#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:08:10 2021

@author: tkpci
"""

import numpy as np
import pandas as pd
import os
import subprocess
import re 
import csv
import cv2
from tkinter import * 
import sys
sys.path.insert(1, '/cis/staff/tkpci/Code/Python/TIRS')
import createSiteInfo
from tkinter import filedialog 

def popup_get_folder():
    
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfile(initialdir = "/dirs/data/tirs/downloads/")
    return root.filename
    root.destroy()
    


def getSurfradValues(MTL_file, pathrow, siteInfo):
    
    line = siteInfo.pathrow.index(pathrow)
    site_name = siteInfo.sitename[line]
    short_name = siteInfo.shortname[line]
    lat = siteInfo.lat[line]
    lon = siteInfo.lon[line]
    
    # for each scene, get time and date from MTL file
    f = open(MTL_file,"r")
    content = f.read()
    f.close()
    
    date_line = content.index('DATE_ACQUIRED')
    time_line = content.index('SCENE_CENTER_TIME')
    
    date_val = content[date_line+16:date_line+16+10].rstrip()
    
    time_val = content[time_line+21:time_line+21+8].rstrip()
    hours_L8 = float(time_val[0:2])
    min_L8 = float(time_val[3:5])
    time_L8 = hours_L8 + min_L8/60
    
    my_date = pd.to_datetime(date_val, format='%Y-%m-%d')
    new_year_day = pd.Timestamp(year=my_date.year, month=1, day=1)
    num_of_days = (my_date - new_year_day).days + 1
    year = str(my_date.year)
    
    # create 3 character day of year
    if num_of_days < 10:
        num_of_days_str = '00' + str(num_of_days)
    elif num_of_days < 100:
        num_of_days_str = '0' + str(num_of_days)
    else:
        num_of_days_str = str(num_of_days)

    
    #create download name
    ftp_path = 'ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/' + site_name + '/' + str(year) + '/'  
    ftp_name =  short_name + year[2:4] + num_of_days_str + '.dat'
    ftp_fullname = ftp_path + ftp_name
    
    ftp_dest = '/dirs/data/tirs/downloads/test/' + ftp_name
        
    import urllib 
    from contextlib import closing
    import shutil

    with closing(urllib.request.urlopen(ftp_fullname))as r:
        with open(ftp_dest, 'wb') as f:
            shutil.copyfileobj(r, f)       
            
    data = np.loadtxt(ftp_dest, skiprows=2)
    
    # find closest time to surfrad data  
    time_SR = data[:,6]
    index_closest = np.abs(time_SR-time_L8).argmin()
    
    data[index_closest,0:7]
    
    data_SR = {}
    data_SR['SR_sitename'] = site_name
    data_SR['SR_time'] = data[index_closest,6]
    data_SR['SR_lat'] = lat
    data_SR['SR_lon'] = lon
    data_SR['SR_solar_zen'] = data[index_closest,7]
    data_SR['SR_dw_ir'] = data[index_closest,16]
    data_SR['SR_uw_ir'] = data[index_closest,22]
    data_SR['SR_airtemp'] = data[index_closest,38]
    data_SR['SR_rh'] = data[index_closest,40]
    data_SR['SR_windspd'] = data[index_closest,42]
    data_SR['SR_winddir'] = data[index_closest,44]
    data_SR['SR_pressure'] = data[index_closest,46]
    data_SR['L8_time'] = time_L8
    data_SR['L8_date'] = date_val
    
    os.remove(ftp_dest)

    return data_SR

def main():
    
    MTL_file = popup_get_folder()
    MTL_file = MTL_file.name
    cnt = [m.end(0) for m in re.finditer('/',MTL_file)]
    pathrow = MTL_file[cnt[-1]+10:cnt[-1]+16]
    siteInfo = createSiteInfo.CreateSiteInfo()
    