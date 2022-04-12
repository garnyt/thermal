#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:18:15 2021

@author: tkpci
"""

from osgeo import gdal
import subprocess
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
    root.foldername = filedialog.askdirectory(initialdir = "/dirs/data/tirs/downloads/")
    return root.foldername
    root.destroy()


def getSurfradpixelno(LST_full_path, pathrow, siteInfo):
    
    line = siteInfo.pathrow.index(pathrow)
    site_name = siteInfo.sitename[line]
    short_name = siteInfo.shortname[line]
    lat = siteInfo.lat[line]
    lon = siteInfo.lon[line]
    
    # find pixel location based on lat lon coordinates of site
    pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + LST_full_path + ' ' + str(lon) + ' ' + str(lat), shell=True )
   
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    col = int(numbers[0])
    row = int(numbers[1])
    
    return short_name, col, row
    
    
def main():
    
    data_SR = {}
        
    folder = popup_get_folder()
    
    scenes = os.listdir(folder)
    siteInfo = createSiteInfo.CreateSiteInfo()
    

    
    for scene in scenes:
        
        try:
        
            files_in_folder = os.listdir(folder+'/'+scene)
            
            for ele in files_in_folder:
                if 'MTL.txt' in ele:
                    MTL_file = ele
                if 'SW_LST.tif' in ele:
                    LST_full_path = folder+'/'+scene + '/' + ele                
            
            pathrow = MTL_file[10:16]
            
            short_name, col, row = getSurfradpixelno(LST_full_path, pathrow, siteInfo)
    
    
            data_SR['scene'] = scene    
            data_SR['short_name'] = short_name
            data_SR['col'] = col
            data_SR['row'] = row
            
                
            headers = ",".join(data_SR.keys()) 
            values = ",".join(str(e) for e in data_SR.values())     
        
            # write data to txt file
            filename_out = '/dirs/data/tirs/code/matlab/CloudDistGUI/SurfRadCoordPerScene.csv'
                
            if not os.path.isfile(filename_out):
                
                with open(filename_out, mode='w') as file_out:
                    csv.excel.delimiter=';'
                    file_writer = csv.writer(file_out, dialect=csv.excel)
                
                    file_writer.writerow([headers])
                    file_writer.writerow([values])
                    
            else:
                with open(filename_out, mode='a') as file_out:
                    csv.excel.delimiter=';'
                    file_writer = csv.writer(file_out, dialect=csv.excel)
                
                    file_writer.writerow([values])
        except:
            print(scene, ' not working')
                
        
        
        
        