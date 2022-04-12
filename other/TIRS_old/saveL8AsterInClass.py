#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:43:08 2020

@author: tkpci
"""

import osgeo.gdal as gdal
from dataLandsatASTER import DataLandsatASTER
import numpy as np
import pdb
import createSiteInfo
import subprocess
import re 
import os


def dataForEmisCalc(pathrow, folder, folderID, sceneID):
    
    dataOut = DataLandsatASTER()
    
    #asterName = '/dirs/data/tirs/downloads/aster/AutoDownloads/e13_e14_ndvi_registered.tif' 
    asterName = folder + '/' + folderID + '/''e13_e14_ndvi_registered.tif' 
   
    b10Name = folder + '/' + folderID + '/' + sceneID + '_B10.TIF'
    b11Name = folder + '/' + folderID + '/' + sceneID + '_B11.TIF'
    bqaName = folder + '/' + folderID + '/' + sceneID + '_BQA.TIF'
    
    #pdb.set_trace()
    try:
        files_in_folder = os.listdir(folder+'/'+folderID)
        for ele in files_in_folder:
                if '_SR_B3' in ele:
                    SC_file = ele
                    b3Name = folder+'/' + folderID + '/' + SC_file
    except:
        b3Name = folder + '/' + folderID + '/' + sceneID + '_SR_B3.TIF'
        
    try:
        files_in_folder = os.listdir(folder+'/'+folderID)
        for ele in files_in_folder:
                if '_SR_B4' in ele:
                    SC_file = ele
                    b4Name = folder+'/' + folderID + '/' + SC_file
    except:
        b4Name = folder + '/' + folderID + '/' + sceneID + '_SR_B4.TIF'
        
    try:
        files_in_folder = os.listdir(folder+'/'+folderID)
        for ele in files_in_folder:
                if '_SR_B5' in ele:
                    SC_file = ele
                    b5Name = folder+'/' + folderID + '/' + SC_file
    except:
        b5Name = folder + '/' + folderID + '/' + sceneID + '_SR_B5.TIF'
        
    try:
        files_in_folder = os.listdir(folder+'/'+folderID)
        for ele in files_in_folder:
                if '_SR_B6' in ele:
                    SC_file = ele
                    b6Name = folder+'/' + folderID + '/' + SC_file
    except:
        b6Name = folder + '/' + folderID + '/' + sceneID + '_SR_B6.TIF'
    
    try:
        files_in_folder = os.listdir(folder[:-1]+'2/'+folderID)
        for ele in files_in_folder:
                if 'ST_B10' in ele:
                    SC_file = ele
                    SCName = folder[:-1]+'2/' + folderID + '/' + SC_file
    except:
        SCName =  'none'
        
    try:
        files_in_folder = os.listdir(folder[:-1]+'2/'+folderID)
        for ele in files_in_folder:
                if 'ST_QA.TIF' in ele:
                    QA_file = ele
                    QAName = folder[:-1]+'2/' + folderID + '/' + QA_file
    except:
        QAName =  'none'
        
    try:
        files_in_folder = os.listdir(folder[:-1]+'2/'+folderID)
        for ele in files_in_folder:
                if 'ST_CDIST' in ele:
                    CDIST_file = ele
                    CDISTName = folder[:-1]+'2/' + folderID + '/' + CDIST_file
    except:
        CDISTName =  'none'
    
    # download aster stacked files
    #pdb.set_trace()
    
    gdal.UseExceptions()
    rd=gdal.Open(asterName)
    e13= rd.GetRasterBand(1)
    e13_data = e13.ReadAsArray()
    e14= rd.GetRasterBand(2)
    e14_data = e14.ReadAsArray()
    ndvi= rd.GetRasterBand(3)
    ndvi_data = ndvi.ReadAsArray()
    e13_std= rd.GetRasterBand(4)
    e13_std_data = e13_std.ReadAsArray()
    e14_std= rd.GetRasterBand(5)
    e14_std_data = e14_std.ReadAsArray()
    
    where_are_NaNs = np.isnan(ndvi_data)
    ndvi_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e13_data)
    e13_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e14_data)
    e14_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e13_std_data)
    e13_std_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(e14_std_data)
    e14_std_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b3Name)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    b3= rd.GetRasterBand(1)
    b3_data = b3.ReadAsArray()
    
    where_are_NaNs = np.isnan(b3_data)
    b3_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b4Name)
    b4= rd.GetRasterBand(1)
    b4_data = b4.ReadAsArray()
    
    where_are_NaNs = np.isnan(b4_data)
    b4_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b5Name)
    b5= rd.GetRasterBand(1)
    b5_data = b5.ReadAsArray()
    
    where_are_NaNs = np.isnan(b5_data)
    b5_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b6Name)
    b6= rd.GetRasterBand(1)
    b6_data = b6.ReadAsArray()
    
    where_are_NaNs = np.isnan(b6_data)
    b6_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b10Name)
    b10= rd.GetRasterBand(1)
    b10_data = b10.ReadAsArray()
    
    where_are_NaNs = np.isnan(b10_data)
    b10_data[where_are_NaNs] = 0
    
    gdal.UseExceptions()
    rd=gdal.Open(b11Name)
    b11= rd.GetRasterBand(1)
    b11_data = b11.ReadAsArray()
    
    where_are_NaNs = np.isnan(b11_data)
    b11_data[where_are_NaNs] = 0
    
    try:
        gdal.UseExceptions()
        rd=gdal.Open(SCName)
        ST= rd.GetRasterBand(1)
        ST_data = ST.ReadAsArray()        
        where_are_NaNs = np.isnan(ST_data)
        ST_data[where_are_NaNs] = 0
        ST_data = ST_data * 0.00341802 + 149.0  # as per MTL file collection 2 level 2 data
    except:
        ST_data = 0
        
    try:
        gdal.UseExceptions()
        rd=gdal.Open(QAName)
        QA= rd.GetRasterBand(1)
        QA_data = QA.ReadAsArray() * 0.01   # scale factor according to DFCB (Coll2 Lev2)     
        where_are_NaNs = np.isnan(QA_data)
        QA_data[where_are_NaNs] = 0
    except:
        QA_data = 0

    try:
        gdal.UseExceptions()
        rd=gdal.Open(CDISTName)
        CDIST= rd.GetRasterBand(1)
        CDIST_data = CDIST.ReadAsArray() * 0.01   # scale factor according to DFCB (Coll2 Lev2)       
        where_are_NaNs = np.isnan(CDIST_data)
        CDIST_data[where_are_NaNs] = 0
    except:
        CDIST_data = 0        
    
    #bqaName = '/dirs/data/tirs/downloads/test/LC08_L1TP_026028_20190708_20190719_01_T1/LC08_L1TP_026028_20190708_20190719_01_T1_BQA.TIF'
    #b10Name = '/dirs/data/tirs/downloads/test/LC08_L1TP_026028_20190708_20190719_01_T1/LC08_L1TP_026028_20190708_20190719_01_T1_B10.TIF'
    
    
    try:
    # create cloud mask
        gdal.UseExceptions()
        rd=gdal.Open(bqaName)
        bqa= rd.GetRasterBand(1)
        bqa_data = bqa.ReadAsArray()
        bqa_data = bqa_data.astype(float)
    except:
        bqa_data = np.zeros([b11_data.shape[0],b11_data.shape[1]])
    
    
    
    # https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con 
    
    cloud = [2800,2804,2808,2812,6896,6900,6904,6908]
    
    # note cloud = 0, clear = 1 >> this is easier as a mask to apply
    for val in cloud:
        bqa_data[bqa_data == val] = -10
        
    bqa_data[bqa_data >= 0] = 1
    bqa_data[bqa_data < 1] = 0
    
    dataOut.setRad('rad3',b3_data.astype(float))
    dataOut.setRad('rad4',b4_data.astype(float))
    dataOut.setRad('rad5',b5_data.astype(float))
    dataOut.setRad('rad6',b6_data.astype(float))
    dataOut.setRad('rad10',b10_data.astype(float))
    dataOut.setRad('rad11',b11_data.astype(float))
    dataOut.setRad('cloud',bqa_data.astype(float))
    try:
        dataOut.setRad('SC_LST',ST_data.astype(float))
        dataOut.setRad('SC_QA',QA_data.astype(float))
        dataOut.setRad('SC_CDIST',CDIST_data.astype(float))
    except:
        print('No SC files')
    dataOut.setAster('emis13',e13_data.astype(float))
    dataOut.setAster('emis14',e14_data.astype(float))
    dataOut.setAster('ndvi',ndvi_data.astype(float))
    dataOut.setAster('emis13_std',e13_std_data.astype(float))
    dataOut.setAster('emis14_std',e14_std_data.astype(float))
    
    return dataOut, wkt_projection, geoTransform


# calculate smaller image based on lat lon info and size
def reduceImageSize(image_large, image_small, folder, folderID, sceneID, site_name, image_size_to_save):
    
    
    
    siteInfo = createSiteInfo.CreateSiteInfo()
    
    # find site data based on site name
    line = siteInfo.sitename.index(site_name)
    lat_value = siteInfo.lat[line]
    lon_value = siteInfo.lon[line]
    
    # select reduced size of image for ASTER data to be saved
    num_of_pix = round(image_size_to_save/0.03)  # distance divided by 30m for landsat - need uneven value
    if (num_of_pix % 2) == 0:
        num_of_pix = num_of_pix + 1
      
    
    gdal.UseExceptions()
    rd=gdal.Open(image_large)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    im= rd.GetRasterBand(1)
    im_data = im.ReadAsArray()   

    #pdb.set_trace()
    # find pixel location based on lat lon coordinates of site
    pixelsLocate = subprocess.check_output('gdallocationinfo -wgs84' +' ' + image_large + ' ' + str(lon_value) + ' ' + str(lat_value), shell=True )
   
    temp = pixelsLocate.decode()

    numbers = re.findall('\d+',temp) 

    x = int(numbers[0])
    y = int(numbers[1])

    #get area around pixel of interest and save smaller file to stack aster to 
    x_TL = x-(num_of_pix - 1)/2
    y_TL = y-(num_of_pix - 1)/2
    translate = 'gdal_translate -srcwin %s %s %s %s %s %s' %(x_TL, y_TL, num_of_pix, num_of_pix, image_large, image_small)
    os.system(translate)


# add data for emis and SW calc to object for a single point
def dataForEmisCalc_single(asterName, pathrow, folder, folderID, sceneID, site_name, image_size_to_save):
    
    dataOut = DataLandsatASTER()
    
    band_list = ['B3','B4','B5','B6','B10','B11','BQA']
    data_name = ['rad3','rad4','rad5','rad6','rad10','rad11','cloud']
    
    print(' Resampling down to ' + str(image_size_to_save) + 'km tile...') 
    
    for val, dat in zip(band_list, data_name):
        
        
        
        # open landsat image and reduce size based on smaller size with lat lon in middle
        image_large = folder + '/' + folderID + '/' + sceneID +'_' + val + '.TIF'
        image_small = folder + '/' + folderID + '/' + sceneID +'_' + val + '_sml.TIF'
        
        
        if not os.path.isfile(image_small):
            reduceImageSize(image_large, image_small, folder, folderID, sceneID, site_name, image_size_to_save)
        # open landsat data
        
        gdal.UseExceptions()
        rd=gdal.Open(image_small)
        im= rd.GetRasterBand(1)
        im_data = im.ReadAsArray()
        im_data = im_data.astype(float)
        
        if val == 'BQA':
            
            #pdb.set_trace()
            
            im_data = im_data.astype(int)
            
            #bin(a)
            
            # https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con 
  
            cloud = [2800,2804,2808,2812,6896,6900,6904,6908]
    
            # note cloud = 0, clear = 1 >> this is easier as a mask to apply
            for value in cloud:
                im_data[im_data == value] = -10
        
            im_data[im_data >= 0] = 1
            im_data[im_data < 1] = 0
            
        else:
                
            where_are_NaNs = np.isnan(im_data)
            im_data[where_are_NaNs] = 0
        
        dataOut.setRad(dat,im_data.astype(float))
 
    
    
    gdal.UseExceptions()
    rd=gdal.Open(asterName)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    
    e13 = rd.GetRasterBand(1)
    e13_data = e13.ReadAsArray()
    e14 = rd.GetRasterBand(2)
    e14_data = e14.ReadAsArray()
    ndvi = rd.GetRasterBand(3)
    ndvi_data = ndvi.ReadAsArray()
    e13_std= rd.GetRasterBand(4)
    e13_std_data = e13_std.ReadAsArray()
    e14_std= rd.GetRasterBand(5)
    e14_std_data = e14_std.ReadAsArray()
    

    dataOut.setAster('emis13',e13_data.astype(float))
    dataOut.setAster('emis14',e14_data.astype(float))
    dataOut.setAster('ndvi',ndvi_data.astype(float))
    dataOut.setAster('emis13_std',e13_std_data.astype(float))
    dataOut.setAster('emis14_std',e14_std_data.astype(float))
    
    return dataOut, wkt_projection, geoTransform 
    