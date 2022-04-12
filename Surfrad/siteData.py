#!/usr/bin/env python3
###
#
# Program Description : Store radiance, emissivity Temperature from TIRG data
# Created By          : Tania Kleynhans
# Creation Date       : November 11, 2019
# Authors             : Tania Kleynhans#
# Last Modified By    : Tania Kleynhans
# Last Modified Date  : December 19, 2019
# Filename            : siteData.py
#
###

import numpy as np

class SiteData():

    def __init__(self):
        
        self.sitename = []
        self.shortname = []
        self.lat = []
        self.lon = []
        self.e13 = []
        self.e14 = []
        self.asterNDVI = []
        self.pathrow = []

    def addSitename(self, data):
        self.sitename.append(data)
        
    def getSitename(self):
        return self.sitename
        
    def addShortname(self, data):
        self.shortname.append(data)
        
    def getShortname(self):
        return self.shortname
        
    def addLat(self, data):
        self.lat.append(data)
        
    def getLat(self):
        return self.lat
        
    def addLon(self, data):
        self.lon.append(data)
        
    def getLon(self):
        return self.lon

    def addE13(self, data):
        self.e13.append(data)
        
    def getE13(self):
        return self.e13
    
    def addE14(self, data):
        self.e14.append(data)
        
    def getE14(self):
        return self.e14

    def addAsterNDVI(self, data):
        self.asterNDVI.append(data)
        
    def getAsterNDVI(self):
        return self.asterNDVI
    
    def addPathrow(self, data):
        self.pathrow.append(data)
        
    def getPathrow(self):
        return self.pathrow

    
    def toString(self):
        print('sitename    ' + str(self.sitename))
        print('shortname    ' + str(self.shortname))
        print('lat   ' + str(self.lat))
        print('lon   ' + str(self.lon))
        print('e13   ' + str(self.e13))
        print('e14   ' + str(self.e14))
        print('asterNDVI   ' + str(self.asterNDVI))
        print('pathrow   ' + str(self.pathrow))

    # Destroy the reader object
    def __del__(self):
        pass
