#!/usr/bin/env python3
###
#
# Program Description : Store Landsat data and aster emissivity and NDVI for emissivity calculations
# Created By          : Tania Kleynhans
# Creation Date       : November 15, 2019
# Authors             : Tania Kleynhans#
# Last Modified By    : Tania Kleynhans
# Last Modified Date  : November 15, 2019

###

import numpy as np

class DataLandsatASTER():

    def __init__(self):
        
        self.rad = {
                'rad3': [],
                'rad4': [],
                'rad5': [],
                'rad6': [],
                'rad10': [],
                'rad11': [],
                't10': [],
                't11': [],
                'e10': [],
                'e11': [],
                'cloud': [],
                'SW_LST': [],
                'SW_LST_CWV': [],
                'SW_error': [],
                'SC_LST': [],
                'SC_QA': [],
                'SC_CDIST': []
        }
        
        self.aster = {
                'emis13': [],
                'emis14': [],
                'ndvi': [],
                'emis13_std': [],
                'emis14_std': []
        }
        
    def setRad(self, key, value):
        self.rad[key] = value

    # Use this to add an additional band to the rad dictionary
    def addRad(self, key, value):
        self.rad[key].append(value)
        
    def getRad(self, key):
        return np.asarray(self.rad[key])
    
    def setAster(self, key, value):
        self.aster[key] = value
    
    def addAster(self, key, value):
        self.aster[key].append(value)
        
    def getAster(self, key):
        return np.asarray(self.aster[key])
    
    def toString(self):
        print('rad3    ' + str(np.shape(self.getRad('rad3'))))
        print('rad4    ' + str(np.shape(self.getRad('rad4'))))
        print('rad5    ' + str(np.shape(self.getRad('rad5'))))
        print('rad6    ' + str(np.shape(self.getRad('rad6'))))
        print('rad10    ' + str(np.shape(self.getRad('rad10'))))
        print('rad11    ' + str(np.shape(self.getRad('rad11'))))
        print('t10   ' + str(np.shape(self.getRad('t10'))))
        print('t11    ' + str(np.shape(self.getRad('t11'))))
        print('e10 ' + str(np.shape(self.getRad('e10'))))
        print('e11 ' + str(np.shape(self.getRad('e11'))))
        print('cloud    ' + str(np.shape(self.getRad('cloud'))))
        print('SW_LST    ' + str(np.shape(self.getRad('SW_LST'))))
        print('SW_LST_CWV' + str(np.shape(self.getRad('SW_LST_CWV'))))
        print('SW_error    ' + str(np.shape(self.getRad('SW_error'))))
        print('SC_LST    ' + str(np.shape(self.getRad('SC_LST'))))
        print('SC_QA    ' + str(np.shape(self.getRad('SC_QA'))))
        print('SC_CDIST    ' + str(np.shape(self.getRad('SC_CDIST'))))
        print('aster_emis13   ' + str(np.shape(self.getAster('emis13'))))
        print('aster_emis14   ' + str(np.shape(self.getAster('emis14'))))
        print('aster_ndvi ' + str(np.shape(self.getAster('ndvi'))))
        print('aster_emis13_std ' + str(np.shape(self.getAster('emis13_std'))))
        print('aster_emis14_std ' + str(np.shape(self.getAster('emis14_std'))))

    # Destroy the reader object
    def __del__(self):
        pass
