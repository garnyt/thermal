#!/usr/bin/env python3
###
#
# Program Description : Store radiance, emissivity Temperature from TIRG data
# Created By          : Tania Kleynhans
# Creation Date       : November 11, 2019
# Authors             : Tania Kleynhans#
# Last Modified By    : Tania Kleynhans
# Last Modified Date  : November 11, 2019
# Filename            : dataTIGR.py
#
###

import numpy as np

class DataTIGR():

    def __init__(self):
        
        self.rad10 = []
        self.rad11 = []
        self.emis10 = []
        self.emis11 = []
        self.skinTemp = []

    def addRad10(self, data):
        self.rad10.append(data)
        
    def getRad10(self):
        return self.rad10
        
    def addRad11(self, data):
        self.rad11.append(data)
        
    def getRad11(self):
        return self.rad11
        
    def addEmis10(self, data):
        self.emis10.append(data)
        
    def getEmis10(self):
        return self.emis10
        
    def addEmis11(self, data):
        self.emis11.append(data)
        
    def getEmis11(self):
        return self.emis11
    
    def addSkinTemp(self, data):
        self.skinTemp.append(data)
        
    def getSkinTemp(self):
        return self.skinTemp
    
    def toString(self):
        print('rad10    ' + str(np.asarray(self.rad10).shape))
        print('rad11    ' + str(np.asarray(self.rad11).shape))
        print('emis10   ' + str(np.asarray(self.emis10).shape))
        print('emis11   ' + str(np.asarray(self.emis11).shape))
        print('skinTemp ' + str(np.asarray(self.skinTemp).shape))

    # Destroy the reader object
    def __del__(self):
        pass