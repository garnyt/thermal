#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:51:23 2020

@author: tkpci

Create information to calculate singel point SW data
e.g. SURFRAD, Ameriflux etc...
"""

from siteData import SiteData

def CreateSiteInfo():

    siteInfo = SiteData()
    
    # adding SURFRAD Sioux Falls
    
    siteInfo.addSitename('Sioux_Falls_SD')
    siteInfo.addShortname('sxf')
    siteInfo.addLat(43.73403)
    siteInfo.addLon(-96.62328)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('029030')
    
    # adding SURFRAD Bondville Illinois
    
    siteInfo.addSitename('Bondville_IL')
    siteInfo.addShortname('bon')
    siteInfo.addLat(40.05192)
    siteInfo.addLon(-88.37309)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('022032')
    
    # adding SURFRAD Bondville Illinois
    
    siteInfo.addSitename('Bondville_IL')
    siteInfo.addShortname('bon')
    siteInfo.addLat(40.05192)
    siteInfo.addLon(-88.37309)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('023032')
    
    # adding SURFRAD Desert Rock
    
    siteInfo.addSitename('Desert_Rock_NV')
    siteInfo.addShortname('dra')
    siteInfo.addLat(36.62373)
    siteInfo.addLon(-116.01947)
    siteInfo.addE13(0.9710)
    siteInfo.addE14(0.9650)
    siteInfo.addAsterNDVI(0.1)
    siteInfo.addPathrow('040034')
    
    # adding SURFRAD Desert Rock
    
    siteInfo.addSitename('Desert_Rock_NV')
    siteInfo.addShortname('dra')
    siteInfo.addLat(36.62373)
    siteInfo.addLon(-116.01947)
    siteInfo.addE13(0.9710)
    siteInfo.addE14(0.9650)
    siteInfo.addAsterNDVI(0.1)
    siteInfo.addPathrow('040035')
    
    # adding SURFRAD Fort Peck
    
    siteInfo.addSitename('Fort_Peck_MT')
    siteInfo.addShortname('fpk')
    siteInfo.addLat(48.30783)
    siteInfo.addLon(-105.10170)
    siteInfo.addE13(0.9780)
    siteInfo.addE14(0.9750)
    siteInfo.addAsterNDVI(0.2)
    siteInfo.addPathrow('035026')
    
    # adding SURFRAD Fort Peck
    
    siteInfo.addSitename('Fort_Peck_MT')
    siteInfo.addShortname('fpk')
    siteInfo.addLat(48.30783)
    siteInfo.addLon(-105.10170)
    siteInfo.addE13(0.9780)
    siteInfo.addE14(0.9750)
    siteInfo.addAsterNDVI(0.2)
    siteInfo.addPathrow('035027')
    
    # adding SURFRAD Fort Peck
    
    siteInfo.addSitename('Fort_Peck_MT')
    siteInfo.addShortname('fpk')
    siteInfo.addLat(48.30783)
    siteInfo.addLon(-105.10170)
    siteInfo.addE13(0.9780)
    siteInfo.addE14(0.9750)
    siteInfo.addAsterNDVI(0.2)
    siteInfo.addPathrow('036026')
    
    # adding SURFRAD Goodwin MS
    
    siteInfo.addSitename('Goodwin_Creek_MS')
    siteInfo.addShortname('gwn')
    siteInfo.addLat(34.2547)
    siteInfo.addLon(-89.8729)
    siteInfo.addE13(0.9670)
    siteInfo.addE14(0.9680)
    siteInfo.addAsterNDVI(0.5)
    siteInfo.addPathrow('022036')
    
    # adding SURFRAD Goodwin MS
    
    siteInfo.addSitename('Goodwin_Creek_MS')
    siteInfo.addShortname('gwn')
    siteInfo.addLat(34.2547)
    siteInfo.addLon(-89.8729)
    siteInfo.addE13(0.9670)
    siteInfo.addE14(0.9680)
    siteInfo.addAsterNDVI(0.5)
    siteInfo.addPathrow('023036')
    
    # adding SURFRAD Penn State 
    
    siteInfo.addSitename('Penn_State_PA')
    siteInfo.addShortname('psu')
    siteInfo.addLat(40.72012)
    siteInfo.addLon(-77.93085)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('016032')
    
    # adding SURFRAD Boulder CO 
    
    siteInfo.addSitename('Boulder_CO')
    siteInfo.addShortname('tbl')
    siteInfo.addLat(40.12498)
    siteInfo.addLon(-105.23680)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('034032')
    
    # adding SURFRAD Boulder CO 
    
    siteInfo.addSitename('Boulder_CO')
    siteInfo.addShortname('tbl')
    siteInfo.addLat(40.12498)
    siteInfo.addLon(-105.23680)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('033032')
    
    # adding SURFRAD ARM Southern Great Plains OK
    
    siteInfo.addSitename('ARM_OK')
    siteInfo.addShortname('arm')
    siteInfo.addLat(36.60406)
    siteInfo.addLon(-97.48525)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('028035')
    
    # adding Lake Tahoe buoy 1
    
    siteInfo.addSitename('Tahoe_t1')
    siteInfo.addShortname('tx1')
    siteInfo.addLat(39.155111)
    siteInfo.addLon(-120.004128)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('043033')
    
    # adding Lake Tahoe buoy 2
    
    siteInfo.addSitename('Tahoe_t2')
    siteInfo.addShortname('tx2')
    siteInfo.addLat(39.108767)
    siteInfo.addLon(-120.006167)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('043033')
    
    # adding Lake Tahoe buoy 3
    
    siteInfo.addSitename('Tahoe_t3')
    siteInfo.addShortname('tx3')
    siteInfo.addLat(39.111983)
    siteInfo.addLon(-120.082000)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('043033')
    
    # adding Lake Tahoe buoy 4
    
    siteInfo.addSitename('Tahoe_t4')
    siteInfo.addShortname('tx4')
    siteInfo.addLat(39.178687)
    siteInfo.addLon(-120.103653)
    siteInfo.addE13(1)
    siteInfo.addE14(1)
    siteInfo.addAsterNDVI(1)
    siteInfo.addPathrow('043033')
    
   
    # print out data to see what you have
    
    #siteInfo.toString()
    
    return siteInfo

