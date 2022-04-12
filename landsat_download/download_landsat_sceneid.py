# =============================================================================
#  USGS/EROS Inventory Service Example
#  Python - JSON API
# 
#  Script Last Modified: 6/17/2020
#			 1/20/2020   ngrpci added buoy formatting and specific Landsat products and download 
#			 3/14/2022   ngrpci added request by scene id for Tania
#  Note: This example does not include any error handling!
#        Any request can throw an error, which can be found in the errorCode proprty of
#        the response (errorCode, errorMessage, and data properies are included in all responses).
#        These types of checks could be done by writing a wrapper similiar to the sendRequest function below

#  Usage: python download_L8_C2_L1.py -u username -p password
#           for buoy code                          -y 2020 -m 09 -d 22 -la 43.6 -lo -77.4 -pa 16 -r 30
#
#  Test:  python L8_C2_L2_scenedownload_sceneid.py -u EEusername -p EEpassword -y 2020 -m 09 -d 22 -la 43.6 -lo -77.4 -pa 16 -r 30
#
#  Usage:  FOR just by SCENEID
#	   python  download_L9_C2_L1_sceneid.py -u EEusername -p EEpassword   -id LC90160302022055LGN00
# =============================================================================

import json
import requests
import sys
import time
import argparse
import os
import shutil
import pdb
import datetime


# send http request
def sendRequest(url, data, apiKey = None):  
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers = headers)    


    try:
      httpStatusCode = response.status_code 
      if response == None:
          print("No output from service")
          sys.exit()
      output = json.loads(response.text)	
      if output['errorCode'] != None:
          print(output['errorCode'], "- ", output['errorMessage'])
          sys.exit()
      if  httpStatusCode == 404:
          print("404 Not Found")
          sys.exit()
      elif httpStatusCode == 401: 
          print("401 Unauthorized")
          sys.exit()
      elif httpStatusCode == 400:
          print("Error Code", httpStatusCode)
          sys.exit()
    except Exception as e: 
          response.close()
          print(e)
          sys.exit()
    response.close()
    
    return output['data']


def product2entityid(product_id):
    """ convert product landsat ID to entity ID

    Ex:
    LC08_L1TP_017030_20131129_20170307_01_T1 ->
    LC80170302013333LGN01
    """
    
    collection = '00'
    
    if len(product_id) == 21:
        return product_id
    else:
        path = product_id[10:13]
        row = product_id[13:16]
        lnum = product_id[3:4]
    
        date = datetime.datetime.strptime(product_id[17:25], '%Y%m%d')
    
    return 'LC{lnum}{path}{row}{date}LGN{coll}'.format(lnum = lnum, path=path, row=row, date=date.strftime('%Y%j'), coll=collection)


def main(scene_id,filepath,datasetName = 'landsat_ot_c2_l1'): 
       
    
    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    
    # login
    loginParameters = {'username' : 'cisthermal', 'password' : 'C15Th3rmal'}
    
    apiKey = sendRequest(serviceUrl + "login", loginParameters)
    

    # ------------------------------------------------------  
    # SET PRODUCT NAME HERE  datasetName
    # other options "landsat_ot_c2_l1" "landsat_ot_c2_l2" "landst_etm_c2_l1"
    # ------------------------------------------------------
    #datasetName = "landsat_ot_c2_l1"


    datasetSearchParameters = {'datasetName' : datasetName, 'entityIds' :  scene_id }                     
    datasets = sendRequest(serviceUrl + "dataset-search", datasetSearchParameters, apiKey)
   
    
    # download datasets
    for dataset in datasets:
        
        for scene in scene_id:
            
            try:
                
                if len(scene) > 21:
                    scene = product2entityid(scene)
                    
                sceneSearchParameters = {'datasetName' : dataset['datasetAlias'], 'entityIds' :  scene }

                # download options
                scenes = sendRequest(serviceUrl + "download-options", sceneSearchParameters, apiKey);
        
                downloadOptionsParameters = {'datasetName' : dataset['datasetAlias'],
                                                 'entityIds' : scene}
                                
                downloadOptions = sendRequest(serviceUrl + "download-options", downloadOptionsParameters, apiKey)
       
                downloads = []
                for product in downloadOptions:
                	# Make sure the product is available for this scene
                       	if product['available'] == True:
                               downloads.append({'entityId' : product['entityId'],'productId' : product['id']})

                label = "download-sample"
                downloadRequestParameters = {'downloads' : [downloads[0]],
                                                         'label' : label}
                # Call the download to get the direct download urls
                requestResults = sendRequest(serviceUrl + "download-request", downloadRequestParameters, apiKey)             
	
                avail = requestResults['availableDownloads']
	    
                url_n = avail[0]	

                download_call = 'wget -O '+scene+'.tar.gz "'+url_n['url']+'"'
                
                os.system(download_call)
                
                os.mkdir(filepath+scene)
                
                # untar files
                os.system("tar -xvf ./"+ scene +".tar.gz "+ " -C " +filepath+scene)
                os.system("rm ./" + scene +".tar.gz")
            except:
                print('Download failed: ', scene)
	
   	# Logout so the API Key cannot be used anymore
    endpoint = "logout"  
    if sendRequest(serviceUrl + endpoint, None, apiKey) == None:        
        print("Logged Out\n\n")
    else:
        print("Logout Failed\n\n")            
    
