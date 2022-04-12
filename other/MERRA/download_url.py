import os
import re
import sys
import shutil
import urllib.error
import urllib.parse
import urllib.request
import warnings
import time
import requests
import pdb



def url_download(url, out_dir, _filename=None, auth=None):
    """ download a file (ftp or http), optional auth in (user, pass) format """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = _filename if _filename else url.split('/')[-1]
    filepath = os.path.join(out_dir, filename)

    if os.path.isfile(filepath):
        return filepath

#    if url[0:3] == 'ftp':
#        download_ftp(url, filepath, shared_args)
#    else:

    download_http(url, filepath, auth)

    return filepath


def download_http(url, filepath, auth=None):
    """ download a http or https resource using requests. """
        
    with requests.Session() as session:
            
        req = session.request('get', url)
        pdb.set_trace()

        if auth:
            resource = session.get(req.url, auth=auth)
    
            if resource.status_code != 200:
                error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                print(error_string)
                        
        else:
            resource = session.get(req.url)
            
            if resource.status_code != 200:
                error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
                print(error_string)
                
            else:
                output_string = "     Session opened successfully"
                print(output_string)
                
        with open(filepath, 'wb') as f:
            output_string = "    Downloading %s " % (filepath[filepath.rfind('/') + 1:])            
            print(output_string)
            
            f.write(resource.content)
            
            output_string = "     Download completed..."
            print(output_string)
                
    return filepath

#
#
#def download_ftp(url, filepath, shared_args):
#    """ download an FTP resource. """
#    
#    total_size = 0
#    
#    try:
#        request = urllib.request.urlopen(url)
#        total_size = int(request.getheader('Content-Length').strip())
#    except urllib.error.URLError as e:
#        print(url)
#        error_string = '\n    url: {0} does not exist, trying other sources\n'.format(url)
#                    
#        raise RemoteFileException(error_string)
#        
#    downloaded = 0
#    filename = filepath[len(filepath) - filepath[::-1].index('/'):]
#
#    with open(filepath, 'wb') as fileobj:        
#        while True:
#                         
#            chunk = request.read(CHUNK)
#            if not chunk:
#                break
#            fileobj.write(chunk)
#            downloaded += len(chunk)
#
#        output_string = "     Download completed..."
#        
#
#    return filepath

