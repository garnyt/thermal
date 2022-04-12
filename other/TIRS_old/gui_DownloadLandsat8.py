#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:38:41 2019

@author: tkpci
"""


from tkinter import *   
from tkinter import filedialog 
import downloadLandsat
import csv
import gui_CalculateEmisAndSWLST
import warnings
import pdb


def popup_choose_download():      
    # create a tkinter window to upload scene list, ID or lat lon
    root = Tk() 
    root.title('Choose list or enter sceneID for download(s)')            
      
    # Open window 
    root.geometry('400x150+200+200')
        
    # Create a Tkinter variable
    tkvar = StringVar(root)
    
    # Dictionary with options
    choices = { 'Upload SceneID list','Enter single SceneID'}
    tkvar.set('Scene select options') # set the default option
    
    def execute_choice(value):
        global chosen
        chosen = value
        root.destroy()
    
    
    popupMenu = OptionMenu(root, tkvar, *choices, command=execute_choice)    
    popupMenu.place(x=120, y=20)
    #popupMenu.grid(row = 2, column =1)
    
    root.mainloop() 

def popup_upload_list():
    
    root = Tk()
    root.withdraw()
    #root.place(x=120, y=20)
    root.filename = filedialog.askopenfilename(initialdir = "/dirs/data/tirs/downloads/",title = "Select file",filetypes = (("csv file","*.csv"),("all files","*.*")))
    return root.filename
    root.destroy()

def popup_get_sceneID():
    
    def clear_search(event):
        ipt.delete(0, END) 
    
    def get_data():
        global sceneID
        sceneID = ipt.get()
        root.destroy()
               
    root = Tk()
    root.title('Enter SceneID')  
              
    # Open window 
    root.geometry('350x150+100+100') 
    ipt = Entry(root, width=30)
    ipt.place(x=50, y= 20)
    ipt.insert(0,'Enter SceneID here')
    ipt.bind("<Button-1>", clear_search) 
    b = Button(root, text="run", width=10, command=get_data)
    b.place(x=50, y= 70)
    
    mainloop()
    
    return sceneID


def popup_get_latlon():
    
    def clear_search_lat(event):
        lat.delete(0, END) 
    def clear_search_lon(event):
        lon.delete(0, END) 
    
    def get_data():
        global lat_value
        lat_value = lat.get()
        global lon_value
        lon_value = lon.get()
        root.destroy()
        
    root = Tk()
    root.title('Enter Let Lon')  
              
    # Open window 
    root.geometry('380x150+100+100') 
    lat = Entry(root, width=15)
    lat.place(x=50, y= 30)
    lat.insert(0,'Lat e.g. 43.1566')
    lat.bind("<Button-1>", clear_search_lat) 
    #lat_value = lat.get()
    lon = Entry(root, width=15)
    lon.place(x=200, y= 30)
    lon.insert(0,'Lon e.g. -77.6088')
    lon.bind("<Button-1>", clear_search_lon) 
    #lon_value = lon.get()
    b = Button(root, text="run", width=10, command=get_data)
    b.place(x=140, y= 70)
    
    mainloop()


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

############################### RUN FUNCTIONS ######################   
warnings.simplefilter('ignore')


chosen = 'none'        
popup_choose_download()

if chosen == 'Upload SceneID list':
    filename = popup_upload_list()
    temp = open(filename)
    csv_reader = csv.reader(temp)
    sceneID = list(csv_reader)
    sceneID = flatten(sceneID)
    folder = downloadLandsat.downloadL8data(sceneID,paths=0, rows=0)
elif chosen == 'Enter single SceneID':
    popup_get_sceneID()
    sceneID = [sceneID]
    folder = downloadLandsat.downloadL8data(sceneID,paths=0, rows=0)
else:
    print('Aborted')
    
# calculate SW for files
#gui_CalculateEmisAndSWLST.calcEmisAndSW()    


# next steps... 


# test data below - they are already downloaded, so should not run long
#latlon = [43.1566,-77.6088]
#sceneID=['LC80030172015001LGN00']
#sceneID=['LC08_L1TP_003017_20150101_20170415_01_T1']



