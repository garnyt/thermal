"""
Created on Fri Feb 14 21:37:34 2020

@author:    Tania Kleynhans
@function:  Calculate endmembers using Maximum Distance Method (MaxD) and Gram Matrix
@date:      Spring 2020

"""

import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from tkinter import *   
from tkinter import filedialog
from numpy import matlib as mb
import csv

def readENVI(filepath, filename):
    
    if filename == '':
        full_filename = popup_openfile(filepath)
    
    # test is file ends with .hdr
    if not full_filename.find('.hdr'):
        full_filename = full_filename + '.hdr'
    
    # read in ENVI data file    
    temp = envi.open(full_filename)
    img = temp.open_memmap(writeable = True)
    
    # read in header info
    header_dict = envi_header.read_hdr_file(full_filename)
    # Extract wavelengths, splitting into a list
    wavelengths = header_dict['wavelength'].split(',')   
    
    # Convert from string to float (optional)    
    wave = [float(l) for l in wavelengths] 
    wavelengths = np.array(wave) 
    
    return img, wavelengths

def readTIFF(filepath, filename):
    
    if filename == '':
        full_filename = popup_openfile(filepath)
        
    img = plt.imread(full_filename)
    
    return img

def popup_openfile(initDir):
    
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.filename = filedialog.askopenfilename(initialdir = initDir,title = "Select image file",filetypes = (("hdr file","*.hdr"),("all files","*.*")))
    return root.filename
    root.destroy()
    
 
def maximumDistance(data, num, mnf_data, gram):
    
    # data = 2D data [npixels, nbands]
    # num = number of endmembers to be calculated (choose more than expected to find)
    # if MNF data is not available, code will assign img as mnf_data
    if mnf_data == 0:
        mnf_data = data

    data = np.transpose(data)
    data2 = np.transpose(mnf_data)
    
    # find data size
    num_bands = data.shape[0]
    num_pix = data.shape[1]

    # calculate magnitude of all vectors to find min and max
    magnitude = np.sum(np.square(data), axis=0)
    idx1 = np.int(np.where(magnitude == np.max(magnitude))[0])
    try:
        idx2 = np.int(np.where(magnitude == np.min(magnitude))[0])
    except:
        idx2 = 0    
    
    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num])
    endmembers_index = np.zeros([1,num])
    
    # assign largest and smallest vector as first and second endmembers
    endmembers[:,0] = np.transpose(data[:,idx1])
    endmembers[:,1] = np.transpose(data[:,idx2])
    endmembers_index[0,0] = idx1
    endmembers_index[0,1] = idx2
    
    data_proj = np.matrix(data2)
    identity_matrix = np.identity(num_bands)
    
    # create array for volume of determinant of Gram matrix
    volume = np.zeros([num])
    
    loop = np.arange(3,num+1)    
    for i in loop:
        diff = []
        pseudo = []
        # calc difference between endmembers
        diff = np.matrix(data_proj[:,idx2] - data_proj[:,idx1])
        # caclualte pseudo inverse of difference vector
        pseudo = np.linalg.pinv(diff)       
        data_proj = np.matmul((identity_matrix-np.matmul(diff,pseudo)),data_proj)
        
        idx1 = idx2
        diff_new = np.sum(np.square((np.matmul(data_proj[:,idx2],np.ones([1,num_pix]))-data_proj)),axis=0)
        
        # find ne maximum distance for next endmember
        idx2 = np.int(np.where(diff_new == np.max(diff_new))[1])
        
        # assign to endmember file
        endmembers[:,i-1] = np.transpose(data[:,idx2])
        endmembers_index[0,i-1] = idx2
        
        if gram == 'local':
            # calculate local gram matrix
            loc_gram = calcGramLocal(endmembers, i)
            volume[i-1] = np.sqrt(np.abs(np.linalg.det(loc_gram)))
        
        elif gram == 'general':
            # calculate general gram matrix
            gen_gram = calcGramGeneral(endmembers[:,0:i])
            volume[i-1] = np.sqrt(np.abs(np.linalg.det(gen_gram)))
        
    return endmembers, endmembers_index, volume

    
def calcGramGeneral(data_endmembers):
    
    #calculate gram matrix = V^T * V
    gram = np.matmul(np.transpose(data_endmembers),data_endmembers)

    return gram
    
    
    
def calcGramLocal(data_endmembers, iteration):
    
    # use only endmembers already calculated
    data_endmembers = data_endmembers[:,0:iteration]
    
    # calculate the Gram matrix based on local information (points nearest to mean)
    #num_bands = data_endmembers.shape[0]
    num_pix = data_endmembers.shape[1]
    
    # create mean vector
    mean_spec = np.mean(data_endmembers, axis=1)
    
    # calculate normalized difference between mean vector and endmembers and find closest vector to mean vector
    diffdist = np.linalg.norm(np.transpose(mb.repmat(mean_spec,num_pix,1))-data_endmembers, axis=0)
    min_idx = np.int(np.where(diffdist == np.min(diffdist))[0])

    # create index of rows to keep
    index = np.ones([num_pix])
    # keep all but min distance one
    index[min_idx] = 0
    # find index of all nonzero entires
    keep_idx = np.squeeze(np.where(index == 1))
    nearpix = data_endmembers[:,keep_idx]
    
    # calculate local Gram
    num_neighbors = nearpix.shape[1]
    #gram = np.zeros([num_neighbors, num_neighbors])
    diff_matrix = nearpix - np.transpose(mb.repmat(mean_spec,num_neighbors,1))
    
    gram = np.matmul(np.transpose(diff_matrix),diff_matrix)
    
    return gram
    

def main():

    # open image file: can hardcode filepath and filename, or run without and program will ask for file to open
    filepath =  "/cis/staff2/tkpci/"
    filename = 'emissivities/rock_ems_v2.csv'
    
    #choose between using general gram matrix or local gram matrix
    gram = 'general'
    #gram = 'local'
    
    # if you have mnf data of the image, set mnf_data to that, else code will use image data as mnf_data
    mnf_data = 0
    
    # enter number of endmembers to start off with (this should be more than the expected endmembers)
    num = 20
    
    # choose to read in ENVI or tiff file
    #img, wavelength = readENVI(filepath, filename)
    #img = readTIFF(filepath, filename)        # not yet coded to read wavelengths as not sure how data will look
    
    # img = n x m x z with n and m spatial and z, spectral
    # vectorize the spatial data
#    img_row = img.shape[0]
#    img_col = img.shape[1]
#    img_z = img.shape[2]
#    data = np.reshape(img, [img_row*img_col, img_z], order="F")
#    #data = np.where(data < 0, 0, data)
    
    
    
    data = np.genfromtxt(filepath+filename, delimiter=',')
    wave = data[0,:]
    data = data[1:,:]
    # for emissivity data where first column is wavelength
    #data = np.transpose(data)
    
    
    #endmembers, endmembers_index, volume = maximumDistance(data[1:-1,:], num, mnf_data, gram)
    endmembers, endmembers_index, volume = maximumDistance(data, num, mnf_data, gram)
    
    # normalize volume
    volume_norm = volume/sum(volume)
    
    #plot volume function
    x = np.arange(3,num+1)
    plt.plot(x,volume_norm[2:])
    plt.xlabel('Number of endmembers')
    plt.ylabel('Normalized estimated volume')
    plt.show()
    
    return endmembers, endmembers_index, volume, wave

    
endmembers, endmembers_index, volume, wave = main()

endmembers = np.transpose(endmembers)

plt.figure()

for i in range(endmembers.shape[0]-10):
    plt.plot(wave, endmembers[i,:])
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylim([0,1])
    plt.ylabel('Rock emssivity')
    plt.title('10 Rock emis spectra')
    
#for i in range(data.shape[0]-1):
#    plt.plot(wave, data[i+1,:])

fname = '/cis/staff2/tkpci/emissivities/WaterIceSnowParsedEmissivities_10.csv'
data_ref = np.genfromtxt(fname, delimiter=',')
wave_ref = data_ref[:,0]
emis_spectral = []
    
for i in range(endmembers.shape[0]):
    temp = np.interp(wave_ref, wave, endmembers[i,:])
    emis_spectral.append(temp)
    
emis_spectral = np.asarray(emis_spectral)    
    
emis_spectral = np.vstack([wave_ref,emis_spectral])
emis_spectral = np.transpose(emis_spectral)

savefile = filepath + filename[0:-4]+'_10.csv'

np.savetxt(savefile, emis_spectral, delimiter=',')

#choose number of endmembers to keep based on graph
 num_to_keep = 10
 endmembers = endmembers[0:num_to_keep-1]
 endmembers_index[:,0:num_to_keep-1]




