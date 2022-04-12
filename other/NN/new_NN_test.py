#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:56:24 2021

@author: tkpci
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import spectral.io.envi as envi

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from osgeo import gdal

#from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

def save_model(model, savefilename):
    # serialize model to JSON
    model_json = model.to_json()
    
    path = '/cis/staff/tkpci/Code/Python/NN/'
    save_model_json = 'model_1DCNN_{}.json'.format(savefilename)
    save_model_h5 = 'model_1DCNN_{}.h5'.format(savefilename)
    
    with open(path+save_model_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+save_model_h5)
    print("Saved model to disk")


def load_model(savefile):
    
    path = '/cis/staff/tkpci/Code/Python/NN/'
    save_model_json = 'model_1DCNN_{}.json'.format(savefile)
    save_model_h5 = 'model_1DCNN_{}.h5'.format(savefile)
    # load json and create model
    json_file = open(path+save_model_json, 'r')

    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json, custom_objects=None)
    # load weights into new model
    model.load_weights(path+save_model_h5)

    print("Loaded model from disk")
    
    
    return model

def LUT_radiance_apptemp(radiance, band='b10'):
    
    if band == 'b10':
        LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/TIGR/LUT_TIRS10.csv', delimiter=',')
    elif band == 'b11':
        LUT = np.loadtxt('/cis/staff/tkpci/Code/Python/TIGR/LUT_TIRS11.csv', delimiter=',')
        
    rad = np.tile(radiance, (LUT.shape[0],1))
    LUT_rad = np.tile(LUT[:,0], (radiance.shape[0],1))
    LUT_rad = LUT_rad.T
    
    A = np.abs(LUT_rad-rad)
    A = np.matrix(A)
    idx = A.argmin(0)
   
    T = LUT[idx[0,:],1]
    T = np.squeeze(T)

    return T    


def load_data():
    
    #df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_200_30_water_emis_fixed.csv', header=None)
    df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_200_30.csv', header=None)
    #df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_200_30_water_emis_fixed_TlessRange.csv', header=None)
    #df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_all.csv', header=None) 
    
    trainX = pd.DataFrame.to_numpy(df)
    
    trainX = tf.random.shuffle(trainX, seed = None, name = None)
    
    trainX = trainX.numpy()
    
    train_labels = trainX[:,4]
    train_features = trainX[:,0:4]
    
    # if np.mean(train_features[:,1]) < 200:
    
    #     t10 = LUT_radiance_apptemp(trainX[:,0], band='b10')
    #     t11 = LUT_radiance_apptemp(trainX[:,1], band='b11')
    #     train_features[:,0] = t10
    #     train_features[:,1] = t11
    
    
    

    return train_labels, train_features



def model_build(train_labels, train_features):
    
    normalizer = preprocessing.Normalization()
    
    normalizer.adapt(np.array(train_features))
    
    print(normalizer.mean.numpy())
    
    # # linear model RMSE = 0.75
    # linear_model = tf.keras.Sequential([
    #     normalizer,
    #     layers.Dense(units=1)
    # ])
    
    # DNN model RMSE = 0.75 200 * 30 * 7 - fixed learning rate
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(2, activation='relu'),
        #layers.Dropout(0.2),
        #layers.Dense(2, activation='relu'),
        # layers.Dense(2, activation='relu'),
        layers.Dense(1)
    ])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9)
    
    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error')
    
    # linear_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    #     loss='mean_squared_error')
    
    history = linear_model.fit(
        train_features, train_labels, 
        epochs=30,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2)
        
    
    
    test_y = linear_model.predict(train_features)
    print('Standard Deviation train set: ',np.sqrt(mean_squared_error(test_y,train_labels)))
    print('RMSE train set: ',np.std(np.squeeze(test_y)- train_labels))
    print('Mean error train set: ',np.mean(np.squeeze(test_y)- train_labels))
    
    return linear_model

def load_test_and_predict(linear_model):
    
    rd = open('/cis/staff/tkpci/Code/Python/NN/r10_r11_e10_e11_PaperDataAll.csv')
    #rd = open('/cis/staff/tkpci/Code/Python/NN/buoy_NN.csv')


    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    temp = data.astype(float)
    
    trainX = temp[:,0:4]
    
    
    if np.mean(trainX[:,1]) < 200:
    
        t10 = LUT_radiance_apptemp(trainX[:,0], band='b10')
        t11 = LUT_radiance_apptemp(trainX[:,1], band='b11')
        trainX[:,0] = t10
        trainX[:,1] = t11
    
    
    test_y = linear_model.predict(trainX)
    test_y = svr_rbf.predict(trainX)
    
    try:
        label = temp[:,4]
        print('RMSE Surfrad: ',np.sqrt(mean_squared_error(test_y,label)))
        print('Standard Deviation Surfrad: ',np.std(np.squeeze(test_y)- label))
        print('Mean error Surfrad: ',np.mean(np.squeeze(test_y)- label))
    except:
        print('No labels available')
    
    np.savetxt('/cis/staff/tkpci/Code/Python/NN/t1011_e1011_predictions_paper_SVR_all.csv', test_y,delimiter=",")
    
    ave = (t10+t11)/2
    np.savetxt('/cis/staff/tkpci/Code/Python/NN/t1011_e1011_predictions_paper_aveB10B11.csv', ave,delimiter=",")
    

def main():
    
    train_labels, train_features = load_data()
    
    linear_model = model_build(train_labels, train_features)
    
    load_test_and_predict(linear_model)

    save_model(linear_model, 'NN_WD_20210909')
    
    linear_model = load_model('TIGR_LST_1DCNN_20210521')
        
    # test on surfrad data
    
     
    
    linear_model.summary()
    

    # def plot_loss(history):
    #   plt.plot(history.history['loss'], label='loss')
    #   plt.plot(history.history['val_loss'], label='val_loss')
    #   plt.ylim([0, 10])
    #   plt.xlabel('Epoch')
    #   plt.ylabel('Error [MPG]')
    #   plt.legend()
    #   plt.grid(True)
    
    # plot_loss(history)
    
def imageNN(linear_model):

    filepath = '/dirs/data/tirs/downloads/test/LC80160302019103LGN00/'
    filename_T10 = 'LC08_L1TP_016030_20190413_20200829_02_T1_T10.tif'
    filename_T11 = 'LC08_L1TP_016030_20190413_20200829_02_T1_T11.tif'
    filename_e10 = 'LC08_L1TP_016030_20190413_20200829_02_T1_emis10.tif'
    filename_e11 = 'LC08_L1TP_016030_20190413_20200829_02_T1_emis11.tif'
    filename_SW = 'LC08_L1TP_016030_20190413_20200829_02_T1_SW_LST.tif'
    filename_SW_NN = 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST.tif'
    filename_SW_NN_filter = 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST_filter.tif'
 
    gdal.UseExceptions()
    rd=gdal.Open(filepath + filename_T10)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    T10 = data.ReadAsArray()
        
    gdal.UseExceptions()
    rd=gdal.Open(filepath+ filename_T11)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    T11 = data.ReadAsArray()
 
    gdal.UseExceptions()
    rd=gdal.Open(filepath + filename_e10)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    e10= data.ReadAsArray()
    
    gdal.UseExceptions()
    rd=gdal.Open(filepath + filename_e11)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    e11= data.ReadAsArray()
    
    gdal.UseExceptions()
    rd=gdal.Open(filepath + filename_SW)
    wkt_projection =rd.GetProjection()
    geoTransform= rd.GetGeoTransform()
    data= rd.GetRasterBand(1)
    SW= data.ReadAsArray()
    
    NN_LST, wave = readENVI(filepath + 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST.tif.hdr')
    NN_LST_filter, wave = readENVI(filepath + 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST_filter.tif.hdr')
    #NN_LST = np.squeeze(NN_LST)
    # NN_LST = np.where(NN_LST > 500, np.NaN, NN_LST)
    
    # create test dataset
    
    from scipy import signal
    
    kernel = np.ones((5,5),np.float32)/25
    
    T11[T11 == 0] = np.nan
    # create fill value mask to applly to appTemp_bxx_ave after convolution
    
    mask11 = np.isnan(T11)*1   #to get 0-1 values
    
    T11 = signal.convolve2d(T11,kernel,mode='same')
    
    T10 = T10.flatten()
    T11 = T11.flatten()
    e10 = e10.flatten()
    e11 = e11.flatten()
    
    test = np.zeros([T10.shape[0],4])
    test[:,0] = T10
    test[:,1] = T11
    test[:,2] = e10
    test[:,3] = e11
      
    test_y = linear_model.predict(test)
    
    NN_LST = np.reshape(test_y,[8001,7891])
    
    NN_LST = np.where(NN_LST < 209, np.NaN, NN_LST)
    writeENVI(filepath + 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST_no_filter.tif', NN_LST, wavelengths=[])
    
  
    
    plt.subplot(1,3,1)
    plt.imshow(SW-NN_LST_filter, vmin = -0.5, vmax = 0.5)
    plt.colorbar()
    plt.axis('off')
    plt.title('Difference: NN less SW [K]')
    
    
    
    plt.subplot(1,3,2)
    plt.imshow(NN_LST_filter, cmap = 'jet', vmin = 265, vmax = 305)
    plt.colorbar()
    plt.axis('off')
    plt.title('SW LST [K]')
    
    plt.subplot(1,3,3)
    plt.imshow(SW, cmap = 'jet', vmin = 265, vmax = 305)
    plt.colorbar()
    plt.axis('off')
    plt.title('NN LST [K]')
    
    SW[SW > 350] = np.nan
    NN_LST_filter[NN_LST_filter > 350] = np.nan
    NN_LST[NN_LST > 350] = np.nan
    
    SW_ = SW[1500:6500,1500:6500]
    NN_LST_filter_ = NN_LST_filter[1500:6500,1500:6500]
    NN_LST_ = NN_LST[1500:6500,1500:6500]
    
    
    
    SW_ = SW_.flatten()
    NN_LST_filter_ = NN_LST_filter_.flatten()
    NN_LST_ = NN_LST_.flatten()
    
    
    
    sw_filt = np.nanstd(SW_-NN_LST_filter_)
    sw_no = np.nanstd(SW_-NN_LST_)
    NN_std = np.nanstd(NN_LST_filter_-NN_LST_)
    np.nanmean(NN_LST_filter_-NN_LST_)
    
    diff = NN_LST_filter_-NN_LST_
    diff[diff > 1 ] = 0
    diff[diff <-1 ] = 0
    
    diff = NN_LST-NN_LST_filter
    diff = diff[3234:3894, 1948]
    im = plt.imshow(NN_LST-NN_LST_filter, cmap = 'jet', vmin = -2, vmax = 2)
    cb = plt.colorbar().set_label(label='Kelvin',size=20)
    
    
    plt.axis('off')
    plt.title('NN LST [K]')


    
    writeENVI(filepath + 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST.tif', NN_LST2, wavelengths=[])
    writeENVI(filepath + 'LC08_L1TP_016030_20190413_20200829_02_T1_NN_LST_filter.tif', NN_LST2, wavelengths=[])


def writeENVI(full_filename, img, wavelengths):
    
        
    if full_filename[-3:] != 'hdr':
        full_filename = full_filename + '.hdr'
    
    
    header_dict = {}
    header_dict['header offset'] = 0
    header_dict['file type'] = 'ENVI Standard'
    header_dict['data type'] = 4
    header_dict['interleave'] = 'bsq'
    header_dict['sensor type'] = 'Unknown'
    header_dict['byte order'] = 0
    
    header_dict['lines'] = str(img.shape[0])
    header_dict['samples'] = str(img.shape[1])
    try:
        header_dict['bands'] = str(img.shape[2])
    except:
        header_dict['bands'] = str(1)
    try:
        header_dict['wavelength'] = '{'+ ', '.join(str(np.round(x,0)) for x in wavelengths)+'}'
    except:
        header_dict['wavelength'] = []    
    
    envi.save_image(full_filename, img, metadata = header_dict)


def readENVI(full_filename):
    
    # test is file ends with .hdr
    if not full_filename.find('.hdr'):
        full_filename = full_filename + '.hdr'
    
    # read in ENVI data file    
    temp = envi.open(full_filename)
    img = temp.open_memmap(writeable = True)
    
       
    wavelengths = []  
    
    return img, wavelengths  





