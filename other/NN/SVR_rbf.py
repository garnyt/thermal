#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:49:51 2021

@author: tkpci
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import time
import csv
import sklearn



def load_dataset(reduced = True, SW_only = False):
    
    
    # if reduced:
    #     df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/emis09_10_11_rad09_10_11_skintemp_2311_30.csv', header=None)
    #     #df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/emis09_10_11_rad09_10_11_skintemp_reduced.csv', header=None)
    #     trainX = pd.DataFrame.to_numpy(df)
    # else:
    #     df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/emis09_10_11_rad09_10_11_skintemp.csv', header=None)
    #     trainX = pd.DataFrame.to_numpy(df)
    
        
    # trainy = trainX[:,6]
    # trainX = trainX[:,0:6]
    
    # if SW_only:
    #     trainX = np.append(trainX[:,1:3],trainX[:,4:6] , axis=1) 
    
    
    df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_200_30.csv', header=None)
    trainX = pd.DataFrame.to_numpy(df)
    
    # rd = open('/cis/staff/tkpci/Code/Python/NN/t10_t11_e10_e11.csv')
    # csv_reader = csv.reader(rd)
    # data = list(csv_reader)
    # data = np.asarray(data)
    # trainX = data.astype(float)
    
    trainy = trainX[:,4]
    trainX = trainX[:,0:4]
    
    
    
    # adjust temperature to have less weight
    # trainX[:,3] = trainX[:,3] / 15
    # trainX[:,4] = trainX[:,4] / 15
    # trainX[:,5] = trainX[:,5] / 15
    
    # normalize data
    
    # min_emis = np.min(trainX[:,1:3])
    # max_emis = np.max(trainX[:,1:3])
    # min_rad = np.min(trainX[:,4:6])
    # max_rad = np.max(trainX[:,4:6])
    
    # trainX[:,0:3] = (trainX[:,0:3]-min_emis)/(max_emis-min_emis)
    # trainX[:,3:6] = (trainX[:,3:6]-min_rad)/(max_rad-min_rad)
    
    
    # min_label = min(trainy)
    # max_label = max(trainy)
    # # trainy = (trainy - min_label)/(max_label - min_label)

    
    # # Randomize data    
    # temp = np.zeros([trainX.shape[0],trainX.shape[1]+1])
    # temp[:,0:trainX.shape[1]] = trainX
    # temp[:,trainX.shape[1]] = np.squeeze(trainy)
    
    # #shuffle data
    # temp =  temp[np.random.permutation(temp.shape[0]),:]
    # trainX = temp[:,0:temp.shape[1]-1]
    # trainy = temp[:,temp.shape[1]-1]
    
    # scale1= preprocessing.scale(trainX[:,0:3])
    # scale2= preprocessing.scale(trainX[:,3:6])
    # trainy = trainy/350
    
    # trainX[:,0:3] = scale1
    # trainX[:,3:6] = scale2
    
    print(trainX.shape, trainy.shape)
   
   
    return trainX, trainy


def main():
    
    # load dataset for training
    trainX,trainy = load_dataset(reduced = True, SW_only = True)
    
    X_train, X_test, y_train, y_test = train_test_split(trainX,trainy,test_size=0.1)

    start_time = time.time()
    # create support vector regression model with radial basis kernel (for non-linear model)
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.5)
    
    
    
    # build model and predict
    pred_y = svr_rbf.fit(X_train, y_train).predict(X_train)
    
    # end = time.time()
    # print('Runtime of the program is (min) ', round((end - start_time)/60,2))
    
    rmse = sqrt(mean_squared_error(y_train, pred_y))
    print('Model RMSE [K]: ', np.round(rmse,2))
    
    # predict
    # pred_y = svr_rbf.predict(X_test)
    # sqrt(mean_squared_error(y_test, pred_y))

    # load full dataset - see how model performs
    X_all,y_all = load_dataset(reduced = False, SW_only = False)
    #X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,test_size=0.5)
 
    start_time = time.time()
    # model accuracy on full dataset 
    pred_y_test = svr_rbf.predict(X_all)
    rmse_test = sqrt(mean_squared_error(y_all, pred_y_test))
    
    print('Test RMSE [K]: ', np.round(rmse_test,2))
    
    
    end = time.time()
    print('Runtime of the program is (min) ', round((end - start_time)/60,2))
    
    # test on surfrad data

    rd = open('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_Surfrad3LST.csv')
    
    rd = open('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_BuoyOceanLSTLST.csv')
    
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    temp = data.astype(float)
    
    trainX = temp[:,0:4]
    label = temp[:,4]
    
    
    test_y = svr_rbf.predict(trainX)
    print('Standard Deviation Surfrad: ',np.sqrt(mean_squared_error(test_y,label)))
    print('RMSE Surfrad: ',np.std(np.squeeze(test_y)- label))
    print('Mean error Surfrad: ',np.mean(np.squeeze(test_y)- label))
    
    np.savetxt('/cis/staff/tkpci/Code/Python/NN/SurfRad3_t1011_e1011_predictions_SVR.csv', test_y,delimiter=",")
    np.savetxt('/cis/staff/tkpci/Code/Python/NN/SurfRad3_t1011_e1011_predictions_SVR_buoy.csv', test_y,delimiter=",")


