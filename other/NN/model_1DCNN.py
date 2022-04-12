#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:33:06 2020

@author: tkpci
"""
# adding confusion matrix and only one test set 

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten, BatchNormalization
#from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
#from keras import optimizers
from tensorflow.keras import optimizers
# import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import csv
import tensorflow as tf
import pandas as pd
#from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy.matlib as mb
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


def load_dataset():
    
    
    
    #df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_all.csv', header=None)
    
    df = pd.read_csv('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_skintemp_200_30.csv', header=None)
    trainX = pd.DataFrame.to_numpy(df)
    
    # rd = open('/cis/staff/tkpci/Code/Python/NN/t10_t11_e10_e11.csv')
    # csv_reader = csv.reader(rd)
    # data = list(csv_reader)
    # data = np.asarray(data)
    # trainX = data.astype(float)
    
    trainy = trainX[:,4]
    trainX = trainX[:,0:4]
    
    trainX, trainy = shuffle(trainX, trainy)
    #trainX = np.append(trainX[:,1:3],trainX[:,4:6] , axis=1) 
    
    
    
    
    # adjust temperature to have less weight
    # trainX[:,3] = trainX[:,3] / 15
    # trainX[:,4] = trainX[:,4] / 15
    # trainX[:,5] = trainX[:,5] / 15
    
    # normalize data
    
    # min_temp = np.min(trainX[:,0:2])
    # max_temp = np.max(trainX[:,0:2])
    # min_emis = np.min(trainX[:,2:4])
    # max_emis = np.max(trainX[:,2:4])
    
    # trainX[:,0:2]= (trainX[:,0:2]-min_temp)/(max_temp-min_temp)
    # trainX[:,2:4] = (trainX[:,2:4]-min_emis)/(max_emis-min_emis)
    
    
    # min_label = min(trainy)
    # max_label = max(trainy)
    # trainy = (trainy - min_label)/(max_label - min_label)

    
    # load labels
    # rd_labels = open("/cis/staff/tkpci/Code/Python/NN/label.csv")
    # csv_labels = csv.reader(rd_labels)
    # labels = list(csv_labels)
    # trainy = np.asarray(labels)
    # trainy = trainy.astype(float)
    
    
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


def evaluate_model(X_train, y_train, X_test, y_test):
    
    verbose, epochs, batch_size = 2, 50, 50

    # model = Sequential()
    # model.add(Dense(16, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu'))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    # model.add(Dense(16, input_dim=4, activation='relu'))
    # model.add(Dense(32, input_dim=4, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(12, input_dim=4, activation='relu'))
    # model.add(Dense(8, activation='relu'))

    # #model.add(MaxPool1D(pool_size=2))
    # model.add(Dense(1, activation='relu'))
    
    
    model = Sequential()
    model.add(Dense(4, input_shape=(X_train.shape[1],), kernel_initializer='glorot_normal', activation='relu'))
    #model.add(MaxPool1D(pool_size=(2,)))
    model.add(Dropout(0.2))
    # model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
    # model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu'))
    # model.add(Dropout(0.5))
    #model.add(Dense(12, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(4, activation='relu'))
       
    #model.add(MaxPool1D(pool_size=2))
    model.add(Dense(1, activation='relu'))
    
   
    
    
    adm = optimizers.Adam(learning_rate=0.01) #, decay=0, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['categorical_accuracy'])
    model.compile(loss='mean_squared_error', optimizer=adm)
    # fit model
    loss_list = []
    valloss_list = []

    
    
    es = EarlyStopping(monitor='val_loss',patience=4,verbose=0,mode='auto')
    #es = EarlyStopping(monitor='loss',patience=5,verbose=0,mode='auto')
    for _ in range(4):
        
        #history = model.fit(X_train, y_train,epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=[es])
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=[es])
      
        
        K.set_value(adm.lr, K.get_value(adm.lr)/10)
        #model.optimizer.lr.set_value(np.float32(model.optimizer.lr.get_value()/10.0))
        loss_list.extend(history.history['loss'])
        valloss_list.extend(history.history['val_loss'])
        
        
    # evaluate model
    # pdb.set_trace()
    # results = model.evaluate(X_test, y_test)
    # print(results)
    y_test_pred = model.predict(X_test)
    print('Standard Deviation test set: ',np.sqrt(mean_squared_error(y_test_pred,y_test)))
    print('RMSE test set: ',np.std(np.squeeze(y_test_pred)- y_test))
    print('Mean error test set: ',np.mean(np.squeeze(y_test_pred)- y_test))
    
    plt.scatter(y_test_pred,y_test)
    plt.xlim([200,330])
    plt.ylim([200,330])
    plt.title('Test data')
    plt.ylabel('Truth')
    plt.xlabel('Predicted')
    x = list(range(200, 330))
    y = list(range(200, 330))
    plt.plot(x,y)
    
    y_pred = model.predict(X_train)
    print('Standard Deviation train set: ',np.sqrt(mean_squared_error(y_pred,y_train)))
    print('RMSE train set: ',np.std(np.squeeze(y_pred)- y_train))
    print('Mean error train set: ',np.mean(np.squeeze(y_pred)- y_train))
    
    return model

 
# run an experiment
def run_experiment(savefile):
    # load data
    trainX, trainy = load_dataset()
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(trainX,trainy,test_size=0.3)
    
    X_train = preprocessing.scale(X_train)

    X_test = preprocessing.scale(X_test)

    model = evaluate_model(X_train, y_train, X_test, y_test)
                
    #save_model(model, savefile)
    
    return  model
    

def save_model(model, savefile):
    # serialize model to JSON
    model_json = model.to_json()
    
    path = '/cis/staff/tkpci/Code/Python/NN/'
    save_model_json = 'model_1DCNN_{}.json'.format(savefile)
    save_model_h5 = 'model_1DCNN_{}.h5'.format(savefile)
    
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
    model = tf.keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights(path+save_model_h5)

    print("Loaded model from disk")
    
    
    return model

def testdata():
    rd = open('/cis/staff/tkpci/Code/Python/NN/T10_11_emis10_11_Surfrad3LST.csv')

    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    temp = data.astype(float)
    
    trainX = temp[:,0:4]
    label = temp[:,4]
    
    # normalize data
    
    trainX = preprocessing.scale(trainX)
    
    # min_temp = np.min(trainX[:,0:2])
    # max_temp = np.max(trainX[:,0:2])
    # min_emis = np.min(trainX[:,2:4])
    # max_emis = np.max(trainX[:,2:4])
    
    # trainX[:,0:2]= (trainX[:,0:2]-min_temp)/(max_temp-min_temp)
    # trainX[:,2:4] = (trainX[:,2:4]-min_emis)/(max_emis-min_emis)
    
    # min_emis = np.min(trainX[:,0:2])
    # max_emis = np.max(trainX[:,0:2])
    # min_rad = np.min(trainX[:,2:4])
    # max_rad = np.max(trainX[:,2:4])
    
    # trainX[:,0:2]= (trainX[:,0:2]-min_emis)/(max_emis-min_emis)
    # trainX[:,2:4] = (trainX[:,2:4]-min_rad)/(max_rad-min_rad)
    
   

    return trainX, label

def main():
    
    savefile = '/cis/staff/tkpci/Code/Python/NN/TIGR_LST_20210603_200_30_T1011_e1011'
        
    # train model
    model = run_experiment(savefile)
    
    model = load_model(savefile)
    
    trainX, label = testdata()
    
    # predict pigments
    prediction = model.predict(trainX)
    print('Standard Deviation test set: ',np.sqrt(mean_squared_error(prediction[:,0],label)))
    print('RMSE test set: ',np.std(np.squeeze(prediction[:,0])- label))
    print('Mean error test set: ',np.mean(np.squeeze(prediction[:,0])- label))
    
    
    np.savetxt('/cis/staff/tkpci/Code/Python/NN/SurfRad3_t1011_e1011_predictions.csv', prediction,delimiter=",")

def check_data(X_train):
    
    
    
    for i in range(4):
        plt.figure()
        plt.hist(X_test[:,i],bins=50)
        
    plt.scatter(trainX[:,2],trainX[:,3])
    
    
    