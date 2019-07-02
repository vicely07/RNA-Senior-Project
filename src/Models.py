# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

import sys

from numpy import array

from numpy import argmax

from keras.utils import to_categorical

import numpy as np

import string

import talos as ta

from talos.model.normalizers import lr_normalizer




from sklearn.metrics import roc_curve, auc

#training

import numpy as np

#simplest model in keras

from keras.models import Sequential

from keras.layers.core import  Dense, Activation, Flatten, Dropout

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD, RMSprop, Adam

from keras.utils import np_utils

#convolutional layers

from keras.layers.convolutional import Conv2D,Conv1D

from keras.layers.convolutional import MaxPooling2D,MaxPooling1D

from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,AveragePooling1D

from keras.layers import Bidirectional

from keras.models import load_model

#plotting

#import matplotlib.pyplot as plt



np.random.seed(1671)





# network and training

NB_EPOCH = 150

BATCH_SIZE = 50

VERBOSE = 1

NB_CLASSES = 2 # number of classes

OPTIMIZER = RMSprop(lr=0.05)

N_HIDDEN = 128

VALIDATION_SPLIT = 0.1

METRICS =['accuracy']

LOSS = 'binary_crossentropy'

#DropOut1 = 0.2

#DropOut2 = 0.3

#image rows and columns size

IMG_ROWS=1662

IMG_CHANNELS = 4 #color rgb



#if data is the format of sample size x image rows x image cols x channels

#INPUT_SHAPE = (IMG_ROWS, IMG_COLS,IMG_CHANNELS)

#if data is the format of sample size x x channels x image rows x image cols

#INPUT_SHAPE = (IMG_CHANNELS,IMG_ROWS, IMG_COLS)

#kernel initializer

KERNEL_INITIAL ='glorot_uniform'







#stop training if

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint # to save models at each epoch

#Stop training when a monitored quantity has stopped improving.

#patience: number of epochs with no improvement after which training will be stopped.





from keras.layers import SimpleRNN

from keras.layers import LSTM

from keras.layers import Reshape

from keras.constraints import maxnorm

#y=to_categorical(y,2)

from keras.callbacks import Callback

from sklearn.metrics import roc_auc_score

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import classification_report



def GetMetrics(model,x,y):

    pred = model.predict_classes(x)

    pred_p=model.predict(x)

    fpr, tpr, thresholdTest = roc_curve(y, pred_p)

    aucv = auc(fpr, tpr) 

    #print('auc:',aucv)

    print('auc,acc,mcc',aucv,accuracy_score(y,pred),matthews_corrcoef(y,pred))

    print(classification_report(y,pred))

    #print('mcc:',matthews_corrcoef(y,pred))

    

class roc_callback(Callback):

     def __init__(self,training_data,validation_data):

         self.x = training_data[0]

         self.y = training_data[1]

         self.x_val = validation_data[0]

         self.y_val = validation_data[1]





     def on_train_begin(self, logs={}):

         return



     def on_train_end(self, logs={}):

         return



     def on_epoch_begin(self, epoch, logs={}):

         return



     def on_epoch_end(self, epoch, logs={}):

         y_pred = self.model.predict(self.x)

         roc = roc_auc_score(self.y, y_pred)

         y_pred_val = self.model.predict(self.x_val)

         roc_val = roc_auc_score(self.y_val, y_pred_val)

         print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')

         return

#

     def on_batch_begin(self, batch, logs={}):

         return

#

     def on_batch_end(self, batch, logs={}):

         return





#perform prediction on the test data

def deepPredict(model,Test_Predictors,Test_class):

    score = model.evaluate(Test_Predictors,Test_class)

    print("Test score: ", score[0] )

    print("Test accuracy: ", score[1])



def smallRNN(x_train,y_train, x_test, y_test,params): 

    #HIDDEN_UNITS = 5

    model = Sequential()

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3),input_shape =x_train.shape[1:],merge_mode='concat'))

        #model.add(AveragePooling1D(pool_size=400,strides=400))

        #model.add(Bidirectional(LSTM(HIDDEN_UNITS,activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True),input_shape=x_train.shape[1:],merge_mode='concat'))

        #model.add(AveragePooling1D(pool_size=400,strides=400))

    model.add(Flatten())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    #model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    filepath="SmallRNN"+str(params["hidden_unit"])+"best_smallRNN.hdf5"

    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True)

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics=METRICS)

    print(model.summary())

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = 0.2,callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    ## if you want early stopping

    #Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                                #validation_split = VALIDATION_SPLIT,callbacks=[early_stopping_monitor,checkpoint])



    #finalModel = load_model(filepath)

    #deepPredict(finalModel,Test_Predictors,Test_class)

    return out, model #,roc_train,roc_test,acc_train,acc_test





def mediumRNN(x_train,y_train, x_test, y_test,params):

    #HIDDEN_UNITS = 15

    model = Sequential()

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3),input_shape =x_train.shape[1:]))

    model.add(MaxPooling1D(pool_size=10,strides=5))


    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True),input_shape=x_train.shape[1:]))

    model.add(MaxPooling1D(pool_size=10,strides=5))

    model.add(Flatten())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    #filepath="mediumRNN_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    filepath="mediumRNN"+"best_mediumRNN_dropout_"+str(params["hidden_unit"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    #model.compile(loss=LOSS, optimizer = 'Adam', metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_data = (x_test,y_test),callbacks=[checkpoint,early_stopping_monitor]) #,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)
    

    return out, model


def largeRNN(x_train,y_train, x_test, y_test,params):

    HIDDEN_UNITS = 20

    model = Sequential()

    model.add(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3,input_shape =x_train.shape[1:]))
    
    model.add(MaxPooling1D(pool_size=10,strides=5))

    model.add(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3))


    model.add(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,input_shape=x_train.shape[1:]))

    model.add(MaxPooling1D(pool_size=10,strides=5))

    model.add(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True))

    model.add(Flatten())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    #filepath="largeRNN_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    filepath="largeRNN"+"best_largeRNN_dropout_"+str(params["hidden_unit"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    #model.compile(loss=LOSS, optimizer = 'Adam', metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_data = (x_test,y_test),callbacks=[checkpoint,early_stopping_monitor]) #,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print(model.summary())

    return out, model



def deepBind(x_train,y_train,x_test,y_test,DROP_OUT,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,name):

    x_val = x_test[range(0,x_test.shape[0],2),:] 

    y_val = y_test[range(0,y_test.shape[0],2)]

    x_test = x_test[range(1,x_test.shape[0],2),:]

    y_test = y_test[range(1,y_test.shape[0],2)]

    

    model = Sequential()

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = INPUT_SHAPE))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(GlobalMaxPooling1D())

    #model.add(MaxPooling1D())

    #HIDDEN_UNITS=20

    #model.add(Bidirectional(LSTM(HIDDEN_UNITS,activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3)))

    #model.add(Flatten())

    #model.add(Dense(1))

    #model.add(BatchNormalization())

    #model.add(Activation('relu')) #14

    #model.add(BatchNormalization())

    #model.add(Dropout(0.5))

    #model.add(BatchNormalization())

    #model.add(Activation("relu"))

    model.add(Dense(1))

    model.add(Activation("sigmoid"))

    filepath=name+"best_deepBind_dropout_"+str(NUM_KERNEL)+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)

    #print(model.summary())

    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH,

                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])  #,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print("train") 

    GetMetrics(load_model(filepath),x_train,y_train)

    print("test")

    GetMetrics(load_model(filepath),x_test,y_test)

    print("val")

    GetMetrics(load_model(filepath),x_val,y_val)

    return model #,Tuning#,roc_train,roc_test,acc_train,acc_test


def miniCNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    #model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    #model.add(BatchNormalization())
    
    model.add(GlobalMaxPooling1D())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    #filepath="smallCNN_dropout_"+str(DROP_OUT)+".hdf5"

    filepath="MiniCNN"+"best_miniCNN_dropout_"+str(params["num_kernel"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS) 

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = 0.2,callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)



    #finalmodel = load_model(filepath)

    return out, model #,roc_train,roc_test,acc_train,acc_test



def smallCNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    #model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    #model.add(BatchNormalization())

    

    model.add(MaxPooling1D())

    model.add(Conv1D(20,kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    #model.add(MaxPooling2D(pool_size=5,strides=5))

    model.add(GlobalMaxPooling1D())

    #model.add(BatchNormalization())

    #model.add(BatchNormalization())

    #model.add(Dropout(0.5))

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    #filepath="smallCNN_dropout_"+str(DROP_OUT)+".hdf5"

    filepath="SmallCNN"+"best_smallCNN_dropout_"+str(params["num_kernel"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS) 

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = 0.2,callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)



    #finalmodel = load_model(filepath)

    return out, model #,roc_train,roc_test,acc_train,acc_test





def mediumCNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(32,kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(MaxPooling1D())

    

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(MaxPooling1D())



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(GlobalMaxPooling1D())

    

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    #filepath="smallCNN_dropout_"+str(DROP_OUT)+".hdf5"

    filepath="MediumCNN"+"best_mediumCNN_dropout_"+str(params["num_kernel"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = 0.2,callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)



    #finalmodel = load_model(filepath)

    return out, model #,roc_train,roc_test,acc_train,acc_test





def largeCNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(64,kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(MaxPooling1D())



    model.add(Conv1D(64,kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(MaxPooling1D())



    model.add(Conv1D(64,kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(MaxPooling1D())




    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    #filepath="smallCNN_dropout_"+str(DROP_OUT)+".hdf5"

    filepath="LargeCNN"+"best_largeCNN_dropout_"+str(params["num_kernel"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = 0.2,callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)



    #finalmodel = load_model(filepath)

    return out, model #,roc_train,roc_test,acc_train,acc_test



def verylargeCNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(MaxPooling1D())



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(MaxPooling1D())



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(MaxPooling1D())



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

   

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(GlobalMaxPooling1D())



    



    model.add(Dense(1))

    model.add(Activation("sigmoid"))

    filepath="VerylargeCNN"+"best_veryLargeCNN_dropout_"+str(params["kernel_size"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    print(model.summary())

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = 0.2,callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)



    #finalmodel = load_model(filepath)

    return out, model #,roc_train,roc_test,acc_train,acc_test

def miniCNN_RNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(MaxPooling1D(pool_size=10,strides=10))



    #model.add(GlobalMaxPooling1D())

    #model.add(Flatten())

    #model.add(Reshape((50,1))) # shape becomes (batch_size,200,1)

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3)))

    #model.add(GlobalMaxPooling1D())

    model.add(Flatten())

    #model.add(Dense(25))

    #model.add(Activation("relu"))

    #model.add(GlobalMaxPooling1D())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)

    #filepath="largeRNN_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    filepath="best_smallCNN_RNN_dropout_"+str(params["hidden_unit"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])) , metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = VALIDATION_SPLIT,callbacks=[checkpoint,early_stopping_monitor]) #,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    return out, model


def smallCNN_RNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(MaxPooling1D(pool_size=10,strides=10))



    #model.add(GlobalMaxPooling1D())

    #model.add(Flatten())

    #model.add(Reshape((50,1))) # shape becomes (batch_size,200,1)

    HIDDEN_UNITS = 5


    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3)))

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3)))


    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True)))

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True)))



    #model.add(GlobalMaxPooling1D())

    model.add(Flatten())

    #model.add(Dense(25))

    #model.add(Activation("relu"))

    #model.add(GlobalMaxPooling1D())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)

    #filepath="largeRNN_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    filepath="best_smallCNN_RNN_dropout_"+str(params["hidden_unit"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])) , metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = VALIDATION_SPLIT,callbacks=[checkpoint,early_stopping_monitor]) #,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    return out, model




def mediumCNN_RNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(MaxPooling1D(pool_size=10,strides=10))

    #model.add(Flatten())

    #model.add(Reshape((50,1))) # shape becomes (batch_size,200,1)

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3)))

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)

    #filepath="largeRNN_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    filepath="best_mediumCNN_RNN_dropout_"+str(params["hidden_unit"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_split = VALIDATION_SPLIT,callbacks=[checkpoint,early_stopping_monitor]) #,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)


    return out, model



def largeCNN_RNN(x_train,y_train, x_test, y_test,params):

    model = Sequential()

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = x_train.shape[1:3]))

    model.add(Activation("relu"))

    model.add(MaxPooling1D(pool_size=10,strides=5))

    model.add(Conv1D(params["num_kernel"],kernel_size=params["kernel_size"],kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2)))

    model.add(Activation("relu"))

    model.add(MaxPooling1D(pool_size=10,strides=5))

    #model.add(Flatten())

    #model.add(Reshape((50,1))) # shape becomes (batch_size,200,1)

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.3)))

    model.add(Bidirectional(LSTM(params["hidden_unit"],activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    model.compile(loss=LOSS, optimizer = params["optimizer"](lr_normalizer(params["lr"], params['optimizer'])), metrics =METRICS)

    #filepath="largeRNN_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    filepath="largeRNN"+"best_largeCNN_RNN_dropout_"+str(params["hidden_unit"])+".hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    #model.compile(loss=LOSS, optimizer = 'Adam', metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    out = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_data = (x_test,y_test),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print(model.summary())

    return out, model



#input layer => conv layer (relu) => pool(max) = > output (softmax)

#DROP_OUT = 1 or 0, 1 meaning include dropout, 0 meaning don't include dropout

def ConvNet2DModel1(x_train,y_train,x_test,y_test,DROP_OUT,INPUT_SHAPE):

    #define the ConvNet

    model = Sequential()

    #conv layer # 20 kernels of size 3x3

    model.add(Conv1D(20,kernel_size=10,kernel_initializer=KERNEL_INITIAL,kernel_constraint = maxnorm(2),input_shape = INPUT_SHAPE))

    #conv layer activation

    model.add(Activation("relu"))

    #conv layer (relu) => pool(max)

    model.add(GlobalAveragePooling1D())

    if(DROP_OUT == 1):

        model.add(Dropout(0.3))

    # output layer

    model.add(BatchNormalization())

    #model.add(Flatten()) #Flattens

    model.add(Dense(50))

    model.add(Activation('relu'))

    model.add(Dropout(0.3))



    model.add(Reshape((50,1))) # shape becomes (batch_size,200,1)

    #input_shape to RNN layer is (batch_size,timeSteps,num_features), only provide (timeSteps,num_features)

    model.add(BatchNormalization())

    model.add(LSTM(10,activation='tanh',inner_activation='sigmoid',kernel_constraint = maxnorm(2),kernel_initializer=KERNEL_INITIAL,return_sequences=False,dropout=0.3))



    model.add(Dense(1))

    #a soft max classifier

    model.add(Activation("sigmoid"))

    filepath="model1_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

    #callbacks_list = [checkpoint]

    model.compile(loss=LOSS, optimizer = 'Adam', metrics =METRICS)

    print(model.summary())

    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                            validation_data = (x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    ## if you want early stopping

    #Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,

                                #validation_split = VALIDATION_SPLIT,callbacks=[early_stopping_monitor,checkpoint])



    return model,Tuning





from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



def trainRF(x_train,y_train,x_test,y_test):

    clf = RandomForest()

    param_grid = {

                 'n_estimators': [100,200]

             }

    grid_clf = GridSearchCV(clf, param_grid, cv=5,scoring='accuracy')

    grid_clf.fit(X_train, y_train)

    return grid_clf



#import matplotlib

#import matplotlib.pyplot as plt





# def plot_history(Tuning,Title):

#      fig, axs = plt.subplots(1,2,figsize=(15,5))

#      # summarize loss history

#      axs[0].plot(Tuning.history['loss'])

#      axs[0].plot(Tuning.history['val_loss'])

#      axs[0].set_ylabel('loss')

#      axs[0].set_xlabel('epoch')

#      axs[0].legend(['train', 'vali'], loc='best')

#      axs[0].set_title(Title)

#

#      # summarize history for loss

#      axs[1].plot(Tuning.history['acc'])

#      axs[1].plot(Tuning.history['val_acc'])

#      axs[1].set_ylabel('accuracy')

#      axs[1].set_xlabel('epoch')

#      axs[1].legend(['train', 'vali'], loc='best')

#      axs[1].set_title(Title)

#

#      plt.show(block = False)

#      plt.show()



def SaveHistory(Tuning,outfile):

    #keys = Tunning.history.keys()

    Hist = np.empty(shape=(len(Tuning.history['val_loss']),4))

    Hist[:,0] = Tuning.history['val_loss']

    Hist[:,1] = Tuning.history['val_acc']

    Hist[:,2] = Tuning.history['loss']

    Hist[:,3] = Tuning.history['acc']

    np.savetxt(outfile, Hist, fmt='%.8f',delimiter=",",header="val_loss,val_acc,train_loss,train_acc",comments="")

    return Hist



def SaveResult(roc_train,roc_test,acc_train,acc_test,outfile):

    #keys = Tunning.history.keys()

    f=open(outfile,"w")

    f.write("roc_train,roc_test,acc_train,acc_test"+"\n")

    f.write(str(roc_train) + "," + str(roc_test) + "," + str(acc_train) + "," + str(acc_test))

    f.close()

    #Hist = np.empty(shape=(1,4))

    #Hist[0,:] = [roc_train,roc_test,acc_train,acc_test]

   # Hist = np.array([roc_train,roc_test,acc_train,acc_test])

    #np.savetxt(outfile, np.transpose(Hist), fmt='%.8f',delimiter=",",header="roc_train,roc_test,acc_train,acc_test",comments="")



# def ExtractMotif(x_data,y_data,model,motif_len,stride):

#     #shuffle

#     i = 0

#     while i < x_data.shape[1] - motif_len - 1:

#         tmp = x_data

#         for j in range(0,x_data.shape[0]):

#             np.random.shuffle(tmp[j,i:i+motif_len+1,:])

#         #perform prediction

#         #accuracy error for 1 i

#     #accuracy error for all i's

#     #get the worst error

#     i = i + stride