# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:47:47 2019

@author: Mango
"""

import os

import sys

from numpy import array

from numpy import argmax

from keras.utils import to_categorical

import numpy as np

import string



from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import numpy





import h5py

#filename = sys.argv[1]

f = h5py.File("RNA_OnehotEncoded.h5", 'r')

x_train = np.array(f['x_train'])

#x_test = np.array(f['x_test'])

y_train = np.array(f['y_train'])
#y_test = np.array(f['y_test'])

print(x_train.shape)

#print(x_test.shape)

print(y_train.shape)



import Models
import talos as ta
from talos import Reporting, Evaluate, Deploy

#name = sys.argv[2]

p = {    'lr': [1, 0.1, 10],
         'kernel_size': [6, 8, 10, 12, 14],
		 'hidden_unit': [5,10,15,20,30,40],
         'num_kernel': [32, 64, 96, 120],
         'optimizer': [Adam, RMSprop, Adagrad, Adadelta]}
'''
scan_smallRNN = ta.Scan(x_train, y_train, model=Models.smallRNN, params=p, grid_downsample=0.1)
print('################################# 2-layer CNN ######################################')
print(scan_smallRNN)
scan_smallRNN.x = np.zeros(500)
scan_smallRNN.y = np.zeros(500)
Deploy(scan_smallRNN, 'SmallCNN_experiment', metric="val_loss", asc=True)
'''
scan_mediumRNN = ta.Scan(x_train, y_train, model=Models.mediumRNN, params=p, grid_downsample=0.1)
print('################################# 2-layer CNN ######################################')
print(scan_mediumRNN)
scan_mediumRNN.x = np.zeros(500)
scan_mediumRNN.y = np.zeros(500)
Deploy(scan_mediumRNN, 'mediumRNN_experiment', metric="val_loss", asc=True)
'''
scan_largeRNN = ta.Scan(x_train, y_train, model=Models.largeRNN, params=p, grid_downsample=0.1)
print('##################################### 3-layer CNN #############################################')
print(scan_largeRNN)
scan_largeRNN.x = np.zeros(500)
scan_largeRNN.y = np.zeros(500)
Deploy(scan_largeRNN, 'largeRNN_experiment', metric="val_loss", asc=True)

scan_smallCNN_RNN = ta.Scan(x_train, y_train, model=Models.smallCNN_RNN, params=p, grid_downsample=0.1)
print('##################################### 4-layer CNN #############################################')
print(scan_smallCNN_RNN)
scan_smallCNN_RNN.x = np.zeros(500)
scan_smallCNN_RNN.y = np.zeros(500)
Deploy(scan_smallCNN_RNN, 'smallCNN_RNN_experiment', metric="val_loss", asc=True)

scan_mediumCNN_RNN = ta.Scan(x_train, y_train, model=Models.mediumCNN_RNN, params=p, grid_downsample=0.1)
print('##################################### 8-layer CNN #############################################')
print(scan_mediumCNN_RNN)
scan_mediumCNN_RNN.x = np.zeros(500)
scan_mediumCNN_RNN.y = np.zeros(500)
Deploy(scan_mediumCNN_RNN, 'mediumCNN_RNN_experiment', metric="val_loss", asc=True)

scan_largeCNN_RNN = ta.Scan(x_train, y_train, model=Models.largeCNN_RNN, params=p, grid_downsample=0.1)
print('##################################### 8-layer CNN #############################################')
print(scan_largeCNN_RNN)
scan_largeCNN_RNN.x = np.zeros(500)
scan_largeCNN_RNN.y = np.zeros(500)
Deploy(scan_largeCNN_RNN, 'largeCNN_RNN_experiment', metric="val_loss", asc=True)
'''
'''	 
scan_miniCNN = ta.Scan(x_train, y_train, model=Models.miniCNN, params=p, grid_downsample=0.1)
scan_smallCNN = ta.Scan(x_train, y_train, model=Models.smallCNN, params=p, grid_downsample=0.1)
scan_mediumCNN = ta.Scan(x_train, y_train, model=Models.mediumCNN, params=p, grid_downsample=0.1)
scan_largeCNN = ta.Scan(x_train, y_train, model=Models.largeCNN, params=p, grid_downsample=0.1)
scan_verylargeCNN = ta.Scan(x_train, y_train, model=Models.verylargeCNN, params=p, grid_downsample=0.1)

print('################################# 2-layer CNN ######################################')
print(scan_miniCNN)
scan_miniCNN.x = np.zeros(500)
scan_miniCNN.y = np.zeros(500)
Deploy(scan_miniCNN, 'MiniCNN_experiment', metric="val_loss", asc=True)
print('################################# 2-layer CNN ######################################')
print(scan_smallCNN)
scan_smallCNN.x = np.zeros(500)
scan_smallCNN.y = np.zeros(500)
Deploy(scan_smallCNN, 'SmallCNN_experiment', metric="val_loss", asc=True)
print('##################################### 3-layer CNN #############################################')
print(scan_mediumCNN)
scan_mediumCNN.x = np.zeros(500)
scan_mediumCNN.y = np.zeros(500)
Deploy(scan_smallCNN, 'MediumCNN_experiment', metric="val_loss", asc=True)
print('##################################### 4-layer CNN #############################################')
print(scan_largeCNN)
scan_largeCNN.x = np.zeros(500)
scan_largeCNN.y = np.zeros(500)
Deploy(scan_largeCNN, 'LargeCNN_experiment', metric="val_loss", asc=True)

print('##################################### 8-layer CNN #############################################')
print(scan_verylargeCNN)
scan_verylargeCNN.x = np.zeros(500)
scan_verylargeCNN.y = np.zeros(500)
Deploy(scan_verylargeCNN, 'verylargeCNN_experiment', metric="val_loss", asc=True)
'''