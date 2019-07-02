# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:31:10 2019

@author: Mango
"""
import numpy as np
import pandas as pd
import h5py

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

def OutputResult(x,y, bestmodel_json_file, bestmodel_weights_h5_file):
    from keras.models import model_from_json
    file = open(bestmodel_json_file,'r')
    lines = file.read()
    model = model_from_json(lines)
    model.load_weights(bestmodel_weights_h5_file)
    print(model.summary())
    pred = model.predict_classes(x)
    pred_p=model.predict(x)
    fpr, tpr, thresholdTest = roc_curve(y, pred_p)
    aucv = auc(fpr, tpr)
    precision,recall,fscore,support=precision_recall_fscore_support(y,pred,average='macro')
    print('auc',aucv)
    print('acc',accuracy_score(y,pred))
    print('mcc',matthews_corrcoef(y,pred))
    print('precision',precision)
    print('recall',recall)
    print('fscore', fscore)
    print('support:',support)
    #output = {'auc': [aucv], 'acc': [accuracy_score(y,pred)], 'mcc': [matthews_corrcoef(y,pred)], 'precision': [precision], 'recall': [recall], 'fscore': [fscore]}
    #outputdf = pd.DataFrame(output, columns = ['Metrics', 'Value'])
    #print(outputdf)
    #return outputdf

f = h5py.File("RNA_OnehotEncoded.h5", 'r')
 
x_test = np.array(f['x_test'])
y_test = np.array(f['y_test'])

#print("Medium RNN")
#OutputResult(x_test, y_test, 'mediumRNN_experiment/mediumRNN_experiment_model.json', 'mediumRNN_experiment/mediumRNN_experiment_model.h5')
#print("Large RNN")
#OutputResult(x_test, y_test, 'largeRNN_experiment/largeRNN_experiment_model.json', 'largeRNN_experiment/largeRNN_experiment_model.h5')
print("MiniCNN_RNN")
OutputResult(x_test, y_test, 'miniCNN_RNN_experiment/miniCNN_RNN_experiment_model.json', 'miniCNN_RNN_experiment/miniCNN_RNN_experiment_model.h5')
#print("LargeCNN_RNN")
#OutputResult(x_test, y_test, 'largeCNN_RNN_experiment/largeCNN_RNN_experiment_model.json', 'largeCNN_RNN_experiment/largeCNN_RNN_experiment_model.h5')

#OutputResult(x_test, y_test, 'verylargeCNN_experiment/verylargeCNN_experiment_model.json', 'verylargeCNN_experiment/verylargeCNN_experiment_model.h5')





    