# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:34:01 2016

@author: Pravinth Samuel Vethanayagam
"""

import load_data
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten

class cnn_model():
    
    cols = None
    model = None
    numEpoch = None
    numBatch = None
    
    def __init__(self, numEpoch = 20, numBatch = 200):
        self.numEpoch = numEpoch
        self.numBatch = numBatch
        
        print 'Initializing model...'
        self.model = Sequential()
        self.model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 96,96), init = 'glorot_uniform'))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Convolution2D(32,3,3, init = 'glorot_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Convolution2D(32,3,3, init = 'glorot_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(500, init = 'glorot_uniform'))
        self.model.add(Activation("relu"))
        self.model.add(Dense(200, init = 'glorot_uniform'))
        self.model.add(Activation("relu"))
        self.model.add(Dense(30, init = 'glorot_uniform'))
        self.model.add(Activation("relu"))
        
        self.model.compile(loss="mean_absolute_error", optimizer="sgd")
        
    def fit(self, tr_df):
        print 'Fitting the model...'
        X = np.vstack(tr_df.Image.values)
        X = X.reshape(-1, 1, 96, 96) / 255.0
        y = tr_df.drop('Image', 1).as_matrix() / 255.0
        self.cols = tr_df.columns
        self.model.fit(X, y, nb_epoch=self.numEpoch, batch_size=self.numBatch)
        
    def predict(self, te_df):
        print 'Predicting values...'
        X = np.vstack(te_df.Image.values)
        X = X.reshape(-1, 1, 96, 96) / 255.0
        
        y = self.model.predict(X)
        y = 255.0*pd.DataFrame(y)
        y.columns = self.cols.drop('Image')
        
        if 'ImageId' in te_df.columns:
            y['ImageId'] = te_df['ImageId']
        
        return y
        
    
if __name__ == '__main__':
    train_df, test_df = load_data.load_fkdata()
    train_df = train_df.dropna()
    idLookup = load_data.loadIdLookup()
    
    c_model1 = cnn_model(1, 1000)
    c_model1.fit(train_df)
    
    train_df_y = c_model1.predict(train_df)
    test_df_y = c_model1.predict(test_df)
    
    load_data.createSubmissionFile(test_df_y, idLookup, 'results/cnn_model_1_results.csv')
    
    
    