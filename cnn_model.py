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

from keras.models import model_from_json

class cnn_model():
    
    cols = None
    model = None
    numEpoch = None
    numBatch = None
    name = None

    def kerasModel(self):   
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 96,96), init = 'glorot_uniform'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(64,3,3, init = 'glorot_uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(128,3,3, init = 'glorot_uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(500, init = 'glorot_uniform'))
        model.add(Activation("relu"))
        model.add(Dense(500, init = 'glorot_uniform'))
        model.add(Activation("relu"))
        model.add(Dense(30, init = 'glorot_uniform'))
        model.add(Activation("linear"))
        
        model.compile(loss="mean_absolute_error", optimizer="sgd")
        
        return model
    
    def __init__(self, name, numEpoch = 20, numBatch = 200):
        self.numEpoch = numEpoch
        self.numBatch = numBatch
        self.name = name
        
        print 'Initializing model...'
        self.model = self.kerasModel()
        
    def fit(self, tr_df):
        print 'Fitting the model...'
        X = np.vstack(tr_df.Image.values)
        X = X.reshape(-1, 1, 96, 96) / 255.0
        y = tr_df.drop('Image', 1).as_matrix() / 255.0
        self.cols = tr_df.columns
        self.model.fit(X, y, nb_epoch=self.numEpoch, batch_size=self.numBatch)
        self.save()
        
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
        
    def save(self, name=None):
        if name is None:
            name = self.name
        open('./models/' + name + '.json', 'w').write(self.model.to_json())
        self.model.save_weights('./models/' + name + '_weights.h5', overwrite=True)
    
    def load(self, name):
        self.model = model_from_json(open('./models/' + name + '.json').read())
        self.model.load_weights('./models/' + name + '_weights.h5')

class cnn_model_simple(cnn_model):
    
    def kerasModel(self):
        model = Sequential()
        model.add(Convolution2D(32, 11, 11, border_mode='valid', input_shape=(1, 96,96), init = 'glorot_uniform'))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(500, init = 'glorot_uniform'))
        model.add(Activation("relu"))
        model.add(Dense(30, init = 'glorot_uniform'))
        model.add(Activation("linear"))
        
        model.compile(loss="mean_absolute_error", optimizer="sgd")
        return model
        
    def __init__(self, numEpoch = 20, numBatch = 200):
        self.numEpoch = numEpoch
        self.numBatch = numBatch
        
        print 'Initializing model...'
        self.model = self.kerasModel()
    
if __name__ == '__main__':
    train_df, test_df = load_data.load_fkdata()
    train_df = train_df.dropna()
    idLookup = load_data.loadIdLookup()
    
    c_model1 = cnn_model('dnouri_model', 50, 100)
    c_model1.fit(train_df)
    
    train_df_y = c_model1.predict(train_df)
    test_df_y = c_model1.predict(test_df)
    
    load_data.createSubmissionFile(test_df_y, idLookup, 'results/cnn_dnouri_model_150epochs_results.csv')
    
    
    