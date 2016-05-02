# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:56:43 2016

@author: Pravinth Samuel Vethanayagam
"""
import load_data
import pandas as pd

class mean_model():
    
    listOfMeans = None    
    cols = None
    
    def __init__(self):
        print 'Initializing model...'
    
    def fit(self, tr_df):
        print 'Fitting the model...'
        self.listOfMeans = tr_df.mean().drop('Image')
        self.cols = self.listOfMeans.index
        self.listOfMeans = self.listOfMeans.get_values().reshape(1,-1)
        
        
    
    def predict(self, te_df):
        print 'Predicting values...'
        y = pd.DataFrame(self.listOfMeans.repeat(len(te_df), 0))
        y.columns = self.cols
        
        if 'ImageId' in te_df.columns:
            y['ImageId'] = te_df['ImageId']
        
        return y
    

if __name__ == '__main__':
    train_df, test_df = load_data.load_fkdata()
    #train_df = train_df.dropna()
    idLookup = load_data.loadIdLookup()
    
    m_model = mean_model()
    m_model.fit(train_df)
    
    train_df_y = m_model.predict(train_df)
    test_df_y = m_model.predict(test_df)
    
    load_data.createSubmissionFile(test_df_y, idLookup, 'results/benchmark_mean_results.csv')    