# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 07:33:54 2016

@author: Pravinth Samuel Vethanayagam
"""
import os.path
import zipfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def unzip_datasets(folderpath = './data/'):
    # unzip training.zip and test.zip
    if not os.path.isfile(folderpath + 'training.csv'):
        print 'Extracting training.zip...'
        with zipfile.ZipFile(folderpath + 'training.zip', "r") as z:
            z.extractall(folderpath)
    
    if not os.path.isfile(folderpath + 'test.csv'):
        print 'Extracting test.zip...'
        with zipfile.ZipFile(folderpath + 'test.zip', "r") as z:
            z.extractall(folderpath)

def loadIdLookup(folderpath = './data/'):
    idLookup = pd.read_csv(folderpath + 'IdLookupTable.csv')
    return idLookup

def load_fkdata(folderpath = './data/'):
    unzip_datasets(folderpath)
    
    train_df = pd.read_csv(folderpath + 'training.csv')
    test_df = pd.read_csv(folderpath + 'test.csv')
    
    train_df.Image = train_df.Image.apply(lambda im: np.reshape(np.fromstring(im, sep = ' '), (96,96)))    
    test_df.Image = test_df.Image.apply(lambda im: np.reshape(np.fromstring(im, sep = ' '), (96,96)))
    # reshape
    return train_df, test_df
        
def plotGridOfFaces(M, N, images, startIx):
    plt.figure(1)
    
    for m in range(M):
        for n in range(N):
            plt.subplot(M,N,1+m+M*n)
            plt.imshow(train_df.Image[startIx+ m + n*N], cmap = 'gray')
            plt.axis('off')
    
    plt.show()

def markPointsOnImage(image, points):
    plt.imshow(image, cmap = 'gray')
    plt.scatter([float(p[0]) for p in points], [float(p[1]) for p in points])
    plt.show()

def pointsFromLabel(row, labels):
    points = []
    if 'all' in labels:
        labels.append('eyes')
        labels.append('eyebrows')
        labels.append('mouth')
        labels.append('nose')
    if 'eyes' in labels:
        labels.append('left_eye')
        labels.append('right_eye')
    if 'eyebrows' in labels:
        labels.append('left_eyebrow')
        labels.append('right_eyebrow')
    if 'left_eye' in labels:
        points.append((row['left_eye_center_x'], row['left_eye_center_y']))
        points.append((row['left_eye_inner_corner_x'], row['left_eye_inner_corner_y']))
        points.append((row['left_eye_outer_corner_x'], row['left_eye_outer_corner_y']))
    if 'left_eyebrow' in labels:
        points.append((row['left_eyebrow_inner_end_x'], row['left_eyebrow_inner_end_y']))
        points.append((row['left_eyebrow_outer_end_x'], row['left_eyebrow_outer_end_y']))
    if 'right_eye' in labels:
        points.append((row['right_eye_center_x'], row['right_eye_center_y']))
        points.append((row['right_eye_inner_corner_x'], row['right_eye_inner_corner_y']))
        points.append((row['right_eye_outer_corner_x'], row['right_eye_outer_corner_y']))
    if 'right_eyebrow' in labels:
        points.append((row['right_eyebrow_inner_end_x'], row['right_eyebrow_inner_end_y']))
        points.append((row['right_eyebrow_outer_end_x'], row['right_eyebrow_outer_end_y']))
    if 'mouth' in labels:
        points.append((row['mouth_left_corner_x'], row['mouth_left_corner_y']))
        points.append((row['mouth_right_corner_x'], row['mouth_right_corner_y']))
        points.append((row['mouth_center_top_lip_x'], row['mouth_center_top_lip_y']))
        points.append((row['mouth_center_bottom_lip_x'], row['mouth_center_bottom_lip_y']))
    if 'nose' in labels:
        points.append((row['nose_tip_x'], row['nose_tip_y']))
    
    return points
    
def markLabelsOnRow(row, labels):
    markPointsOnImage(row.Image.iloc[0], pointsFromLabel(row, labels))
        
def createSubmissionFile (y, idLookup, filename = 'results/model_results.csv'):
    results = []    
    for idl in idLookup.as_matrix():
        val = y[y['ImageId']==1][idl[2]].iloc[0]
        
        if val < 0:
            val = 0
        if val > 96:
            val = 96
            
        results.append((idl[0], val))
    
    print 'Generating the results file...'
    
    results_df = pd.DataFrame(results)
    results_df.columns = ['RowId', 'Location']
    results_df.to_csv(filename, index=False)
    return results_df
    
if __name__ == '__main__':
    train_df, test_df = load_fkdata()
    markLabelsOnRow(train_df[6:7], ['eyes', 'mouth', 'eyebrows'])
    plotGridOfFaces(3, 4, train_df.Image, 10)