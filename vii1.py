#vii1
import sys
import numpy as np
import pickle
import re
import time
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
vii=('V.I.I-1')
#modul
def text_cleaner(text):
#    stemmer = Stemmer('russian')
    text = ' '.join( stemmer.stemWords( text.split() ) )
    text = re.sub( r'\b\d+\b', ' digit ', text ) # замена цифр
    return text
def start_data():
    data = { 'text':[],'tag':[] }
    for line in open('start.txt'):
        if(not('#' in line)):
            row = line.split("*")
            data['text'] += [row[0]]
            data['tag'] += [row[1]]
    return data
def train_test_split( data, validation_split = 0.1):
    sz = len(data['text'])
    indices = np.arange(sz)
    np.random.shuffle(indices)
    X = [ data['text'][i] for i in indices ]
    Y = [ data['tag'][i] for i in indices ]
    nb_validation_samples = int( validation_split * sz )
    return {
    'train': { 'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples] },
    'test': { 'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]}
    }
def openai(msg):
    data=start_data()
    D = train_test_split( data )
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', SGDClassifier(loss='hinge')),])
    text_clf.fit(D['train']['x'], D['train']['y'])
    predicted = text_clf.predict( D['train']['x'] )
    zz=[]
    zz.append(msg)
    predicted = text_clf.predict(zz)
    if predicted[0]=='no_id':
        os.system(msg)
    elif 'github.com' in msg:
        print(predicted[0],msg)
        os.system('echo off')
        os.system('git clone '+msg)
    else:
        print(predicted[0])
#отладка
while True:
    openai(input('vii1: '))
