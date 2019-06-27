#!/bin/env python3
#-*- coding=utf-8 -*-
# add stopwords

import os
import numpy as np
import tensorflow as tf
import _pickle as pk
import jieba
from nltk.corpus import stopwords

#define stop words by myself
#stop = ["the","be","am","are","is","at","on","to","in","from","as",
#        "and","so","or","of","a","an","for","with"]

def segment_word(line, jiebacut):
    ws_insen = jieba.cut(line, cut_all=jiebacut)
    return ws_insen

class DataManager(object):
    def __init__(self):
        self.data = {}
        # Read data from data_path
        #  name       : string, name of data, usually 'train' 'test' 'semi'
        #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, jiebacut=False,with_label=False):
        print ('read data from %s...'%data_path)
        X,Y = [],[]
        
        jieba.initialize()
        jieba.set_dictionary('big5dict/dict.txt.big')

        stop = stopwords.words('chinese')
        stop.append("\n") 
        stop.append("\t") 

        filters = '"#%&()*+,-./:;<=>@[\\]^_`{|}~\t\nabcdefghijklmnopqrstuvwxyz'
        
        for filter in filters:
            stop.append(filter)
        
        #input is no-label data
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split('|')
                    ws_insen = jieba.cut(lines[1], cut_all=jiebacut)
                    X.append([ws for ws in ws_insen if ws not in stop])
                    Y.append(int(lines[0]))
                else:
                    if name=="pylady_data":
                        ws_insen = line.split(" ")
                    else:
                        ws_insen = jieba.cut(line, cut_all=jiebacut)
                    X.append([ws for ws in ws_insen if ws not in stop])

        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]
                    
    # Build dictionary
    #  vocab_size : maximum number of word in your dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        #filters='"#%&()*+,-./:;<=>@[\\]^_`{|}~?\$!\t\n'
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='"#%&()*+,-./:;<=>@[\\]^_`{|}~?\$!\t\n')

        for key in self.data:
            print ('tokenizing %s'%key)
            # already seperate them into lists, do not have to seperate again
            texts = self.data[key][0]
            # remove stopwords here
            #texts = [w for text in texts
            #         if text not in stop]
            #texts = [w for text in texts
            #         for w in text.split()
            #         if w not in stop]

            #Updates internal vocabulary based on texts
            self.tokenizer.fit_on_texts(texts)
        
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))

    # Convert words in sentences to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            #Convert a list of texts to a Numpy matrix.
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            #Pads sequences to the same length.
            self.data[key][0] = np.array(tf.keras.preprocessing.sequence.pad_sequences(tmp, maxlen=maxlen))

    # Convert words in setences to index without padding
    def to_sequence_nopad(self):
        for key in self.data:
            print ('Converting %s to sequences'%key)
            #Convert a list of texts to a Numpy matrix.
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = tmp
            
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            # Convert a list of texts to a Numpy matrix.
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                #Converts a class vector (integers) to binary class matrix.
                self.data[key][1] = np.array(tf.keras.utils.to_categorical(self.data[key][1]))

    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

        # get data by name
    def get_data(self,name):
        return self.data[name][0]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
