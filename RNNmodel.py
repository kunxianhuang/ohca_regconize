#!/bin/env python3
#-*- coding=utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

# class that builds model 
class simpleRNN(object):

    def __init__(self,args):
        # Embedding layer
        # Define Embedding
        with tf.name_scope("embeddings"):
            if(args.pretrain_emb==True):
                self.embedding_mat = tf.Variable(tf.constant(0.1,shape=[args.vocab_size, args.embedding_dim]),
                                                 trainable=True, name='embedding_mat')
            else:
                self.embedding_mat = tf.get_variable('embedding_mat', [args.vocab_size, args.embedding_dim],
                                                 tf.float32, tf.random_normal_initializer())
            self.RNNcells=None
        

    def model(self, args, x_data, keep_prob):

        embedding_output = tf.nn.embedding_lookup(self.embedding_mat, x_data)
        #build RNN from here
        return_sequence = False
        #dropout_rate = args.dropout_rate

        with tf.name_scope("RNNlayers"):
            if args.cell == 'GRU':
                grucell = tf.contrib.rnn.GRUCell(args.hidden_size)
                #drop = tf.contrib.rnn.DropoutWrapper(grucell, output_keep_prob=args.keep_prob)
                drop = tf.contrib.rnn.DropoutWrapper(grucell, output_keep_prob=keep_prob)
                self.RNNcells = tf.contrib.rnn.MultiRNNCell([drop for _ in range(args.num_layers)])
            
                #RNN_cell = GRU(args.hidden_size, 
                #               return_sequences=return_sequence, 
                #               dropout=dropout_rate)
            elif args.cell == 'LSTM':
                def get_lstmcell(args):
                    #usage of peephole
                    #lstmcell = tf.contrib.rnn.LSTMCell(args.hidden_size, use_peepholes=True,reuse=tf.get_variable_scope().reuse)
                    lstmcell = tf.contrib.rnn.LSTMCell(args.hidden_size, reuse=tf.get_variable_scope().reuse)
                    drop = tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=args.keep_prob)
                    return drop
                self.RNNcells = tf.contrib.rnn.MultiRNNCell([get_lstmcell(args) for _ in range(args.num_layers)])
                #RNN_cell = LSTM(args.hidden_size, 
                #                return_sequences=return_sequence, 
                #dropout=dropout_rate)
                
            #initial state for RNN cell 
            self.initial_state = self.RNNcells.zero_state(args.batch_size, tf.float32)

        with tf.name_scope("RNN_forward"):
            outputs, final_state = tf.nn.dynamic_rnn(self.RNNcells, embedding_output, initial_state=self.initial_state)
            self.outputs = outputs
        #prediction use last output of each sample sentence in batch
        with tf.name_scope('predictions'):
            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
            tf.summary.histogram('predictions', predictions)

        return predictions


    def loss(self, labels,predictions):
        #loss function: MSE 
        cost = tf.losses.mean_squared_error(tf.squeeze(labels), tf.squeeze(predictions))
        #record for tensorboard
        tf.summary.scalar('cost', cost)
        return cost

    def optimizer(self, args, loss):
        #optim = tf.train.AdamOptimizer(args.learning_rate)
        optim = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate, rho=0.95, epsilon=1e-08)
        training_op = optim.minimize(loss)
        return training_op
    

    def accuracy(self, labels, predictions):
        """
        calculate accuracy used in the validation
        """
        correct_pred = tf.equal(tf.cast(tf.round(tf.squeeze(predictions)), tf.int32), tf.squeeze(labels))
        accu = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accu)
        return accu
    
    
