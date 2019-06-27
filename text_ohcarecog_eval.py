#!/bin/env python3
#-*- coding=utf-8 -*-

import os, sys
import readline
import argparse
import time
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import progressbar as pb
import _pickle as pk
from utils.util import DataManager
from utils.batch_index import batch_index, get_batches, get_batches_nolabel
from RNNmodel import simpleRNN

def main():
    parser = argparse.ArgumentParser(description='Text OHCA recognition')

    parser.add_argument('--batch_size', default=1, type=float) # change to 1 for single setence
    parser.add_argument('--vocab_size', default=50000, type=int)
    # model parameter
    parser.add_argument('--loss_function', default='binary_crossentropy')
    parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
    parser.add_argument('-num_lay', '--num_layers', default=2, type=int)
    parser.add_argument('-emb_dim', '--embedding_dim', default=256, type=int)
    parser.add_argument('-hid_siz', '--hidden_size', default=400, type=int)
    parser.add_argument('--pretrain_emb', default=True, type=bool)
    parser.add_argument('--emb_matrix', default='cbowemb.npz')


    parser.add_argument('--keep_prob', default=1.0, type=float)
    parser.add_argument('--max_length', default=400,type=int)
    parser.add_argument('--threshold', default=0.6,type=float)
    # output path for your prediction
    parser.add_argument('--result_path', default='evalresult.txt',)

    # input testing file name
    parser.add_argument('--test_file', default="data/ohca_test1.txt")

    # output testing result
    parser.add_argument('--outfile', default="data/ohca_testout.txt")
    # put model in the same directory
    parser.add_argument('--load_model', default = True)
    parser.add_argument('--load_token', default = True, type=bool)
    parser.add_argument('--save_dir', default = 'model/')
    # log dir for tensorboard
    parser.add_argument('--log_dir', default='log_dir/')
    args = parser.parse_args()

    test_path = args.test_file
    
    save_path = 'token/'
    #load token path
    if args.load_token is not None:
        load_path = os.path.join(save_path)
            
    sess = tf.Session() 
    
    #####read data#####
    dm = DataManager()
    print ('Loading test data...')
    dm.add_data('test_data', test_path, with_label=False)

            
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_token is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        raise Exception("Word token is not loaded...")


    # convert to sequences
    dm.to_sequence(args.max_length)

    # Create the graph object
    tf.reset_default_graph() 
    # initial model
    print ('initial model...')
    rnnmodel = simpleRNN(args)    

    
    with tf.name_scope('inputs'):
        #create placeholder for training (testing) data 
        X_ = tf.placeholder(tf.int32, [None, args.max_length], name='X')
        #y_ = tf.placeholder(tf.int32, [args.batch_size, ], name='y_')
        keep_prob = tf.placeholder_with_default(1.0, shape=(),name="keep_prob")
        
    y_predict = rnnmodel.model(args,X_, keep_prob)


    #initial state of LSTM
    init_state = rnnmodel.initial_state
    
    #check outputs of LSTM
    routputs = rnnmodel.outputs
    
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir)
    
        path = os.path.join(load_path,'Sentimen_rnn_final')
        if os.path.exists(path+".meta"):
            print ('load model from %s' % path)
            #model.load_weights(path) change to tensorflow model
        else:
            raise ValueError("Can't find the file %s" %path)
   
    #elif args.action == 'test' :
    X = dm.get_data('test_data')
    print("Load test data (shape {})".format(X.shape))
        #raise Exception ('Implement your testing function')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:

        #if pre-trained, load embedding matrix
        """
        if (args.pretrain_emb==True):
            emb_npfn = save_path+args.emb_matrix
            emb_matrix = np.load(emb_npfn)['embed_m']
            if (emb_matrix.shape[0]!= args.vocab_size or emb_matrix.shape[1]!=args.embedding_dim):
                print("Import embedding matrix shape {} does not match shape of ({},{})...".format(emb_matrix.shape, args.vocab_size, args.embedding_dim))
                exit(1)
            else:
                print("Loading embedding matrix.....")
                sess.run(rnnmodel.embedding_mat.assign(emb_matrix))
        """
        saver.restore(sess, path)
        test_state = sess.run([init_state])
        #for Xs in X:
        test_dict = {X_: X,
                     keep_prob: 1,
                     init_state: test_state}

        test_predict = sess.run(y_predict, feed_dict=test_dict)
        if (test_predict>args.threshold):
            print("Predicted result is OHCA.")
        else:
            print("Predicted result is not OHCA.")
    
    with open(args.outfile,'w+') as outfile:
        outstr = "%f" %(test_predict[0])
        print(outstr)
        outfile.write(outstr)
        
    return

if  __name__ == "__main__":
    main()
