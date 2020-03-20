# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:22:14 2020

@author: garaya
"""


import tensorflow as tf
import numpy as np
import pickle

def restore_NN_as_constant(layers,filename_restore,name_nn=''):  
    print('------------------------------')
    print('Import du modele as constant...')      
    weights = []
    biases = []
    num_layers = len(layers) 
    file = open(filename_restore,'rb')
    Data = pickle.load(file)
    file.close()
    for l in range(0,num_layers-1):
        W = tf.constant(Data[0][l], dtype=tf.float32, shape=[layers[l], layers[l+1]],name = 'weights_'+name_nn+str(l))
        b = tf.constant(Data[1][l], dtype=tf.float32, name ='biases_'+name_nn+str(l))
        weights.append(W)
        biases.append(b)        
    return weights, biases



def restore_NN(layers,filename_restore,name_nn=''):  
    print('------------------------------')
    print('Import du modele as variable...')      
    weights = []
    biases = []
    num_layers = len(layers) 
    file = open(filename_restore,'rb')
    Data = pickle.load(file)
    file.close()
    for l in range(0,num_layers-1):
        W = tf.Variable(Data[0][l], dtype=tf.float32, shape=[layers[l], layers[l+1]],name = 'weights_'+name_nn+str(l))
        b = tf.Variable(Data[1][l], dtype=tf.float32, name ='biases_'+name_nn+str(l))
        weights.append(W)
        biases.append(b)        
    return weights, biases


def initialize_NN(layers,name_nn=''):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_init(size,name_w):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32, name = name_w)

def neural_net(X, weights, biases, layers_fn):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        fn = layers_fn[l]
        W = weights[l]
        b = biases[l]
        H = fn(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def save_DNN(w,b,filename,sess):
    print('------------------------------')
    print('Sauvegarde du modele...')   
    
    Data_fluid = [sess.run(w),sess.run(b)]
    pcklfile = open(filename,'ab+')
    pickle.dump(Data_fluid,pcklfile)
    pcklfile.close()
    print('Modele exporté dans '+filename)
    print('------------------------------')

def callback(lossvalue):
    print('Loss: %.3e' % (lossvalue))
