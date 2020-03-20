# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:18:25 2020

@author: garaya
"""

import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
import tensorflow as tf
import pickle
from DNNFunctions import *

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

def gen_data(N):
    x = np.linspace(1.,50,N)
    y = np.asarray([steamTable.hV_t(k) for k in x])
    return x,y

#p_tf = tf.constant(p,dtype=tf.float32,shape=[len(p),1])
#tsat_data_tf = tf.constant(tsat_data,dtype=tf.float32,shape=[len(tsat_data),1])
#
    
x_tf = tf.placeholder(dtype=tf.float32,shape=[None,1])
y_data_tf = tf.placeholder(dtype=tf.float32,shape=[None,1])
# Données d'entrainement
    
    
N = 1000    
x,y_data = gen_data(N)

tf_dict_train = {x_tf : np.reshape(x,(N,1)),
                 y_data_tf : np.reshape(y_data,(N,1))}

N = 10000    
t,hL_data = gen_data(N)
tf_dict_valid = {x_tf : np.reshape(x,(N,1)),
                 y_data_tf : np.reshape(y_data,(N,1))}


layers = [1,40,1]
layers_fn = [tf.tanh,tf.tanh]

w_tsat_p, b_tsat_p = initialize_NN(layers)

Model = neural_net(x_tf,w_tsat_p,b_tsat_p,layers_fn)

# Erreur

Loss = 1e4*tf.reduce_mean(tf.square(Model - y_data_tf))/tf.reduce_mean(tf.square(y_data_tf))

# Optimiseur

optimizer = tf.contrib.opt.ScipyOptimizerInterface(Loss, method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000, #50000
                                                                           'maxfun': 50000, #50000
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(np.float32).eps}) 
    

optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train_op_Adam = optimizer_Adam.minimize(Loss) 

# Lancement de la session

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

sess.run(init)


# Entrainement


optimizer.minimize(sess,
                fetches = [Loss],
                feed_dict = tf_dict_train,
                loss_callback = callback)

loss_value = sess.run(Loss,tf_dict_train)

tolAdam = 1e-5
it=0
itmin = 1e5
while it<itmin and loss_value>tolAdam:
    sess.run(train_op_Adam, tf_dict_train)
    loss_value = sess.run(Loss, tf_dict_train)
    if it%100 == 0:
        loss_validation = sess.run(tf.reduce_mean(Loss),tf_dict_valid)
        print('Adam it %e - Training Loss :  %.3e - Valid loss : %.3e' % (it, loss_value, loss_validation))
    it += 1


# Validation



yguess,xguess = sess.run([x_tf,Model],tf_dict_train)

plt.figure()
plt.plot(xguess[:,0],xguess[:,0],label='Model')
plt.plot(x,y_data,label='data')
plt.legend()

# Erreur validation

loss_validation = sess.run(Loss,tf_dict_valid)
print('Loss valid : %.3e' % (loss_validation))


# Sauvegarde
str_layers_fluid = [str(j) for j in layers]
filename = 'Models/hV_t_'+'_'.join(str_layers_fluid) +'_tanh.DNN'
save_DNN(w_tsat_p,b_tsat_p,filename,sess)
