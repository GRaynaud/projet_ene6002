# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:51:40 2020

"""

import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
import tensorflow as tf
import pickle
from DNNFunctions import *



#####################################################################
#################### Constantes du Problème #########################
#####################################################################
# Exp 65BV

G = 0.64 #kg/s
D = 13.4e-3 #m
P_th = 250.0e3 #W
L_c = 1.8 #m
T_e = 184.0 #°C
P_s = 20.3 #bars

#####################################################################
###################### Imports tensorflow ###########################
#####################################################################

p_tf   = tf.placeholder(dtype=tf.float32,shape=[None,1])
t_tf   = tf.placeholder(dtype=tf.float32,shape=[None,1])
z_tf   = tf.placeholder(dtype=tf.float32,shape=[None,1])
eps_tf = tf.placeholder(dtype=tf.float32,shape=[None,1])



init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

sess.run(init)

#####################################################################
####################### Modèles XSteam TF ###########################
#####################################################################

# Import
filename_tsat_p = 'Tsat_p_1_40_1_tanh.DNN'
filename_hL_t = 'hL_t_1_40_1_tanh.DNN'
filename_hV_t = 'hV_t_1_40_1_tanh.DNN'

w_tsat_p, b_tsat_p = 


# Vérification







