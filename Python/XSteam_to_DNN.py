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

p = np.linspace(1.,50,300)
p_tf = tf.linspace()

layers = [1,10,10,1]
layers_fn = [tf.tanh,tf.tanh]

w_tsat_p, b_tsat_p = initialize_NN(layers)

