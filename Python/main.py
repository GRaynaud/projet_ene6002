# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:51:40 2020

"""
#####################################################################
######################## Bibliotheques ##############################
#####################################################################


import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
import tensorflow as tf
import pickle
import DNNFunctions as DNN
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

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
filename_tsat_p = 'Models/Tsat_p_1_40_1_tanh.DNN'
filename_hL_t = 'Models/hL_t_1_40_1_tanh.DNN'
filename_hV_t = 'Models/hV_t_1_40_1_tanh.DNN'
layers = [1,40,1]
layers_fn = [tf.tanh, tf.tanh, tf.tanh]
w_tsat_p, b_tsat_p = DNN.restore_NN_as_constant(layers,filename_tsat_p)
w_hL_t,b_hL_t = DNN.restore_NN_as_constant(layers,filename_hL_t)
w_hV_t,b_hV_t = DNN.restore_NN_as_constant(layers,filename_hV_t)

def Tsat_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_tsat_p,b_tsat_p,layers_fn)

def hL_t(t_input_tf):
    return DNN.neural_net(t_input_tf,w_hL_t,b_hL_t,layers_fn)

def hV_t(t_input_tf):
    return DNN.neural_net(t_input_tf,w_hV_t,b_hV_t,layers_fn)

# Vérification

# Tsat_p
ptest = np.linspace(1.,50.,400)
t_data = np.asarray([steamTable.tsat_p(k) for k in ptest])
tf_dict = {p_tf : np.reshape(ptest,(400,1))}
t_guess = sess.run(Tsat_p(p_tf),tf_dict)[:,0]

print('Tsat_p Normalised std : %.3e' % (np.std(t_guess-t_data)/np.mean(t_data)))

plt.figure()
plt.plot(ptest,t_data,label='data')
plt.plot(ptest,t_guess,label='Model')
plt.legend()

# hL_t
Ttest = np.linspace(1.,300.,400)
hL_data = np.asarray([steamTable.hL_t(k) for k in Ttest])
tf_dict = {t_tf : np.reshape(Ttest,(400,1))}
hL_guess = sess.run(hL_t(t_tf),tf_dict)[:,0]

print('hL_p Normalised std : %.3e' % (np.std(hL_guess-hL_data)/np.mean(hL_data)))

plt.figure()
plt.plot(Ttest,hL_data,label='data')
plt.plot(Ttest,hL_guess,label='Model')
plt.legend()

# hV_t
Ttest = np.linspace(1.,300.,400)
hV_data = np.asarray([steamTable.hV_t(k) for k in Ttest])
tf_dict = {t_tf : np.reshape(Ttest,(400,1))}
hV_guess = sess.run(hV_t(t_tf),tf_dict)[:,0]

print('hV_p Normalised std : %.3e' % (np.std(hV_guess-hV_data)/np.mean(hV_data)))

plt.figure()
plt.plot(Ttest,hV_data,label='data')
plt.plot(Ttest,hV_guess,label='Model')
plt.legend()


#####################################################################
#######################  Fonctions du PB  ###########################
#####################################################################

# P = f(z)
layers_P = [1,40,1]
layers_fn_P = [tf.tanh]
w_P,b_P = DNN.initialize_NN(layers_P,'Pressure')

P_z = DNN.neural_net(z_tf,w_P,b_P,layers_fn_P)

# T = f(P) @ saturation --> Hypothèse à vérifier

T_z = Tsat_p(P_z)

# x = f(z) à déterminer

layers_x = [1,40,1]
layers_fn_x = [tf.tanh]
w_x,b_x = DNN.initialize_NN(layers_x,'Quality')

x_z = DNN.neural_net(z_tf,w_x,b_x,layers_fn_x)


# epsilon = f(x) --> Equation derivee du drift flux model
V_gj =  ...
C_O = ...
rho_g = ...
rho_l = ... --> est-ce qu on les fait dépendre de z aussi ?

eps_z = 1./((rho_g*(1.-x_z))/(G_l*x_z)*V_gj + C_0*(rho_l/rho_g*(1-x_z)/x_z+1.))

#####################################################################
#######################  Equations du PB  ###########################
#####################################################################

def loss_energy_equation():
    
def loss_pressure_equation():
