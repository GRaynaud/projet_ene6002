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
import correlations
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
g = 9.81

z_e = 0. # Position de l'entrée
z_s = z_e + L_c # Position de la sortie

q = P_th / (np.pi * D*L_c)

#####################################################################
###################### Imports tensorflow ###########################
#####################################################################

p_tf   = tf.placeholder(dtype=tf.float32,shape=[None,1])
t_tf   = tf.placeholder(dtype=tf.float32,shape=[None,1])
z_tf   = tf.placeholder(dtype=tf.float32,shape=[None,1])
eps_tf = tf.placeholder(dtype=tf.float32,shape=[None,1])


#
#init = tf.compat.v1.global_variables_initializer()
#
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#
#sess.run(init)

#####################################################################
####################### Modèles XSteam TF ###########################
#####################################################################

# Import
filename_tsat_p = 'Models/Tsat_p_1_40_1_tanh.DNN'
filename_hL_p = 'Models/hL_p_1_40_1_tanh.DNN'
filename_hV_p = 'Models/hV_p_1_40_1_elu.DNN'
filename_rhoV_p = 'Models/rhoV_p_1_40_1_tanh.DNN'
filename_rhoL_p = 'Models/rhoL_p_1_40_1_tanh.DNN'
filename_st_p = 'Models/st_p_1_40_1_elu.DNN'
filename_muL_p = 'Models/1e4xmyL_p_1_10_1_elu.DNN'
filename_muV_p = 'Models/1e5xmyV_p_1_10_1_elu.DNN'

layers = [1,40,1]
layers_fn = [tf.tanh, tf.tanh, tf.tanh]
w_tsat_p, b_tsat_p = DNN.restore_NN_as_constant(layers,filename_tsat_p)
w_hL_p,b_hL_p = DNN.restore_NN_as_constant(layers,filename_hL_p)
w_hV_p,b_hV_p = DNN.restore_NN_as_constant(layers,filename_hV_p)
w_rhoV_p,b_rhoV_p = DNN.restore_NN_as_constant(layers,filename_rhoV_p)
w_rhoL_p,b_rhoL_p = DNN.restore_NN_as_constant(layers,filename_rhoL_p)
w_st_p,b_st_p = DNN.restore_NN_as_constant(layers,filename_st_p)
w_muL_p,b_muL_p = DNN.restore_NN_as_constant([1,10,1],filename_muL_p)
w_muV_p,b_muV_p = DNN.restore_NN_as_constant([1,10,1],filename_muV_p)

def Tsat_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_tsat_p,b_tsat_p,layers_fn)

def hL_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_hL_p,b_hL_p,layers_fn)

def hV_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_hV_p,b_hV_p,[tf.nn.elu])

def rhoV_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_rhoV_p,b_rhoV_p,layers_fn)

def rhoL_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_rhoL_p,b_rhoL_p,layers_fn)

def st_p(p_input_tf):
    return DNN.neural_net(p_input_tf,w_st_p,b_st_p,[tf.nn.elu])

def muL_p(p_input_tf):
    return 1e-4*DNN.neural_net(p_input_tf,w_muL_p,b_muL_p,[tf.nn.elu])

def muV_p(p_input_tf):
    return 1e-5*DNN.neural_net(p_input_tf,w_muV_p,b_muV_p,[tf.nn.elu])

# Vérification
#
## Tsat_p
#ptest = np.linspace(1.,50.,400)
#t_data = np.asarray([steamTable.tsat_p(k) for k in ptest])
#tf_dict = {p_tf : np.reshape(ptest,(400,1))}
#t_guess = sess.run(Tsat_p(p_tf),tf_dict)[:,0]
#
#print('Tsat_p Normalised std : %.3e' % (np.std(t_guess-t_data)/np.mean(t_data)))
#
#plt.figure()
#plt.plot(ptest,t_data,label='data')
#plt.plot(ptest,t_guess,label='Model')
#plt.legend()
#
## hL_t
#ptest = np.linspace(1.,50.,400)
#hL_data = np.asarray([steamTable.hL_p(k) for k in ptest])
#tf_dict = {p_tf : np.reshape(ptest,(400,1))}
#hL_guess = sess.run(hL_p(p_tf),tf_dict)[:,0]
#
#print('hL_p Normalised std : %.3e' % (np.std(hL_guess-hL_data)/np.mean(hL_data)))
#
#plt.figure()
#plt.plot(ptest,hL_data,label='data')
#plt.plot(ptest,hL_guess,label='Model')
#plt.legend()
#
## hV_t
#ptest = np.linspace(1.,50.,400)
#hV_data = np.asarray([steamTable.hV_p(k) for k in ptest])
#tf_dict = {p_tf : np.reshape(ptest,(400,1))}
#hV_guess = sess.run(hV_p(p_tf),tf_dict)[:,0]
#
#print('hV_p Normalised std : %.3e' % (np.std(hV_guess-hV_data)/np.mean(hV_data)))
#
#plt.figure()
#plt.plot(ptest,hV_data,label='data')
#plt.plot(ptest,hV_guess,label='Model')
#plt.legend()
#

#####################################################################
#######################  Fonctions du PB  ###########################
#####################################################################

# P = f(z)
layers_P = [1,20,1]
layers_fn_P = [tf.tanh]
w_P,b_P = DNN.initialize_NN(layers_P,'Pressure')

def P_z(z):
    return DNN.neural_net(z,w_P,b_P,layers_fn_P)

# T = f(P) @ saturation --> Hypothèse à vérifier

#T_z = Tsat_p(P_z)

# x = f(z) à déterminer

layers_x = [1,20,1]
layers_fn_x = [tf.tanh]
w_x,b_x = DNN.initialize_NN(layers_x,'Quality')

def x_z(z):
    return DNN.neural_net(z,w_x,b_x,layers_fn_x)


# eps = f(z) à déterminer

layers_eps = [1,20,1]
layers_fn_eps = [tf.tanh]
w_eps,b_eps = DNN.initialize_NN(layers_eps,'Epsilon')

def eps_z(z):
    return DNN.neural_net(z,w_eps,b_eps,layers_fn_eps)


#####################################################################
#######################  Equations du PB  ###########################
#####################################################################


def loss_txVide_DriftFluxModel(z):
    '''
    Retrourne l'écart quadratique moyen entre 
    la valeur devinée du taux de vide et celle
    calculée à posteriori avec le Drift Flux Model
    et la correlation de 
    '''
    P = P_z(z)
    x = x_z(z)
    eps = eps_z(z)
    
    rho_g = rhoV_p(P)
    rho_l = rhoL_p(P) #--> est-ce qu on les fait dépendre de z aussi ?
    
    mu_g = muV_p(P)
    mu_l = muL_p(P)
    sigma = st_p(P)
    
    eps_z_guess = correlations.chexal_tf(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps)
    
    err = eps_z_guess - eps
    
    return tf.reduce_mean(tf.square(err))

def loss_energy_equation(z):
    '''
    Retourne l'erreur quadratoique moyen sur
    le bilan d'énergie avec les valeurs 
    de titre et de pression devinées
    '''
    P = P_z(z)
    x = x_z(z)
    A = hL_p(P)+x*(hV_p(P)-hL_p(P))
    
    dA_dz = tf.gradients(A,z)[0]
    
    err = dA_dz - 4.*q/(D*G)
    return tf.reduce_mean(tf.square(err))
    
    
def loss_pressure_equation(z):
    '''
    Retourne l'erreur quadratique moyenne entre 
    le gradient de pression calculé avec le DNN et 
    celui obtenu avec le facteur de correlation de Fridel
    '''
    P = P_z(z)
    x = x_z(z)
    eps = eps_z(z)
    
    rho_g = rhoV_p(P)
    rho_l = rhoL_p(P) #--> est-ce qu on les fait dépendre de z aussi ?
    
    rho_m = eps*rho_g + (1.-eps)*rho_l
    
    mu_g = muV_p(P)
    mu_l = muL_p(P)
    sigma = st_p(P)
    
    phi2 = correlations.friedel_tf(x, rho_g, rho_l, mu_g, mu_l, G, sigma, D)
    
    mu = x*mu_g + (1-x)*mu_l # eq (10.11) modele de Chicchitti et al, 1960
    f_tp = 0.316/tf.pow(G*D/mu,0.25) # eq (10.13)
    dp_dz_l0 = f_tp*0.5*(G**2)/(rho_m*D) #eq (10.8)
    
    G2vp = (G**2)*( tf.square(x)/(eps*rho_g) + tf.square(1.-x)/((1.-eps)*rho_l) ) # eq(10.21)
    dp_acc = tf.gradients(G2vp,z)[0] # eq (10.20) terme 2
    
    dp_grav = rho_m*g # eq (10.17)
    
    dP_dz = tf.gradients(P,z)[0]
    
    
    err = dP_dz - phi2*dp_dz_l0 - dp_grav - dp_acc # Attention aux signes des termes --> A VERIFIER !!!
    
    return tf.reduce_mean(tf.square(err))


def loss_BC():
    '''
    Erreur sur les conditions aux limites 
    Entrée : T = T_e (°C) à z = z_e
    Sortie : P = P_s (bar) à z = z_s
    '''
    z_e_tf = z_e*tf.ones(shape=[1,1],dtype=tf.float32)
    z_s_tf = z_s*tf.ones(shape=[1,1],dtype=tf.float32)
    
    P_e_guess = P_z(z_e_tf)
    T_e_guess = Tsat_p(P_e_guess)
    P_s_guess = P_z(z_s_tf)
    
    err = tf.square(T_e_guess - T_e) + tf.square(P_s_guess - P_s)
    return tf.reduce_mean(err)


#####################################################################
#######################  Fns Coûts du PB  ###########################
#####################################################################
# Construction de l'erreur que l'on cherche à minimiser
    
Loss =  1e-10*(  loss_txVide_DriftFluxModel(z_tf) \
        + loss_energy_equation(z_tf) \
        + loss_pressure_equation(z_tf) \
        + loss_BC()) # Nan sur loss_txVide... et loss_pressure...
        
Loss_preinit = tf.reduce_mean(tf.square(eps_z(z_tf)-0.5)) \
            + tf.reduce_mean(tf.square( x_z(z_tf) - (0.2 + (0.8-0.2)*(z_tf-z_e)/(z_s-z_e)) )) \
            + tf.reduce_mean(tf.square( P_z(z_tf) - ( steamTable.psat_t(T_e) + (P_s-steamTable.psat_t(T_e))*(z_tf-z_e)/(z_s-z_e) ) ))
            


#####################################################################
###########################  Optimiseurs  ###########################
#####################################################################
# Declaration et parametrage des optimiseurs        

optimizer_preinit = tf.contrib.opt.ScipyOptimizerInterface(Loss_preinit, method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000, #50000
                                                                           'maxfun': 50000, #50000
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(np.float32).eps}) 
    

        
optimizer = tf.contrib.opt.ScipyOptimizerInterface(Loss, method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000, #50000
                                                                           'maxfun': 50000, #50000
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(np.float32).eps}) 
    

optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-6)
train_op_Adam = optimizer_Adam.minimize(Loss)         
        
        
#####################################################################
#########################  2e init vars  ############################
#####################################################################  


init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

sess.run(init)

#####################################################################
#########################  Prepa. train  ############################
#####################################################################  

Ntrain = 1000 # Nombre de points où évaluer l'erreur
z_train = np.linspace(z_e,z_s,Ntrain)
tf_dict_train = {z_tf : np.reshape(z_train,(Ntrain,1))}

#####################################################################
#########################  Pre training  ############################
#####################################################################  
print('Debut du pre entrainement')

optimizer_preinit.minimize(sess,
                fetches = [Loss_preinit],
                feed_dict = tf_dict_train,
                loss_callback = DNN.callback)

#####################################################################
#########################  Entrainement  ############################
#####################################################################  
print('Debut de l\'entrainement')

#optimizer.minimize(sess,
#                fetches = [Loss],
#                feed_dict = tf_dict_train,
#                loss_callback = DNN.callback)

loss_value = sess.run(Loss,tf_dict_train)
print('Loss value : %.3e' % (loss_value))

tolAdam = 1e-5
it=0
itmin = 5e4
while it<itmin and loss_value>tolAdam:
    sess.run(train_op_Adam, tf_dict_train)
    loss_value = sess.run(Loss, tf_dict_train)
    if it%100 == 0:
        print('Adam it %e - Training Loss :  %.3e' % (it, loss_value))
    it += 1