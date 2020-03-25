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

mpoint = 0.47 #kg/s
D = 22.9e-3 #m
P_th = 151.8 #kW --> hL et hV sont en kJ/kg
L_c = 1.8 #m
T_e = 215.3 #°C
P_s = 42.1 #bars
g = 9.81

G = mpoint/(np.pi*0.25*D**2) # Flux massique du mélange
z_e = 0. # Position de l'entrée
z_s = L_c # Position de la sortie

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
layers_P = [1,10,1]
layers_fn_P = [tf.tanh]
w_P,b_P = DNN.initialize_NN(layers_P,'Pressure')

def P_z(z):
    p_z_temp = DNN.neural_net(z,w_P,b_P,layers_fn_P)
#    pmin = z*0. + P_s*0.8
#    pmax = z*0 + P_s*1.2
#    return tf.minimum(pmax,tf.maximum(pmin,p_z_temp))
    return p_z_temp

# T = f(P) @ saturation --> Hypothèse à vérifier

#T_z = Tsat_p(P_z)

# x = f(z) à déterminer

layers_x = [1,10,1]
layers_fn_x = [tf.tanh]
w_x,b_x = DNN.initialize_NN(layers_x,'Quality')

def x_z(z):
    x_z_temp = DNN.neural_net(z,w_x,b_x,layers_fn_x)
#    ones = 0.*z + 1. - 1e-2
#    zeroes = 0.*z + 1e-2
#    return tf.minimum(ones,tf.maximum(zeroes,x_z_temp))
    return x_z_temp


# eps = f(z) à déterminer

layers_eps = [1,10,1]
layers_fn_eps = [tf.tanh]
w_eps,b_eps = DNN.initialize_NN(layers_eps,'Epsilon')

def eps_z(z):
    eps_z_temp = DNN.neural_net(z,w_eps,b_eps,layers_fn_eps)
#    ones = 0.*z + 1. - 1e-2
#    zeroes = 0.*z + 1e-2
#    return tf.minimum(ones,tf.maximum(zeroes,eps_z_temp))
    return eps_z_temp


#####################################################################
#######################  Equations du PB  ###########################
#####################################################################


def loss_DriftFluxModel(z):
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
    
    x_z_guess = correlations.chexal_tf(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps)
    
    err = x_z_guess - x
    
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
    
    err = dA_dz*D*G/(4.*q) - 1. # --> Reflechir sur le signe +/-1. A priori c'est un -
    return tf.reduce_mean(tf.square(err))
    
    
def loss_pressure_equation(z):
    '''
    Retourne l'erreur quadratique moyenne entre 
    le gradient de pression calculé avec le DNN et 
    celui obtenu avec le facteur de correlation de Fridel
    '''
    ones = 0.*z + 1. - 1e-4
    zeroes = 0.*z + 1e-4
    
    P = P_z(z)
    x = x_z(z)
    eps = eps_z(z)
    
    rho_g = rhoV_p(P)
    rho_l = rhoL_p(P) #--> est-ce qu on les fait dépendre de z aussi ?
    
    condition_rho_m = tf.math.logical_and(tf.less(eps,ones),tf.less(zeroes,eps))
    rho_m = tf.where( condition_rho_m, eps*rho_g + (1.-eps)*rho_l, tf.where(tf.less(eps,zeroes), rho_l, rho_g) )
    
    mu_g = muV_p(P)
    mu_l = muL_p(P)
    sigma = st_p(P)
    
    phi2 = correlations.friedel_tf(x, rho_g, rho_l, mu_g, mu_l, G, sigma, D)
    
    
    f= tf.where(tf.less(x,ones), 0.316*tf.pow((1-x)*G*D/mu_l,-0.25), 0.316*tf.pow(x*G*D/mu_g,-0.25)) # eq (10.13) --> A verifier dans le cas x>1
    dp_dz_l0 = f*0.5*(G**2)/(rho_m*D) #eq (10.8)
    
    # dp_acc * eps^2*(1-eps)^2
    Fac_dp_acc = eps*tf.square(1.-eps)*tf.gradients(tf.square(x)/rho_g,z)[0] \
                + tf.square(eps)*(1.-eps)*tf.gradients(tf.square(1-eps)/rho_l, z)[0] \
                + ( tf.square(eps)*tf.square(1.-eps)/rho_l - tf.square(1.-eps)*tf.square(x)/rho_g )*tf.gradients(eps,z)[0]
    
    
    dp_grav = rho_m*g # eq (10.17)
    
    dP_dz = tf.gradients(P,z)[0]
    
    
    err = tf.square(eps)*tf.square(1.-eps)*( dP_dz + phi2*dp_dz_l0 + dp_grav ) + Fac_dp_acc # Attention aux signes des termes --> A priori ok
    err_norm = err/dp_grav
    return tf.reduce_mean(tf.square(err_norm))


def loss_BC():
    '''
    Erreur sur les conditions aux limites 
    Entrée : T = T_e (°C) à z = z_e
    Sortie : P = P_s (bar) à z = z_s
    '''
    z_e_tf = z_e*tf.ones(shape=[1,1],dtype=tf.float32)
    z_s_tf = z_s*tf.ones(shape=[1,1],dtype=tf.float32)
    
    #P_e_guess = P_z(z_e_tf)
    #T_e_guess = Tsat_p(P_e_guess)
    P_s_guess = P_z(z_s_tf)
    x_e = x_z(z_e_tf)
    eps_e = eps_z(z_e_tf)
    
    
    err = tf.square(P_s_guess/P_s - 1.)  #+  tf.square(eps_e) # + tf.square(x_e) 
    return tf.reduce_mean(err)


#####################################################################
#######################  Fns Coûts du PB  ###########################
#####################################################################
# Construction de l'erreur que l'on cherche à minimiser
    
Loss =  loss_pressure_equation(z_tf)  + loss_BC()  + loss_energy_equation(z_tf) \
#        + loss_DriftFluxModel(z_tf) \
#        + loss_energy_equation(z_tf) \
#        + loss_BC() # Nan sur loss_txVide... et loss_pressure...
        
Loss_preinit = tf.reduce_mean(tf.square(eps_z(z_tf)- (0.4 + (0.6-0.4)*(z_tf-z_e)/(z_s-z_e)))) \
            + tf.reduce_mean(tf.square( x_z(z_tf) - (0.05 + (0.8-0.05)*(z_tf-z_e)/(z_s-z_e)) )) \
            + tf.reduce_mean(tf.square( P_z(z_tf) - P_s ))
            


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
    

optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5,epsilon=1e-5)
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

tolAdam = 1e-6
it=0
itmin = 1e5
while it<itmin and loss_value>tolAdam:
#    z,p,eps,x = sess.run([z_tf,P_z(z_tf),eps_z(z_tf),x_z(z_tf)],tf_dict_train)
#    grads = optimizer_Adam.compute_gradients(Loss)
#    grads_value = sess.run(grads, tf_dict_train)
#    mingrads = np.min(np.asarray([np.min(k) for k in grads_value]))
#    if mingrads != mingrads:
#        print('Pb --> Nan in grads')
#        break
#    else:
    sess.run(train_op_Adam, tf_dict_train)
#    optimizer_Adam.apply_gradients(grads)
    loss_value = sess.run(Loss, tf_dict_train)
    if it%10 == 0:
        print('Adam it %e - Training Loss :  %.6e' % (it, loss_value))
    it += 1
    
    
    
### Output


print('Répartition des erreurs résiduelles')
print('Erreur BC : %.3e' % (sess.run(loss_BC(),tf_dict_train)))
print('Erreur Drift flux model : %.3e' % (sess.run(loss_DriftFluxModel(z_tf),tf_dict_train)))
print('Erreur Energy : %.3e' % (sess.run(loss_energy_equation(z_tf),tf_dict_train)))
print('Erreur chute pression : %.3e' % (sess.run(loss_pressure_equation(z_tf),tf_dict_train)))


z_o,p_o,eps_o,x_o = sess.run([z_tf,P_z(z_tf),eps_z(z_tf),x_z(z_tf)],tf_dict_train)

plt.figure()
plt.subplot(211)
plt.plot(z_o,x_o,label='Titre x')
plt.plot(z_o,eps_o,label='Eps')
plt.hlines(0.,z_e,z_s)
plt.hlines(1.,z_e,z_s)
plt.legend()
plt.subplot(212)
plt.plot(z_o,p_o,label='Pression') 
plt.legend()
plt.tight_layout()   

#while it<itmin:
##    z,p,eps,x = sess.run([z_tf,P_z(z_tf),eps_z(z_tf),x_z(z_tf)],tf_dict_train)
#    grads = optimizer_Adam.compute_gradients(Loss)
#    mg = tf.reduce_min(tf.convert_to_tensor([tf.reduce_min(k) for k in grads]))
#    op = tf.where(tf.math.equal(mg,mg),tf.ones(shape=[1,1]),0.*tf.ones(shape = [1,1]))
#    a = sess.run(op, tf_dict_train)
#    if a[0,0] == 0. :
#        print('Err')
#        break
#    elif a[0,0] == 1. :
#        sess.run(train_op_Adam,tf_dict_train)
#        print('.',end='')
#    else :
#        print('Err autre')
#        
#    loss_value = sess.run(Loss, tf_dict_train)
#    if it%10 == 0:
#        print('Adam it %e - Training Loss :  %.6e' % (it, loss_value))
#    it += 1
#    
#    

if False:
    z,p,eps,x = sess.run([z_tf,P_z(z_tf),eps_z(z_tf),x_z(z_tf)],tf_dict_train)
    np.min(p),np.max(p),np.min(eps),np.max(eps),np.min(x),np.max(x)
    p
    eps
    x
    z
    [ sess.run(loss_pressure_equation(z_tf),tf_dict_train), sess.run(loss_energy_equation(z_tf),tf_dict_train), sess.run(loss_DriftFluxModel(z_tf),tf_dict_train) ]
    np.min(x),np.max(x)
    np.min(eps),np.max(eps) 
