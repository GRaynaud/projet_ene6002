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
import experiences
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

#####################################################################
#################### Constantes du Problème #########################
#####################################################################

exp = '65BV' # '19' or '65BV'
choix_corr = 'Chexal' #'Chexal' or 'Inoue'

# Exp 19
if exp == '19':
    mpoint = 0.47 #kg/s
    D = 22.9e-3 #m
    P_th = 151.8 #kW --> hL et hV sont en kJ/kg
    L_c = 1.8 #m
    T_e = 215.3 #°C
    P_s = 42.1 #bars
    z_LC = 1.0

elif exp == '65BV':
    # Exp 65BV
    
    mpoint = 0.64 #kg/s
    D = 13.4e-3 #m
    P_th = 250 #kW --> hL et hV sont en kJ/kg
    L_c = 1.8 #m
    T_e = 184 #°C
    P_s = 20.3 #bars
    z_LC = 0.6
    
else:
    print('Erreur ! Experience mal renseigneé')

g = 9.81
G = mpoint/(np.pi*0.25*D**2) # Flux massique du mélange
z_e = 0. # Position de l'entrée
z_s = L_c # Position de la sortie

q = P_th / (np.pi * D*L_c)


# Titre en sortie pour la BC
L_sc = (steamTable.hL_p(P_s)-steamTable.h_pt(P_s,T_e))*mpoint/(np.pi*D*q)
x_s = 4*q*(L_c-L_sc)/(G*D*(steamTable.hV_p(P_s)-steamTable.hL_p(P_s)))
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
# RHO8G
#ptest = np.linspace(1.,50.,1300)
#mu_g_data = np.asarray([steamTable.my_ph(k,steamTable.hV_p(k)) for k in ptest])
#tf_dict = {p_tf : np.reshape(ptest,(1300,1))}
#mu_g_guess = sess.run(muV_p(p_tf),tf_dict)[:,0]
#
#print('hV_p Normalised std : %.3e' % (np.std(hV_guess-hV_data)/np.mean(hV_data)))
#
#plt.figure()
#plt.plot(ptest,mu_g_data,label='data')
#plt.plot(ptest,mu_g_guess,label='Model')
#plt.legend()


#####################################################################
#######################  Fonctions du PB  ###########################
#####################################################################

# P = f(z)
layers_P = [1,20,1]
layers_fn_P = [tf.tanh,tf.tanh]
w_P,b_P = DNN.initialize_NN(layers_P,'Pressure')

def P_z(z):
    p_z_temp = P_s * DNN.neural_net(z,w_P,b_P,layers_fn_P)
#    pmin = z*0. + P_s*0.8
#    pmax = z*0 + P_s*1.2
#    return tf.minimum(pmax,tf.maximum(pmin,p_z_temp))
    return p_z_temp

# T = f(P) @ saturation --> Hypothèse à vérifier

#T_z = Tsat_p(P_z)

# x = f(z) à déterminer

layers_x = [1,20,1]
layers_fn_x = [tf.tanh,tf.tanh]
w_x,b_x = DNN.initialize_NN(layers_x,'Quality')

def x_z(z):
    x_z_temp = DNN.neural_net(z,w_x,b_x,layers_fn_x)
#    ones = 0.*z + 1. - 1e-2
#    zeroes = 0.*z + 1e-2
#    return tf.minimum(ones,tf.maximum(zeroes,x_z_temp))
    return x_z_temp


# eps = f(z) à déterminer

layers_eps = [1,20,1]
layers_fn_eps = [tf.tanh,tf.tanh]
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
if choix_corr == 'Chexal':
    def correlation_function(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps):
        return correlations.chexal_tf(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps)
elif choix_corr == 'Inoue':
    def correlation_function(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps):
        return correlations.InoueDriftModel(eps,x,P,G,D,rho_g,rho_l)
else:
    print('Erreur - correlation function mal définie')

def loss_Model(z):
    '''
    Retrourne l'écart quadratique moyen entre 
    la valeur devinée du taux de vide et celle
    calculée à posteriori avec le Drift Flux Model
    et la correlation de 
    '''
    P = P_z(z)
    x = x_z(z)
    eps = eps_z(z)
#    
#    rho_g = steamTable.rhoV_p(P_s) + 0.*z #rhoV_p(P)
#    rho_l = steamTable.rhoL_p(P_s) + 0.*z #rhoL_p(P) #--> est-ce qu on les fait dépendre de z aussi ?
#    
    rho_g = rhoV_p(P)
    rho_l = rhoL_p(P) 
#    
#    mu_g = steamTable.my_ph(P_s, steamTable.hV_p(P_s))  + 0.*z  # muV_p(P)
#    mu_l = steamTable.my_ph(P_s, steamTable.hL_p(P_s))  + 0.*z  #muL_p(P)
#    sigma = steamTable.st_p(P_s)  + 0.*z  #st_p(P)
    
    mu_g =  muV_p(P)
    mu_l = muL_p(P)
    sigma = st_p(P)

#    x_z_guess = correlations.InoueDriftModel(eps,x,P,G,D,rho_g,rho_l)
#    x_z_guess = correlations.chexal_tf(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps)
#    
    x_z_guess = correlation_function(rho_g,rho_l,mu_g,mu_l,x,G,D,P,sigma,eps)
##    x_z_guess = correlations.HomogeneousModel(eps,rho_g,rho_l)
    err = x_z_guess - x

#    eps_z_guess = tf.where(tf.less(0.*z,x),correlations.InoueDriftModel_tf_eps(x,P,G,D,rho_g,rho_l),0.*z)
#    
#    err = eps_z_guess - eps
    return tf.reduce_mean(tf.square(err))

def loss_energy_equation(z):
    '''
    Retourne l'erreur quadratoique moyen sur
    le bilan d'énergie avec les valeurs 
    de titre et de pression devinées
    '''
    
    P = P_z(z)
    x = x_z(z)
#    A = hL_p(P)+x*(hV_p(P)-hL_p(P))
    
    A = (1-x)*hL_p(P) + x*hV_p(P)
    
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
       
#    rho_g = steamTable.rhoV_p(P_s) + 0.*z 
#    rho_l = steamTable.rhoL_p(P_s) + 0.*z 
   
    rho_g = rhoV_p(P)
    rho_l = rhoL_p(P) 
    
#    condition_rho_m = tf.math.logical_and(tf.less(eps,ones),tf.less(zeroes,eps))
#    rho_m = tf.where( condition_rho_m,, tf.where(tf.less(eps,zeroes), rho_l, rho_g) )
#    
    rho_m =  tf.nn.relu(eps)*rho_g + tf.nn.relu((1.-eps))*rho_l
#    rho_m = tf.maximum(rho_g , tf.minimum(rho_m_normal, rho_l))
#    
#    mu_g = steamTable.my_ph(P_s, steamTable.hV_p(P_s))  + 0.*z  # muV_p(P)
#    mu_l = steamTable.my_ph(P_s, steamTable.hL_p(P_s))  + 0.*z  #muL_p(P)
#    sigma = steamTable.st_p(P_s)  + 0.*z  #st_p(P)
# 
    mu_g =  muV_p(P)
    mu_l = muL_p(P)
    sigma = st_p(P)
    
    phi2 = correlations.friedel_tf(x, rho_g, rho_l, mu_g, mu_l, G, sigma, D)
    
#    f = 0.316*tf.pow(tf.abs(1-x)*G*D/mu_l + 1e-4, -0.25)
#    Re_l = tf.abs((1-x))*G*D/mu_l 
#    Re_g = tf.abs(x)*G*D/mu_g
#    f= tf.where(tf.less(x,ones), 0.316*tf.pow(Re_l,-0.25),  0.316*tf.pow(Re_g,-0.25)) # eq (10.13) --> A verifier dans le cas x>1
#    
    
#    f = 0.316*tf.pow(G*D/mu_l*tf.exp(-x), -0.25) # --> Idée smoother le 1-x afin de ne pas passer en négatif
        #dp_dz_l0 = f*0.5*(G**2)/(rho_m*D) #eq (10.8)
    
    f = 0.079*tf.pow(G*D/mu_l,-0.25)
    dp_dz_l0 = f*2*G**2/(D*rho_l)
    
    # dp_acc * eps^2*(1-eps)^2
#    Fac_dp_acc = tf.abs(eps)*tf.square(1.-eps)*tf.gradients(tf.square(x)/rho_g,z)[0] \
#                + tf.square(eps)*tf.abs((1.-eps))*tf.gradients(tf.square(1-eps)/rho_l, z)[0] \
#                + ( tf.square(eps)*tf.square(1.-eps)/rho_l - tf.square(1.-eps)*tf.square(x)/rho_g )*tf.gradients(eps,z)[0]
#    
#    
    dp_acc = G**2*tf.gradients(x*(rho_l-rho_g)/(rho_l*rho_g),z)[0] #eq (10.2)
    
    
    dp_grav = rho_m*g # eq (10.17)
    
    dP_dz = 1e5*tf.gradients(P,z)[0] # Attention conversion bar --> Pa
    
    
#    err = tf.square(eps)*tf.square(1.-eps)*( dP_dz + phi2*dp_dz_l0 + dp_grav ) + Fac_dp_acc # Attention aux signes des termes --> A priori ok
    err = dP_dz + phi2*dp_dz_l0 + dp_grav + dp_acc
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
    #x_s_guess = x_z(z_s_tf)
    
    
    err = tf.square(P_s_guess/P_s - 1.)  #+  tf.square(x_s_guess-x_s) # + tf.square(x_e) 
    return tf.reduce_mean(err)


def loss_eps_01(z):
    '''
    Impose au taux de vide de rester borné entre 0 et 1
    '''
    eps = eps_z(z)
    err = tf.nn.relu(-1.*eps) + tf.nn.relu(eps-1.) 
    return tf.reduce_mean(tf.square(err))


def loss_signe_eps_x(z):
    '''
    Impose à epsilon et x d'être de même signe
    '''
    eps = eps_z(z)
    x = x_z(z)
    err = tf.nn.relu(-x*eps) 
    return tf.reduce_mean(tf.square(err))
    
    
#def loss_x_01(z):
#    '''
#    Impose au titre de rester borné entre 0 et 1
#    '''
#    x = x_z(z)
#        
#    err = tf.nn.relu(-1.*x) + tf.nn.relu(x-1.)
#    
#    return tf.reduce_mean(tf.square(err))
#####################################################################
#######################  Fns Coûts du PB  ###########################
#####################################################################
# Construction de l'erreur que l'on cherche à minimiser
    
Loss =  1e3*loss_BC() \
        + loss_energy_equation(z_tf) \
        + loss_eps_01(z_tf) + loss_signe_eps_x(z_tf) \
        + loss_pressure_equation(z_tf) \
        + 1e-1*loss_Model(z_tf) \
       
Loss_preinit = tf.reduce_mean(tf.square(eps_z(z_tf)- (0.4 + (0.6-0.4)*(z_tf-z_e)/(z_s-z_e)))) \
            + tf.reduce_mean(tf.square( x_z(z_tf) - (0.05 + (0.8-0.05)*(z_tf-z_e)/(z_s-z_e)) )) \
            + tf.reduce_mean(tf.square( P_z(z_tf) - (P_s +1.  - 1.*(z_tf-z_e)/(z_s-z_e) ) ))
            


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
    

optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5,epsilon=1e-8) #tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5) #
train_op_Adam_preinit = optimizer_Adam.minimize(Loss_preinit)
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

loss_value_preinit = sess.run(Loss_preinit,tf_dict_train)
it = 0
while loss_value_preinit>1e-5:
    sess.run(train_op_Adam_preinit, tf_dict_train)
    loss_value_preinit = sess.run(Loss_preinit, tf_dict_train)
    if it%1000 == 0:
        print('Pre init %d - Loss %.3e' % (it,loss_value_preinit))
    it += 1

print('Fin du préentrainement : loss pr-init %.3e' % (sess.run(Loss_preinit,tf_dict_train)))

#####################################################################
#########################  Entrainement  ############################
#####################################################################  
print('Debut de l\'entrainement')

#optimizer.minimize(sess,
#                fetches = [Loss],
#                feed_dict = tf_dict_train,spot
#                loss_callback = DNN.callback)

loss_value = sess.run(Loss,tf_dict_train)
print('Loss value : %.3e' % (loss_value))

tolAdam = 1e-5
it=0
itmin = 5e4
lr = optimizer_Adam._lr
for k in range(3):
    it = 0
    while it<itmin and loss_value>tolAdam:
        sess.run(train_op_Adam, tf_dict_train)
        loss_value = sess.run(Loss, tf_dict_train)
        if it%100 == 0:
            print('Adam %d it %e - Training Loss :  %.6e' % (k+1 ,it, loss_value))
        it += 1
        
    lr*=0.1
    optimizer_Adam._lr = lr
        

    
### Output


print('Répartition des erreurs résiduelles')
print('Erreur BC : %.3e' % (sess.run(loss_BC(),tf_dict_train)))
print('Erreur Drift flux model : %.3e' % (sess.run(loss_Model(z_tf),tf_dict_train)))
print('Erreur Energy : %.3e' % (sess.run(loss_energy_equation(z_tf),tf_dict_train)))
print('Erreur chute pression : %.3e' % (sess.run(loss_pressure_equation(z_tf),tf_dict_train)))
print('Erreur pénalisation eps : %.3e' % (sess.run(loss_eps_01(z_tf),tf_dict_train)))
#print('Erreur pénalisation x : %.3e' % (sess.run(loss_x_01(z_tf),tf_dict_train)))
print('Erreur pénalisation x*eps : %.3e' % (sess.run(loss_signe_eps_x(z_tf),tf_dict_train)))
print('Saut de pression total : %.3e bar' % (sess.run(P_z(tf.constant(z_e,shape=[1,1])))[0,0]-P_s))

z_o,p_o,eps_o,x_o = sess.run([z_tf,P_z(z_tf),eps_z(z_tf),x_z(z_tf)],tf_dict_train)

plt.figure(figsize=(7,6))
plt.subplot(211)
plt.plot(z_o,x_o,label='$x$',c='black')
plt.plot(z_o,eps_o,label='$\\epsilon$',c='orange')
plt.hlines(0.,0.,L_c,linestyle='dotted')
plt.hlines(1.,0.,L_c,linestyle='dotted')
if exp == '19':
    plt.plot(experiences.z_eps_19,experiences.eps_19,linestyle='none',marker='s',label='$\\epsilon_{data}$', c='orange')
elif exp == '65BV':
    plt.plot(experiences.z_eps_65,experiences.eps_65,linestyle='none',marker='s',label='$\\epsilon_{data}$', c='orange')
plt.vlines(z_LC,-0.1,1.1,linestyle='dashed',color='red')    
plt.xlabel('$z$ axis (m)')
plt.ylabel('Dimensionless quantities')
plt.legend()
plt.subplot(212)
plt.plot(z_o,p_o,label='$p$',c='blue')   
if exp == '19':      
    plt.plot(experiences.z_p,P_s+1e-2*experiences.p_19,linestyle='none',marker='^',label='$p_{data}$',c='blue')
elif exp == '65BV':
    plt.plot(experiences.z_p,P_s+1e-2*experiences.p_65,linestyle='none',marker='^',label='$p_{data}$',c='blue')
plt.vlines(z_LC,np.min(p_o),np.max(p_o),linestyle='dashed',color='red')
plt.xlabel('$z$ axis (m)')
plt.ylabel('Pressure (bars)')
plt.legend()
plt.tight_layout()


plt.savefig('Resultats/Output_PINN_'+choix_corr+'_'+exp+'.png')
plt.savefig('Resultats/Output_PINN_'+choix_corr+'_'+exp+'.pgf')
print('Figure sauvergardée')
print('Ok, normal End')

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
    [ sess.run(loss_pressure_equation(z_tf),tf_dict_train), sess.run(loss_energy_equation(z_tf),tf_dict_train), sess.run(loss_Model(z_tf),tf_dict_train) ]
    np.min(x),np.max(x)
    np.min(eps),np.max(eps) 
