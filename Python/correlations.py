 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:05:58 2020

@author: thomaschrp
"""

import numpy as np
import tensorflow as tf
#Constantes et dimensions
D2 = 0.091 #m
p_crit = 221.2 #bars
g = 9.81 #m/s2


def chexal_np(txVide,x,G,D,p,sigma,rho_g,rho_l,mu_g,mu_l):

    Re_g = x * G * D / mu_g
    Re_l = (1-x) * G * D / mu_l

    Re = np.where(np.logical_or(np.less(Re_l, Re_g), np.less(Re_g,0)), Re_g, Re_l)

# Calcul de C0
    A1 = 1 / (np.exp(-Re/60000) + 1)
    B1 = np.minimum(0.8,A1)
    r = (1 + 1.57 * rho_g / rho_l) / (1 - B1)
    K0 = B1 + (1 - B1) * np.power(rho_g / rho_l, 0.25)
    C1 = 4 * p_crit**2 / (p * (p_crit - p))

    boolean_L1 = np.less(80,C1*txVide)
    L1 = np.where(boolean_L1,1.-np.exp(-C1*txVide),1)

    boolean_L2 = np.less(80,C1)
    L2 = np.where(boolean_L2, 1. - np.exp(-C1),1)

    L_cor = L1 / L2
    C0 = L_cor / (K0 + (1 - K0) * np.power(txVide,r))

# Calcul de Vgj

    C9 = np.where(np.greater_equal(Re_g,0),np.power(np.abs(1-txVide),B1),np.minimum(0.7,np.power(np.abs(1-txVide),0.65)))

    C5 = np.sqrt(150 * rho_g / rho_l)
    C6 = C5 / (1 - C5)

    C2 = np.where(np.less(C5,1), 1./(1.-np.exp(-C6)), 1)

    C3 = np.maximum(0.5,2*np.exp(-Re_l/60000))
    C7 = np.power(D2/D,0.6)
    C8 = C7 / (1 - C7)

    C4 = np.where(np.less(C7,1), 1./(1.-np.exp(-C8)), 1)

    Vgj = 1.41 * np.power((rho_l - rho_g)* sigma * g / rho_l**2,0.25) * C9 * C2 * C3 * C4

    return C0,Vgj


def chexal_tf(rho_g,rho_l,mu_g,mu_l,x,G,D,p,sigma,txVide):
    '''
    Retourne le titre
    Apres avoir calcule le C0 et V_gj
    Ne fait appel qu'à des fonctions de base tensorflow
    '''
    # Calcul de C0

    ones = 0.*txVide + 1. - 1e-20
    zeroes = 0.*txVide + 1e-20

    Re_g = x * G * D / mu_g
    Re_l = (1-x) * G * D / mu_l

    Re = tf.where(tf.math.logical_or(tf.less(Re_l, Re_g), tf.less(Re_g,zeroes)), Re_g, Re_l)

    A1 = 1. / (tf.exp(-Re/60000.) + 1.)
    B1 = tf.minimum(0.8,A1)
    r = (1 + 1.57 * rho_g / rho_l) / (1 - B1)
    K0 = B1 + (1 - B1) * tf.pow(rho_g / rho_l, 0.25)
    C1 = 4 * p_crit**2 / (tf.maximum(p,1e-2*ones) * (p_crit - p)) # Pb si p = 0...

    boolean_L1 = tf.less(80.*ones,C1*txVide)
    L1 = tf.where(boolean_L1,1.-tf.exp(-C1*txVide),ones) #--> A verifier la condition

    boolean_L2 = tf.less(80.*ones,C1)
    L2 = tf.where(boolean_L2, 1. - tf.exp(-C1),ones)

    L_cor = L1 / L2

    C0 = L_cor / (K0 + (1 - K0) * tf.pow(tf.abs(txVide),r))# Si eps<0 --> C0 = exp(-eps)-1 pour pénaliser un eps <0


    # Calcul de Vgj

    C9 = tf.where(tf.greater_equal(Re_g,zeroes),tf.pow(tf.abs(1-txVide),B1),tf.minimum(0.7,tf.pow(tf.abs(1-txVide),0.65)))

    C5 = tf.sqrt(150 * rho_g / rho_l)
    C6 = C5 / (1 - C5)

    C2 = tf.where(tf.less(C5,ones), 1./(1.-tf.exp(-C6)), ones)

    C3 = tf.maximum(0.5*ones,2.*tf.exp(-tf.abs(Re_l)/60000.)) # valeur absolue en sureté
    C7 = tf.pow(D2/D,0.6)*ones
    C8 = C7 / (1. - C7)

    boolean_C4 = tf.less(C7,ones)
    C4 = tf.where(boolean_C4, 1./(1.-tf.exp(-C8)), ones)


    Vgj = 1.41 * tf.pow((rho_l - rho_g)* sigma * g / rho_l**2,0.25) * C9 * C2 * C3 * C4



    condition_xguess1 = tf.math.logical_and(tf.less(txVide,ones),tf.less(zeroes,txVide))
    condition_xguess2 = tf.math.logical_and(tf.less(x,ones),tf.less(zeroes,x))
    condition_xguess = tf.math.logical_and(condition_xguess1,condition_xguess2)

    xguess_normal = txVide*rho_g*Vgj/G + txVide*C0*( (1.-x)*rho_g/rho_l + x )

    xguess = tf.where(condition_xguess, xguess_normal, x + tf.exp(-1.*x))

    return xguess




def InoueDriftModel_np(txVide,x,p,G,D,rho_g,rho_l):
    C0 = 6.76e-3*p+1.026
    W = G*np.pi*0.25*(D**2)
    Vgj = ( 5.1e-3*W+6.91e-2 ) * ( 9.42e-2*np.square(p) - 1.99*p + 12.6 )
    return C0,Vgj

def InoueDriftModel(txVide,x,p,G,D,rho_g,rho_l):
    C0 = 6.76e-3*p+1.026
    W = G*np.pi*0.25*(D**2)
    Vgj = ( 5.1e-3*W+6.91e-2 ) * ( 9.42e-2*tf.square(p) - 1.99*p + 12.6 )

    xguess = txVide*(rho_g*Vgj/G + C0*( (1.-x)*rho_g/rho_l + x ))
    return xguess

def InoueDriftModel_tf_eps(x,p,G,D,rho_g,rho_l):
    C0 = 6.76e-3*p+1.026
    W = G*np.pi*0.25*(D**2)
    Vgj = ( 5.1e-3*W+6.91e-2 ) * ( 9.42e-2*tf.square(p) - 1.99*p + 12.6 )

    eps_guess = tf.where(x>0.*x,x/(rho_g*Vgj/G + C0*(rho_g*(1-x)/rho_l+x)),0.*x)

    return eps_guess

def HomogeneousModel(txVide,rho_g,rho_l):

    xguess = txVide*rho_g/((1.-txVide)*rho_l + txVide*rho_g )
    return xguess

def friedel(x,rho_g,rho_l,mu_g,mu_l,G,sigma,D):

    rho_h = np.power(x / rho_g + (1 - x) / rho_l,1)
    We = G**2 * D / sigma / rho_h #Nombre de Weber
    Fr = G**2 / g / D / rho_h**2  #Nombre de Froude
    H = np.power(rho_g / rho_l,0.91) * np.power(mu_g / mu_l,0.19) * np.power(1 - mu_g / mu_l,0.7)
    F = np.power(x,0.78)*np.power(1-x,0.224)
    E = np.power(1-x,2) + x**2 * rho_l * frictionFac(G,D,mu_g) / rho_g / frictionFac(G,D,mu_l)
    phi2_xinf1 = E + 3.24 * F * H / (np.power(Fr,0.045) * np.power(We,0.035))

    condition = np.logical_and(x<1., x>0.)
    phi2_positif = np.where(condition, phi2_xinf1, E)
    phi2 = np.where(x>0., phi2_positif, 1.)
    
    return phi2

def friedel_tf(x,rho_g,rho_l,mu_g,mu_l,G,sigma,D):
    '''
    Retourne le coefficient de correlation de Friedel pour un écoulement diphasique
    Compatible avec tensorflow
    '''
    ones = 0.*x + 1. - 1e-20
    zeroes = 0.*x + 1e-20

    rho_h = tf.abs(1./(x / rho_g + (1 - x) / rho_l))
    We = G**2 * D / sigma / rho_h #Nombre de Weber
    Fr = G**2 / g / D / rho_h**2  #Nombre de Froude
    H = tf.pow(rho_g / rho_l,0.91) * tf.pow(mu_g / mu_l,0.19) * tf.pow(1 - mu_g / mu_l,0.7)
    F = tf.pow(tf.abs(x),0.78)*tf.pow(tf.abs(1-x),0.224)
    E = tf.square(1-x) + tf.square(x) * rho_l * frictionFac_tf(G,D,mu_g) / rho_g / frictionFac_tf(G,D,mu_l)
    phi2_xinf1 = E + 3.24 * F * H / (tf.pow(Fr,0.045) * tf.pow(We,0.035))

    condition_phi2 = tf.math.logical_and( tf.less(x,ones) , tf.less(zeroes,x) ) # Vérifie que 0 < x < 1 sinon F --> NaN

    phi2_positiv = tf.where(condition_phi2,phi2_xinf1,E)
    phi2 = tf.where(tf.less(0.*x,x),phi2_positiv,1.+0.*x)
    return phi2

def frictionFac (G,D,mu):

    Cf = 1.325 / np.power((np.log (1e-6/3.7)+5.74/np.power(G * D / mu, 0.9)),2)

    return Cf

def frictionFac_tf(G,D,mu):
    '''
    Retourne le facteur de friction
    Compatible TensorFlow
    Utilise une rugosité xi/D = 1e-6
    '''
    Cf = 1.325 / tf.square(tf.math.log(1e-6/3.7)+5.74/tf.pow(G * D / mu, 0.9))

    return Cf
