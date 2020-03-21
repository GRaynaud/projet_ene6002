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


def chexal(jg,j,rho_g,rho_l,mu_g,mu_l,x,G,D,p,sigma,txVide):
    
    Re_g = x * G * D / mu_g
    Re_l = (1-x) * G * D / mu_l

    if Re_g > Re_l or Re_g < 0 :
        Re = Re_g
    else :
            Re = Re_l
 
# Calcul de C0    
    A1 = 1 / (np.exp(-Re/60000) + 1)
    B1 = np.min(0.8,A1)
    r = (1 + 1.57 * rho_g / rho_l) / (1 - B1)
    K0 = B1 + (1 - B1) * np.power(rho_g / rho_l, 0.25)
    C1 = 4 * p_crit**2 / (p * (p_crit - p))

    if C1*txVide > 80 :
        L1 = 1
    else :
        L1 = 1 - np.exp(-C1*txVide)

    if C1>80 :
        L2 = 1
    else :
        L2 = 1 - np.exp(-C1)   
    
    L_cor = L1 / L2
    C0 = L_cor / (K0 + (1 - K0) * np.power(txVide,r))   
    
# Calcul de Vgj

    if Re_g >= 0 :
        K1 = B1
    else :
        K1 = np.min(0.65,0.5*np.exp(np.abs(Re_g)/4000))        

    C5 = np.sqrt(150 * rho_g / rho_l)
    C6 = C5 / (1 - C5)

    if C5 >= 1 :
        C2 = 1
    else :
        C2 = 1 / (1 - np.exp(-C6))

    C3 = np.max(0.5,2*np.exp(-Re_l/60000))
    C7 = np.power(D2/D,0.6)
    C8 = C7 / (1 - C7)  

    if C7 >= 1 :
        C4 = 1
    else :
        C4 = 1 / (1 - np.exp(-C8))
    
    Vgj = 1.41 * np.power((rho_l - rho_g)* sigma * g / rho_l**2,0.25) * np.power(1 - txVide, K1) * C2 * C3 * C4   
    txVide = np.power(rho_g / G * (1 - x) / x * Vgj + C0 *(rho_g / rho_l * (1 - x) / x + 1),1) #a verifier pour le 1/G (tu as mis G_l dans le TeX)
    
    return txVide     


def chexal_tf(rho_g,rho_l,mu_g,mu_l,x,G,D,p,sigma,txVide):
    '''
    Retourne le taux de vide 
    Apres avoir calcule le C0 et V_gj
    Ne fait appel qu'à des fonctions de base tensorflow
    '''
    # Calcul de C0
    
    ones = 0.*txVide + 1.
    
    Re_g = x * G * D / mu_g
    Re_l = (1-x) * G * D / mu_l
    
    Re = tf.where(tf.math.logical_or(tf.less(Re_g, Re_l), tf.less(Re_g,0.)), Re_g, Re_l)
    
    A1 = 1. / (tf.exp(-Re/60000.) + 1.)
    B1 = tf.minimum(0.8,A1)
    r = (1 + 1.57 * rho_g / rho_l) / (1 - B1)
    K0 = B1 + (1 - B1) * tf.pow(rho_g / rho_l, 0.25)
    C1 = 4 * p_crit**2 / (p * (p_crit - p))
    
    boolean_L1 = tf.less(80.*ones,C1*txVide)
    L1 = tf.where(boolean_L1,ones,1.-tf.exp(-C1*txVide))
#    L1 = tf.where(tf.less(80.*ones,C1*txVide),ones,1.-tf.exp(-C1*txVide))
    
    boolean_L2 = tf.less(80.*ones,C1)
    L2 = tf.where(boolean_L2, ones, 1. - tf.exp(-C1))
    
    L_cor = L1 / L2
    
    C0 = L_cor / (K0 + (1 - K0) * tf.pow(txVide,r))   
    
    # Calcul de Vgj    
        
    K1_2 = tf.minimum(0.65, 0.5*tf.exp(tf.abs(Re_g)/4000.))
    K1 = tf.where(tf.less(Re_g,0.), K1_2, B1)

    C5 = tf.sqrt(150 * rho_g / rho_l)
    C6 = C5 / (1 - C5)

    C2 = tf.where(tf.less(C5,ones), 1./(1.-tf.exp(-C6)), ones)

    C3 = tf.maximum(0.5*ones,2.*tf.exp(-Re_l/60000.))
    C7 = tf.pow(D2/D,0.6)*ones
    C8 = C7 / (1. - C7)  
    
    boolean_C4 = tf.less(ones,C7)
    C4 = tf.where(boolean_C4, 1./(1.-tf.exp(-C8)), ones)
    
    Vgj = 1.41 * tf.pow((rho_l - rho_g)* sigma * g / rho_l**2,0.25) * tf.pow(1 - txVide, K1) * C2 * C3 * C4 
    
    
    txVide = 1./(rho_g/(G*x)*Vgj + C0*(rho_l/rho_g*(1-x)/x+1.))

    return txVide  


def friedel(x,rho_g,rho_l,mu_g,mu_l,G,sigma,D):
    
    rho_h = np.power(x / rho_g + (1 - x) / rho_l,1)
    We = G**2 * D / sigma / rho_h #Nombre de Weber
    Fr = G**2 / g / D / rho_h**2  #Nombre de Froude
    H = np.power(rho_g / rho_l,0.91) * np.power(mu_g / mu_l,0.19) * np.power(1 - mu_g / mu_l,0.7)
    F = np.power(x,0.78)*np.power(1-x,0.224)
    E = np.power(1-x,2) + x**2 * rho_l * frictionFac(G,D,mu_g) / rho_g / frictionFac(G,D,mu_l)
    phi2 = E + 3.24 * F * H / (np.power(Fr,0.045) * np.power(We,0.035))
    
    return phi2

def friedel_tf(x,rho_g,rho_l,mu_g,mu_l,G,sigma,D):
    '''
    Retourne le coefficient de correlation de Friedel pour un écoulement diphasique
    Compatible avec tensorflow
    '''
    ones = 0.*x + 1.
    
    rho_h = tf.pow(x / rho_g + (1 - x) / rho_l,1)
    We = G**2 * D / sigma / rho_h #Nombre de Weber
    Fr = G**2 / g / D / rho_h**2  #Nombre de Froude
    H = tf.pow(rho_g / rho_l,0.91) * tf.pow(mu_g / mu_l,0.19) * tf.pow(1 - mu_g / mu_l,0.7)
    F = tf.pow(x,0.78)*tf.pow(1-x,0.224)
    E = tf.pow(1-x,2) + x**2 * rho_l * frictionFac_tf(G,D,mu_g) / rho_g / frictionFac_tf(G,D,mu_l)
    phi2_xinf1 = E + 3.24 * F * H / (tf.pow(Fr,0.045) * tf.pow(We,0.035))
    
    phi2 = tf.where(tf.less(x,ones),phi2_xinf1,E)
    
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
    Cf = 1.325 / tf.pow((tf.log (1e-6/3.7)+5.74/tf.pow(G * D / mu, 0.9)),2)

    return Cf  