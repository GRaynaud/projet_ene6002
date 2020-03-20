#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:05:58 2020

@author: thomaschrp
"""

import numpy as np
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

    txVide = jg / (C0 * j + Vgj)
    
    return txVide     