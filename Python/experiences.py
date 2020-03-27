#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:41:41 2020

@author: thomaschrp
"""

import numpy as np
import matplotlib.pyplot as plt

z_p = np.array([0,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8])
p_65 = np.array([319.7,312.5,302.2,285.8,260.6,224.1,177.8,119.5,56.2,0])
p_19 = np.array([18.7,16.8,14.9,13.1,11.1,9.0,6.7,4.6,2.2,0])

z_eps_65 = np.array([0.27,0.34,0.50,0.53,0.63,0.72,0.81,0.9,0.99,1.08,1.17,1.22,1.35,1.44,1.54,1.58])
eps_65 = np.array([0,17,22,30,31,51,58,61,68,76,72,83,87,89,87,87])/100
z_eps_19 = np.array([0.2,0.27,0.36,0.45,0.54,0.63,0.72,0.81,0.9,0.99,1.08,1.17,1.26,1.35,1.44,1.53,1.62,1.73])
eps_19 = np.array([2,4,11,16,22,26,34,37,42,44,46,49,53,54,55,57,60,62])/100




if __name__ == "__main__":
    plt.figure(1)
    plt.plot(z_p ,p_65 ,'ro',label="experience 65")
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Perte pression")
    
    plt.figure(2)
    plt.plot(z_p ,p_19 ,'bo',label="experience 19")
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Perte pression")
    
    plt.figure(3)
    plt.plot(z_eps_65 ,eps_65 ,'ro',label="experience 65")
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("txvide")
    
    plt.figure(4)
    plt.plot(z_eps_19 ,eps_19 ,'bo',label="experience 19")
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("txvide")