# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:20:40 2020

@author: garaya
"""


import numpy as np
import matplotlib.pyplot as plt
import correlations
import experiences
import time
from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=14)
plt.rc('axes',titlesize=16)
plt.rc('legend',fontsize=14)
plt.rc('figure',titlesize=20)

mpoint = 0.47 #kg/s
D = 22.9e-3 #m
P_th = 151.8 #kW --> hL et hV sont en kJ/kg
L_c = 1.8 #m
T_e = 215.3 #°C
P_s = 42.1 #bars
z_LC = 1.0


g = 9.81
G = mpoint/(np.pi*0.25*D**2) # Flux massique du mélange
z_e = 0. # Position de l'entrée
z_s = L_c # Position de la sortie

q = P_th / (np.pi * D*L_c)



# =============================================================================
# Debut boucle Nz
# =============================================================================

listeNz = [10,20,50,100,200,500,1000,2000,5000,10000,20000]
listedrop= []
listit = []
listtime = []

for Nz in listeNz:
    print('Iteration pour Nz = %d' % (Nz))
    t1 = time.time()
    # =============================================================================
    # Discretisation
    # =============================================================================
    
    z = np.linspace(z_e,z_s,Nz)
    
    #Operateur de dérivées
    dz = (z_s-z_e)/(Nz-1)
    Dz = (-1.*np.eye(Nz,k=1) + np.eye(Nz))/dz
    #Conditions de dérivée nulle en z=z_e
    Dz[0,0] = 0
    Dz[1,0] = 0
    
    
    # Initialisation des vecteurs P, x, eps
    p = P_s+0.*z
    eps = 0.5 + 0.*z
    x = 0.5 + 0.*z
    
    
    
    # =============================================================================
    # Fonctions du modèle
    # =============================================================================
    
    def ecart_it(x0,x1,p0,p1,eps0,eps1):
        err = np.mean(np.square(x0-x1)) \
            + np.mean(np.square(p0-p1))/P_s \
            + np.mean(np.square(eps0-eps1))
        return np.sqrt(err)
    
    
    
    def DeuxphiModel(x,eps,p):
    
        rho_g = np.asarray([steamTable.rhoV_p(k) for k in p])
        rho_l = np.asarray([steamTable.rhoL_p(k) for k in p])
        mu_g = np.asarray([steamTable.my_ph(k, steamTable.hV_p(k)) for k in p])
        mu_l = np.asarray([steamTable.my_ph(k, steamTable.hL_p(k))for k in p])
        sigma = np.asarray([steamTable.st_p(k)for k in p])
        

        C0,Vgj = correlations.InoueDriftModel_np(eps,x,p,G,D,rho_g,rho_l)
        
        new_eps_diphasique = x/(rho_g*Vgj/G + C0*(rho_g*(1-x)/rho_l+x))
        
        new_eps = np.where(x>0.,new_eps_diphasique,0.)
        
        return new_eps
    
    
    def Energy_equation(x,eps,p):
        
        L_sc = (steamTable.hL_p(P_s)-steamTable.h_pt(P_s,T_e))*mpoint/(np.pi*D*q)
        x_s = 4*q*(L_c-L_sc)/(G*D*(steamTable.hV_p(P_s)-steamTable.hL_p(P_s)))
        
        hvl = np.asarray([steamTable.hV_p(k) - steamTable.hL_p(k) for k in p])
        integrande = 4*q/(D*G*hvl)
        
        new_x = x_s - np.asarray([dz*np.sum(np.where(z>z[k],integrande,0.)) for k in range(Nz)])
        return new_x
    
    def Pressure_equation(x,eps,p):
        
        rho_g = np.asarray([steamTable.rhoV_p(k) for k in p])
        rho_l = np.asarray([steamTable.rhoL_p(k) for k in p])
        mu_g = np.asarray([steamTable.my_ph(k,steamTable.hV_p(k)) for k in p])
        mu_l = np.asarray([steamTable.my_ph(k,steamTable.hL_p(k)) for k in p])
        sigma = np.asarray([steamTable.st_p(k) for k in p])
        
        rho_m = eps*rho_g + (1.-eps)*rho_l
        
        phi2 = correlations.friedel(x,rho_g,rho_l,mu_g,mu_l,G,sigma,D)
        f = 0.079*np.power(G*D/mu_l,-0.25)
        dp_dz_l0 = f*2*G**2/(D*rho_l)    
        
        dp_grav = rho_m*g
        
        dp_acc = (G**2) * np.dot(x*(rho_l-rho_g)/(rho_l*rho_g),Dz)  # Pas certain de l'ordre du produit matrice vecteur avec np.dot
        
        integrande = -phi2*dp_dz_l0 - dp_grav - dp_acc
        
        new_p = P_s - 1e-5*np.asarray([dz*np.sum(np.where(z>z[k],integrande,0.)) for k in range(Nz)])
        return new_p
    
    # =============================================================================
    # Boucle de convergence
    # =============================================================================
    
    
    err = 1.
    it = 0
    target_err = 1e-7
    while err > target_err:
        new_eps = DeuxphiModel(x,eps,p)
        new_x = Energy_equation(x,eps,p)
        new_p = Pressure_equation(x,eps,p)
        
        err = ecart_it(x,new_x,p,new_p,eps, new_eps)
        print('It : %d - Ecart consecutif %.3e' % (it,err))
        it += 1
        
        eps = new_eps
        x = new_x
        p = new_p                         
    
    t2 = time.time()
        
    # =============================================================================
    # Output final
    # =============================================================================
    pdrop = p[0]-p[-1]
    listedrop.append(pdrop)
    listit.append(it)
    listtime.append(t2-t1)





# =============================================================================
# Fin de la boucle
# =============================================================================
errrelative = np.abs(listedrop-listedrop[-1])/listedrop[-1]

plt.figure()
plt.subplot(221)
plt.plot(listeNz,listedrop,marker='o',linestyle='dashed')
plt.xlabel('$N_z$')
plt.xscale('log')
plt.ylabel('$-\\Delta p$')
plt.subplot(222)
plt.plot(listeNz[:-1],errrelative[:-1],marker='o',linestyle='dashed')
plt.plot([215.,2150.,2150.,215.],[1e-3,1e-4,1e-3,1e-3],color='black')
plt.text(300,2e-3,'slope $1/N_z$')
plt.xlabel('$N_z$')
plt.xscale('log')
plt.ylabel('$err_{\\Delta p}$')
plt.yscale('log')
plt.subplot(223)
plt.plot(listeNz,listit,marker='o',linestyle='dashed')
plt.xlabel('$N_z$')
plt.xscale('log')
plt.ylabel('Iterations')
plt.ylim((4.,8.))
plt.subplot(224)
plt.plot(listeNz,listtime,marker='o',linestyle='dashed')
plt.plot([100.,1000.,100.,100.],[1.,10.,10.,1.],color='black')
plt.text(50.,15.,'slope $N_z$')
plt.xlabel('$N_z$')
plt.xscale('log')
plt.ylabel('Time ellapsed (s)')
plt.yscale('log')
plt.tight_layout()