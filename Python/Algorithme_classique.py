# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import correlations
import experiences
from pyXSteam.XSteam import XSteam
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)


#####################################################################
#################### Constantes du Problème #########################
#####################################################################
# Exp 19

mpoint = 0.47 #kg/s
D = 22.9e-3 #m
P_th = 151.8 #kW --> hL et hV sont en kJ/kg
L_c = 1.8 #m
T_e = 215.3 #°C
P_s = 42.1 #bars
g = 9.81

## Exp 65BV
#
#mpoint = 0.64 #kg/s
#D = 13.4e-3 #m
#P_th = 250 #kW --> hL et hV sont en kJ/kg
#L_c = 1.8 #m
#T_e = 184 #°C
#P_s = 20.3 #bars
#g = 9.81

G = mpoint/(np.pi*0.25*D**2) # Flux massique du mélange
z_e = 0. # Position de l'entrée
z_s = L_c # Position de la sortie

q = P_th / (np.pi * D*L_c)


# =============================================================================
# Discretisation
# =============================================================================
# Nombre de points de discretisation
Nz = 1000

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
    
#    C0 = 1.
#    Vgj = 0.01
    
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

    plt.close()
    plt.figure()
    plt.subplot(211)
    plt.plot(z,x,label='Titre x',c='black')
    plt.plot(z,eps,label='Eps',c='orange')
    plt.hlines(0.,0.,L_c,linestyle='dashed')
    plt.hlines(1.,0.,L_c,linestyle='dashed')
    plt.plot(experiences.z_eps_19,experiences.eps_19,linestyle='none',marker='s',label='eps exp', c='orange')
    #plt.plot(experiences.z_eps_65,experiences.eps_65,linestyle='none',marker='s',label='eps exp', c='orange')
    plt.legend()
    plt.subplot(212)
    plt.plot(z,p,label='Pression',c='blue')         
    plt.plot(experiences.z_p,P_s+1e-2*experiences.p_19,linestyle='none',marker='^',label='exp',c='blue')
    #plt.plot(experiences.z_p,P_s+1e-2*experiences.p_65,linestyle='none',marker='^',label='exp',c='blue')
    plt.tight_layout()
    
    plt.pause(1.)
    
    
    