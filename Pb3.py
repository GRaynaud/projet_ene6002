# -*- coding: utf-8 -*-
# Bibliotheques

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)

# Import des données extrqites avec WebPlotDigitizer

liste_vit_massique = [83, 130, 213, 325, 480] #kg/(m^2 s)
liste_filenames = [str(v)+'.csv' for v in liste_vit_massique]

liste_data = []
for i in range(len(liste_vit_massique)) : 
    filename = liste_filenames[i]
    T = np.genfromtxt(filename, skip_header = 0, delimiter = ';')
    X = T[:,0]
    Y = T[:,1]
    liste_data.append(np.array([X,Y]))

liste_marker = ['^', 'o', 's', 'p', 'v']
liste_color = ['blue', 'red', 'orange', 'black', 'purple']

# Constantes physiques

p = 11. # bars
rho_g = 46.65 #kg/m^3
rho_l = 1186. #kg/m^3
sigma = 7811e-6 #N/m tension superficielle
g = 9.81
d_tube = 1e-2
A_tube = np.pi*(d_tube/2.)**2
# Constantes du modèle de Maier et Coddigton

C1, C2 = 2.57e-3, 1.0062
C0_Maier = C1*p + C2
def Vgj_Maier(G):
    v1, v2, v3, v4, v5, v6 = 6.73e-7, -8.81e-5, 1.05e-3, 5.63e-3, -1.23e-1, 8e-1
    return (v1*p**2 + v2*p + v3)*G + v4*p**2 + v5*p + v6

# Constantes d'Inoue et al 
C0_Inoue = 6.76e-3*p + 1.026
def Vgj_Inoue(G):
    W = A_tube*G
    return (5.1e-3*W + 6.91e-2)*(9.42e-2*p**2-1.99*p+12.6)

# Correlation de SUn et al
C0_Sun = 1.13
Vgj_Sun = 1.41*np.power(sigma*g*(rho_l-rho_g)/(rho_l**2),0.25)

# Calcul

def eps(x,G,C0,Vgj):
    G_l = (1-x)*G
    inveps = (rho_g/G_l)*(1-x)/x*Vgj + C0*(rho_g/rho_l * (1-x)/x + 1)
    return 1./inveps



# Plot avec correlation de Maier

plt.figure()
for i in range(len(liste_data)):
    x_data = liste_data[i][0]
    eps_data = liste_data[i][1]
    G = liste_vit_massique[i]
    plt.plot(x_data,eps_data, linestyle = 'none', marker = liste_marker[i], color=liste_color[i], label=str(G))
    x_model = np.linspace(0.05, 0.5, 1000)
    eps_model = np.array([eps(x,G,C0_Maier,Vgj_Maier(G)) for x in x_model])
    plt.plot(x_model,eps_model, linestyle='dashed', color = liste_color[i])
plt.xlim((0.,0.5))
plt.ylim((0.1,0.9))
plt.title('Correlation de Maier')    
plt.ylabel('Taux de vide moyen $\\varepsilon$')
plt.xlabel('Titre thermodynamique $x_{th} = \\hat{x}$')
plt.legend(title='G (kg/m$^2$ s)', loc='lower right')
plt.tight_layout()
plt.savefig('MaierCorrelation.png')
plt.close()


# Plot avec correlation de Inoue

plt.figure()
for i in range(len(liste_data)):
    x_data = liste_data[i][0]
    eps_data = liste_data[i][1]
    G = liste_vit_massique[i]
    plt.plot(x_data,eps_data, linestyle = 'none', marker = liste_marker[i], color=liste_color[i], label=str(G))
    x_model = np.linspace(0.05, 0.5, 1000)
    eps_model = np.array([eps(x,G,C0_Sun,Vgj_Inoue(G)) for x in x_model])
    plt.plot(x_model,eps_model, linestyle='dashed', color = liste_color[i])
plt.xlim((0.,0.5))
plt.ylim((0.1,0.9))
plt.title('Correlation de Inoue')    
plt.ylabel('Taux de vide moyen $\\varepsilon$')
plt.xlabel('Titre thermodynamique $x_{th} = \\hat{x}$')
plt.legend(title='G (kg/m$^2$ s)', loc='lower right')
plt.tight_layout()
plt.savefig('InoueCorrelation.png')
plt.close()

# Plot avec correlation de Sun

plt.figure()
for i in range(len(liste_data)):
    x_data = liste_data[i][0]
    eps_data = liste_data[i][1]
    G = liste_vit_massique[i]
    plt.plot(x_data,eps_data, linestyle = 'none', marker = liste_marker[i], color=liste_color[i], label=str(G))
    x_model = np.linspace(0.05, 0.5, 1000)
    eps_model = np.array([eps(x,G,C0_Sun,Vgj_Sun) for x in x_model])
    plt.plot(x_model,eps_model, linestyle='dashed', color = liste_color[i])
plt.xlim((0.,0.5))
plt.ylim((0.1,0.9))
plt.title('Correlation de Sun')    
plt.ylabel('Taux de vide moyen $\\varepsilon$')
plt.xlabel('Titre thermodynamique $x_{th} = \\hat{x}$')
plt.legend(title='G (kg/m$^2$ s)', loc='lower right')
plt.tight_layout()
plt.savefig('SunCorrelation.png')
plt.close()