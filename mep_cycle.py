"""
Toute ce code s'intéresse à déterminer l'évolution d'une colonne d'athmosphère et de sol
constituée de N_atm boîtes d'athmosphère empilées au dessus [indices 2 à N_atm +1]
d'une boite de sol [indice 1] (qui contient un fine couche d'atmosphère à l'equilibre thermique ave le sol)
et d'une boite de sous-sol [indice 0]
pendant une certaine durée (duration)

Chaque boîte d'atmosphère est de masse constante, si bien que la pression est une fonction linéaire de l'indice des boîtes
 Dans toute la suite:
    -l'indice i fait référence au temps (le ième instant considéré), il est donc entre 0 et N_time-1
    -l'indice j fait référence à la couche considée, il est donc entre 0 (pour le sous-sol) et N_atm+1 (pour la couche d'athmosphère la plus haute)

    -Sous nos hypothèses l'état d'une colonne est entièrement caractérisé par les températures dans chaque boîte
    -On modélise donc le système à un insant i*dt par un vecteur x_i de taille N_atm+2 où x_i[j] = Tref/Température (dans la boite j)
    -Pour modéliser toute l'expérience (avec le passage du temps) on considère le vecteur X de taille N_time*(N_atm+2) obtenu par concaténation de x_0,x_1,...,x_N_time
    -La raison pour laquelle on considère ce <<grand vecteur>> plutôt qu'une matrice est lié au fait que la fonction scipy.optimize.minimize (voir plus loin) prend en argument un vecteur et non une matrice
"""

import constants as constants
import numpy as np


from radiatif import radiation # on importe une classe du code radiatif de Didier Paillard et Quentin Pikeroen
from scipy.optimize import minimize
import matplotlib.pyplot as plt

Tref = constants.Tref #tepérature de référence (en Kelvin) 
Sover4 = constants.Sover4 #constante solaire divisée par 4

def toT (X): #Transforme un vecteur de la forme X en un vecteur des températures
    return Tref/X

def simulation(duration #duree en seconde de l'experience (86400 pour une jounée par exemlpe)
               ,b #coefficient convecto-conductif (surfacique) de transmission de chaleur entre le sol et l'athmosphère
               ,Cp #capacité thermique (surfacique) du sous-sol
               ,N_atm #Nombre de boîtes D'ATMOSPHERE
               ,N_time # Nombre de pas de temps. REMARQUE:  il faut que N_time != 2 (pour des raisons de redondances des contraintes)
               ,forcing #fonction qui donne l'éclairement au temps i
                ):
    
    dt = duration/N_time #le pas de temps
    
    def x(X,i,j):
        return X[i*(N_atm+2)+j]

    def T(X,i,j): #obient la température à l'instant i*dt dans la jème couche
        return Tref/X[i*(N_atm+2)+j]
    
    radi = [radiation(N_atm, eps = forcing((i*dt))) for  i in range(N_time)] # juste un objet abstrait intermédiaire qui permet de créer la fonction R, eps regle la quantité de rayonnment solaire incident
    def R_adim(x_i,i): #Prend en argument un des vecteur x_i[1:] ainsi qu'un instant i et renvoit un vecteur contenant le bilan radiatif adimensionné (divisé par Sover4) dans chaque boîte sous forme d'un vecteur 
        return radi[i].bilanR(x_i)
    
    def Q_adim(X,i): #prends en argument X et un instant i et renvoit Q/Sover4 à l'instant i'
        return b*(T(X,i,0)-T(X,i,1))/Sover4

    def contrainte_bilan_radiatif_colonne (i): #renvoit une fonction qui prend en argument un vecteur X et renvoit le bilan énergétique adimensionné de la colonne à l'instant i*dt
        def bilan_radadiatif_colonne(X) :
            x_i = X[i*(N_atm+2) + 1: (i+1)*(N_atm+2)]
            return np.sum(R_adim(x_i,i)) + Q_adim(X,i)
        return bilan_radadiatif_colonne


    def contrainte_stockage_de_chaleur(i): 
        def bilan_energie_sol(X):
            return (Cp/(2*dt*Sover4))*(T(X,(i+1)%N_time,0)-T(X,(i-1)%N_time,0)) + Q_adim(X,i)
        return bilan_energie_sol
    
    def moins_entropie_cree(X):
        S = 0
        for i in range (N_time):
            x_i = X[i*(N_atm+2) +1 : (i+1)*(N_atm+2)]
            S += np.sum (x_i*R_adim(x_i,i)) + Q_adim(X,i)*x_i[0]
        return S



    X0 = np.ones(N_time*(N_atm+2))
    contraintes = []
    for i in range (N_time) :
        contraintes.append({'type': 'eq', 'fun': contrainte_bilan_radiatif_colonne(i)})#contraintes de bilan énergétique de l'atmosphère'
        contraintes.append({'type': 'eq', 'fun': contrainte_stockage_de_chaleur(i)})#Bilan énergétique du sol

    opti = minimize(moins_entropie_cree, X0, constraints = contraintes, options={"maxiter":500} )

    print(opti['message'])
    print(opti['nit'])
    X_res = toT(opti['x'])

    output = {}
    output["result"]=np.array([X_res[i*(N_atm+2):(i+1)*(N_atm+2)] for i in range(N_time)])
    output["N_atm"]=N_atm
    output["N_time"] = N_time
    output["b"] = b
    output["Cp"] = Cp
    output["duration"] = duration
    output["forcing"] = forcing

    return output
    


