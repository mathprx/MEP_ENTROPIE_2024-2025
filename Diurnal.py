import mep_cycle
import numpy as np
import constants as constants
import display

Tref = constants.Tref
Sover4 = constants.Sover4

duration = 86400 #duree en seconde de l'experience (86400 pour une jounée par exemlpe)
z_ref = 0.2 #profondeur caractéristique pour laquel il y des variations de température
c_p = 2e6 #capacité thermique VOLUMIQUE du sous-sol

#Cp = c_p*z_ref #capacité thermique SURFACIQUE du sous-sol
Cp = 1*Sover4*duration/Tref
#b = 10 #coefficient convecto-conductif (surfacique) de transmission de chaleur entre le sol et l'athmosphère
b = 10* Sover4/Tref


N_time = 24
N_atm = 5

def forcing(t): #fonction duration-périodique qui donne l'éclairement au temps i*dt
    return max(0,-4*np.cos(t*2*np.pi/duration))

simulation = mep_cycle.simulation(duration,b,Cp,N_atm,N_time,forcing)

np.savetxt('courbe7.txt',simulation["result"])
display.atmospheric_profiles(simulation)
display.surface_time_evolution_diurnal(simulation)