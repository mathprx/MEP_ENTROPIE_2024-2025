from sys import maxunicode

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def surface_time_evolution_diurnal(simulation):
    N_time = simulation["N_time"]
    res = simulation["result"]
    forcing = simulation["forcing"]
    duration = simulation["duration"]
    dt = duration/N_time
    y_air = [res[i][2] for i in range(N_time)] + [res[0][2]]
    y_surf = [res[i][1] for i in range(N_time)] + [res[0][1]]
    y_sous = [res[i][0] for i in range(N_time)] + [res[0][0]]
    maxy=max(y_air+y_surf+y_sous)
    miny=min(y_air+y_surf+y_sous)
    f = [forcing(i*dt) for i in range(N_time + 1)]
    minf=min(f)
    maxf=max(f)
    y_ec = [(e-minf)*(maxy-miny)/(maxf-minf)+miny for e in f]

    x = [i*dt/3600 for i in range(N_time+1)] #3600 est le nombre de secondes par heures

    plt.plot(x, y_air, label='Air temperature')
    plt.plot(x, y_surf, label='Surface temperature')
    plt.plot(x, y_sous, label='Underground temperature')
    plt.plot(x, y_ec, color ='red', linestyle='dashed', label='Solar radiative flux')

    plt.ylabel("Temperature (in Kelvin)")
    plt.xlabel("Time of the day")
    ax = plt.gca()
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 24, 1))
    hours = [f"{hour:02d}h" for hour in range(24)]
    ax.set_xticklabels(hours)
    plt.legend()
    plt.tight_layout()
    plt.show()

def surface_time_evolution_annual(simulation):
    N_time = simulation["N_time"]
    res = simulation["result"]
    forcing = simulation["forcing"]
    duration = simulation["duration"]
    dt = duration/N_time

    y_air = [res[i][2] for i in range(N_time)] + [res[0][2]]
    y_surf = [res[i][1] for i in range(N_time)] + [res[0][1]]
    y_sous = [res[i][0] for i in range(N_time)] + [res[0][0]]
    maxy=max(y_air+y_surf+y_sous)
    miny=min(y_air+y_surf+y_sous)
    f = [forcing(i*dt) for i in range(N_time + 1)]
    minf=min(f)
    maxf=max(f)
    y_ec = [(e-minf)*(maxy-miny)/(maxf-minf)+miny for e in f]
    x = [i*dt/2.628e6 for i in range(N_time+1)] #2.628e6 est le nombre de secondes dans un mois (en moyenne)

    plt.plot(x, y_air, label='Air temperature')
    plt.plot(x, y_surf, label='Surface temperature')
    plt.plot(x, y_sous, label='Underground temperature')
    plt.plot(x, y_ec, color ='red', linestyle='dashed', label='Solar radiative flux')

    plt.ylabel("Temperature (in Kelvin)")
    axes = plt.gca()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes.set_xticks(np.arange(0.5, 12.5, 1))
    axes.set_xticklabels(months)
    axes.set_xticks(np.arange(0, 12, 1), minor=True)
    axes.tick_params(axis='x', which='major', length=0)
    axes.set_xlabel("Month")
    axes.set_xlim(0,12)
    plt.tight_layout()
    plt.legend()
    plt.show()


def atmospheric_profiles (simulation) :
    res = simulation["result"]
    N_atm = len(res[0])-2
    N_time = len(res)
    y = 1013.25 - np.array(range(N_atm)) * 1013.25 / (N_atm) - 1013.25 / (N_atm) / 2
    y = np.concatenate((np.array([1013.25]), y))
    graph, L = plt.subplots(1, N_time)
    Tmin, Tmax = np.min(res), np.max(res)
    for i in range(N_time):
        if N_time == 1:
            p = L
        else:
            p = L[i]
        x = res[i][1: N_atm + 2]
        p.scatter(x, y, marker='s')
        p.scatter(res[i][0], 1013.25, c='brown')
        p.set_xlim(Tmin, Tmax)
        p.set_ylim(0, 1013.25)
        p.invert_yaxis()
        p.grid(True)
        if i > 0:  # Remove y-axis labels for all subplots except the first
            p.set_ylabel("") # removes the label


    #plt.tight_layout()
    plt.show()
