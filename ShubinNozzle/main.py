import numpy as np
import matplotlib.pyplot as plt

from utility import AreaCal, DeltaCal, EnergyCal, ConservCal

from upwind import mainUpwind
from McCormack import mainMcCormack

def main():

    #---------------------------------------------------------------------------------------------------------------

    #Geometry and Grid (Nozzle)
    i_Max = 101

    #Grid generation
    x = np.zeros((i_Max))
    for i in range(i_Max):
        x[i] = i * 0.1

    #Area and delta_x and delta_y calculation
    Area, dAdx = AreaCal(x, i_Max)
    dx = DeltaCal(x , i_Max)

    #---------------------------------------------------------------------------------------------------------------

    Eqns = 3    #Number of equations
    gamma = 1.4
    cfl = 0.9

    #---------------------------------------------------------------------------------------------------------------

    #Physical Variables Initialize
    rho = np.ones((i_Max)) * 0.7511383
    u = np.ones((i_Max)) * 0.4416178
    P = np.ones((i_Max)) * 0.2156000
    E = EnergyCal (rho ,u ,P ,gamma)

    #Conservative Variables Initialization
    Q = ConservCal (rho, u, E, Area, Eqns, i_Max)

    #---------------------------------------------------------------------------------------------------------------

    rhoUp, uUp, EUp, MUp = map(np.zeros, ((i_Max), (i_Max), (i_Max), (i_Max)))
    rhoMc, uMc, EMc, MMc = map(np.zeros, ((i_Max), (i_Max), (i_Max), (i_Max)))

    rhoUp, uUp, EUp, MUp = mainUpwind(i_Max, Eqns, gamma, cfl, Area, dAdx, dx, rho, u, P, E, Q)
    rhoMc, uMc, EMc, MMc = mainMcCormack(i_Max, Eqns, gamma, cfl, Area, dAdx, dx, rho, u, P, E, Q)
    
    figNum = 1
    fileName = 'density'
    plotTitle = 'Density, CFL=' + str(cfl)
    label1 = 'Upwind'
    label2 = 'McCormack'
    ylabel = 'density (kg.m^-3)'
    plot(figNum, fileName, plotTitle, ylabel, label1, label2, x, rhoUp, rhoMc)

    figNum = 2
    fileName = 'velocity'
    plotTitle = 'Velocity, CFL=' + str(cfl)
    ylabel = 'velocity (m.s^-1)'
    plot(figNum, fileName, plotTitle, ylabel, label1, label2, x, uUp, uMc)

    figNum = 3
    fileName = 'mach'
    plotTitle = 'Mach, CFL=' + str(cfl)
    ylabel = 'mach (m.s^-1)'
    plot(figNum, fileName, plotTitle, ylabel, label1, label2, x, MUp, MMc)

def plot(figNum, fileName, plotTitle, ylabel, label1, label2, x, phi1, phi2):

    plt.figure(figNum)
    plt.plot(x, phi1, 'b', label=label1)
    plt.plot(x, phi2, 'r', label=label2)
    plt.title(plotTitle)
    plt.xlabel('Length')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(fileName, dpi = 200)

if __name__ == '__main__':
    main()
