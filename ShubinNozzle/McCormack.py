import numpy as np

from utility import TimeStep
from methods import McCormack

def mainMcCormack(i_Max, Eqns, gamma, cfl, Area, dAdx, dx, rho, u, P, E, Q):
    
    #---------------------------------------------------------------------------------------------------------------

    #Main loop

    time = 0.0
    iteration = 3000
    gg = 0
    while gg < iteration :
        dt = TimeStep( dx, rho, u, P, cfl, gamma, i_Max)
        time += dt
        gg += 1

        Q, rho, u, P, E = McCormack (Q, rho, u, P, E, dx, Area, dAdx, gamma, dt, Eqns, i_Max)
    
    #---------------------------------------------------------------------------------------------------------------

    c = np.sqrt(np.fabs((gamma * P) / rho)) #Sound Speed calculation

    return rho, u, E, u/c
    
if __name__ == '__main__':
    mainMcCormack()