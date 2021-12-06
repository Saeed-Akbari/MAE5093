import numpy as np
from utility import EnergyCal, ConservCal, PrimCal, PrimCal2, PressureCal, FluxSplitCal, nuCal, DissCal, FluxCal

#==============================================

def Upwind (Q, rho, u, P, E, dx, Area, dAdx, gamma, dt, Eqns, i_Max):

    rho, u, E = PrimCal(Q, Area)
    P = PressureCal(rho, u, E, gamma)

    #SuperSonic inlet        
    i = 0
    rho[0] = 0.5008261
    u[i] = 1.0991840
    P[i] = 0.2712900
    E[i] = EnergyCal (rho[i] ,u[i] ,P[i] ,gamma)

    #SubSonic outlet
    i = i_Max-1
    rho[i] = 2 * rho[i-1] - rho[i-2]
    u[i] = 2 * u[i-1] - u[i-2]
    P[i] = 0.5156000
    E[i] = EnergyCal (rho[i] ,u[i] ,P[i] ,gamma)
    
    Q = ConservCal (rho, u, E, Area, Eqns, i_Max)
    FP, FN, H = FluxSplitCal(rho, u, P, E, Area, dAdx, gamma, Eqns, i_Max)

    for k in range(Eqns):
        for i in range(1, i_Max-1):
            Q[k][i] = Q[k][i] - ( dt / dx[i] ) * ( FP[k][i] - FP[k][i-1] + FN[k][i+1] - FN[k][i] ) + dt * H[k][i]
    return Q, rho, u, P, E
    
#==============================================


#==============================================

def McCormack (Q, rho, u, P, E, dx, Area, dAdx, gamma, dt, Eqns, i_Max):

    Q1, Q2 = map(np.zeros, ((Eqns, i_Max), (Eqns, i_Max)))
    nu = np.zeros((i_Max))
    eps = 0.035
    
    rho, u, E = PrimCal(Q, Area)
    P = PressureCal(rho, u, E, gamma)

    #SuperSonic inlet        
    i = 0
    rho[0] = 0.5008261
    u[i] = 1.0991840
    P[i] = 0.2712900
    E[i] = EnergyCal (rho[i] ,u[i] ,P[i] ,gamma)

    #SubSonic outlet
    i = i_Max-1
    rho[i] = 2 * rho[i-1] - rho[i-2]
    u[i] = 2 * u[i-1] - u[i-2]
    P[i] = 0.5156000
    E[i] = EnergyCal (rho[i] ,u[i] ,P[i] ,gamma)

    Q = ConservCal (rho, u, E, Area, Eqns, i_Max)
    F, H = FluxCal(rho, u, P, E, Area, dAdx, Eqns, i_Max)
    
    nu = nuCal (P, i_Max)
    D = DissCal (Q, dx, nu, dt, Eqns, i_Max)
    
    for k in range(Eqns):
        for i in range(1, i_Max-1):
            ArtVisc = 0
            if (i > 1 and i < i_Max - 2):
                  ArtVisc = eps * ( Q[k][i-1] - 2 * Q[k][i] + Q[k][i+1] )

            #Q1[k][i] = Q[k][i] - ( dt / dx[i] ) * ( F[k][i+1] - F[k][i] - ArtVisc ) + 0.5 * dt * ( H[k][i] + H[k][i-1] )
            Q1[k][i] = Q[k][i] - ( dt / dx[i] ) * ( F[k][i] - F[k][i-1] - D[k][i] ) + 0.5 * dt * ( H[k][i] + H[k][i-1] )
            #Q1[k][i] = Q[k][i] - ( dt / dx[i] ) * ( F[k][i] - F[k][i-1] - ArtVisc ) + 0.5 * dt * ( H[k][i] + H[k][i-1] )
    
    for i in range(1, i_Max-1):
        rho[i], u[i], E[i] = PrimCal2(Q1, Area, i)
        P[i] = PressureCal(rho[i], u[i], E[i], gamma)

    F, H = FluxCal(rho, u, P, E, Area, dAdx, Eqns, i_Max)

    for k in range(Eqns):
        for i in range(1, i_Max-1):
            ArtVisc = 0
            if (i > 1 and i < i_Max - 1):
                ArtVisc = eps * ( Q1[k][i-1] - 2 * Q1[k][i] + Q1[k][i+1] )

            #Q2[k][i] = Q[k][i] - ( dt / dx[i] ) * ( F[k][i] - F[k][i-1] - ArtVisc ) + 0.5 * dt * ( H[k][i] + H[k][i+1] )
            Q2[k][i] = Q[k][i] - ( dt / dx[i] ) * ( F[k][i+1] - F[k][i] - D[k][i] ) + 0.5 * dt * ( H[k][i] + H[k][i+1] )
            #Q2[k][i] = Q[k][i] - ( dt / dx[i] ) * ( F[k][i+1] - F[k][i] - ArtVisc ) + 0.5 * dt * ( H[k][i] + H[k][i+1] )


    for k in range(Eqns):
        for i in range(1, i_Max-1):
            Q[k][i] = 0.5 * ( Q1[k][i] + Q2[k][i] )

    return Q, rho, u, P, E

#==============================================