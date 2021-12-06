import numpy as np


#==============================================

#Area calculation
def AreaCal (x, i_Max):
    A = np.zeros((i_Max))
    dAdx = np.zeros((i_Max))
    for i in range(i_Max):
        A[i] =  1.398 + 0.347 * np.tanh( 0.8 * x[i] - 4.0 )
        dAdx[i] = 0.2766 * ( 1. - np.tanh( 0.8 * x[i] - 4.0 ) ** 2 )
    return A, dAdx

#==============================================


#==============================================

#Delta_x and Delta_y calculattion
def DeltaCal (l, i_Max ):
    dl = np.zeros((i_Max))
    for i in range(i_Max-1):
        dl[i] = l[i+1] - l[i]
    dl[i_Max-1] = dl[i_Max-2]
    return dl

#==============================================


#==============================================

#Energy calculation
def EnergyCal (rho, u, P, gamma):
    return ( ( P / ( gamma - 1 ) ) + 0.5 * rho * (u ** 2))

#==============================================


#==============================================

#Conservative vector calculation
def ConservCal (rho, u, E, Area, Eqns, i_Max):
    Q = np.zeros((Eqns, i_Max))
    Q[0] = rho * Area
    Q[1] = rho * u * Area
    Q[2] = E * Area
    return Q

#==============================================


#==============================================

#Primitive vector calculation
def PrimCal (Q, Area):
    rho = Q[0] / Area
    u = Q[1] / Q[0]
    E = Q[2] / Area
    return rho, u, E

def PrimCal2 (Q, Area, i):
    rho = Q[0][i] / Area[i]
    u = Q[1][i] / Q[0][i]
    E = Q[2][i] / Area[i]
    return rho, u, E

#==============================================


#==============================================

#Pressure calculation
def PressureCal(rho, u, E, gamma):
    return ((gamma - 1) * (E - 0.5 * rho * (u ** 2)))

#==============================================


#==============================================

def FluxSplitCal(rho, u, P, E, Area, dAdx, gamma, Eqns, i_Max):

    FP, FN, H = map(np.zeros, ((Eqns, i_Max), (Eqns, i_Max), (Eqns, i_Max)))  #Fluxes
    PP, PN, PuP, PuN, M, c = map(np.zeros_like, (u, u, u, u, u, u))

    for i in range(i_Max):
        c[i] = np.sqrt(np.fabs(gamma * P[i] / rho[i]))
        M[i] = u[i] / c[i]
        if np.fabs(M[i]) < 1 :
            PP[i] = 0.5 * ( 1 + M[i] ) * P[i]
            PN[i] = 0.5 * ( 1 - M[i] ) * P[i]
            PuP[i] = 0.5 * ( u[i] + c[i] ) * P[i]
            PuN[i] = 0.5 * ( u[i] - c[i] ) * P[i]
        elif M[i]<-1 :
            PP[i] = 0.0
            PN[i] = P[i]
            PuP[i] = 0.0
            PuN[i] = P[i] * u[i]
        elif M[i]>1 :
            PP[i] = P[i]
            PN[i] = 0.0
            PuP[i] = P[i] * u[i]
            PuN[i] = 0.0

    for i in range(i_Max):
        if u[i]>0 :
            FP[0][i] = rho[i] * u[i] * Area[i]
            FP[1][i] = ( rho[i] * u[i] * u[i] + PP[i] ) * Area[i]
            FP[2][i] = ( u[i] * E[i] + PuP[i] ) * Area[i]
        else:
            FP[0][i] = 0
            FP[1][i] = PP[i] * Area[i]
            FP[2][i] = PuP[i] * Area[i]

        if u[i]<0 :
            FN[0][i] = rho[i] * u[i] * Area[i]
            FN[1][i] = ( rho[i] * u[i] * u[i] + PN[i]) * Area[i]
            FN[2][i] = ( u[i] * E[i] + PuN[i] ) * Area[i]
        else:
            FN[0][i] = 0
            FN[1][i] = PN[i] * Area[i]
            FN[2][i] = PuN[i] * Area[i]

        H[0][i] = 0.0
        H[1][i] = P[i] * dAdx[i]
        H[2][i] = 0.0

    return FP, FN, H

#==============================================


#==============================================

#Time step calculation
def TimeStep (dx, rho, u, P, cfl, gamma, i_Max):

    cmax = 0.0
    c = np.sqrt(np.fabs((gamma * P) / rho)) #Sound Speed calculation
    for i in range(i_Max):
        if ( ( np.fabs ( u[i] ) + c[i] ) >  cmax):
            cmax = ( np.fabs ( u[i] ) + c[i] )
    return ( cfl * dx[1] / cmax )

#==============================================


#==============================================

def nuCal (P, i_Max):
    nu = np.zeros((i_Max))
    for i in range(i_Max):
            if (i == 0):
                nu[i] = np.fabs (P[i+2] - 2 * P[i+1] + P[i]) / (P[i+2] + 2 * P[i+1] + P[i])
            elif (i == i_Max - 1):
                nu[i] = nu[i-1]
            else:
                nu[i] = np.fabs (P[i+1] - 2 * P[i] + P[i-1]) / (P[i+1] + 2 * P[i] + P[i-1])
    return nu
#==============================================


#==============================================
#Dissipation terms calculation

def DissCal (Q, dx, nu, dt, Eqns, i_Max):

    D = np.zeros((Eqns, i_Max))
    k2 = 0.25
    k4 = 0.00390625

    #---------------------------------------------------------------------------------------------------------------

    for i in range(1, i_Max-1):
        for k in range(Eqns):
            eps2 = k2 * nu[i]
            if ( nu[i+1] > nu[i] ):
                eps2 = k2 * nu[i+1]
            eps4 = 0
            if ( ( k4 - eps2 ) > 0 ):
                eps4 = ( k4 - eps2 )

            if ( i < i_Max - 2 ):
                dR = ( 0.5 * ( dx[i] + dx[i+1] ) / dt ) * \
                ( eps2 * ( Q[k][i+1] - Q[k][i] ) - eps4 * ( Q[k][i+2] - 3 * Q[k][i+1] + 3 * Q[k][i] - Q[k][i-1] ) )
            elif ( i == i_Max - 2 ):
                dR = ( 0.5 * ( dx[i] + dx[i+1] ) / dt ) * \
                ( eps2 * ( Q[k][i+1] - Q[k][i] ) - eps4 * ( - Q[k][i-2] + 3 * Q[k][i-1] - 3 * Q[k][i] + Q[k][i+1] ) )

    #---------------------------------------------------------------------------------------------------------------

            eps2 = k2 * nu[i]
            if ( nu[i-1] > nu[i] ):
                eps2 = k2 * nu[i-1]
            eps4 = 0
            if ( ( k4 - eps2 ) > 0 ):
                eps4 = ( k4 - eps2 )

            if ( i > 1 ):
                dL = ( 0.5 * ( dx[i-1] + dx[i] ) / dt ) * \
                ( eps2 * ( Q[k][i] - Q[k][i-1] ) - eps4 * ( Q[k][i+1] - 3 * Q[k][i] + 3 * Q[k][i-1] - Q[k][i-2] ) )
            elif ( i == 1 ):
                dL = ( 0.5 * ( dx[i-1] + dx[i] ) / dt ) * \
                ( eps2 * ( Q[k][i] - Q[k][i-1]  ) - eps4 * ( - Q[k][i-1]  + 3 * Q[k][i] - 3 * Q[k][i+1] + Q[k][i+2] ) )

    #---------------------------------------------------------------------------------------------------------------

            D[k][i] = dR - dL

    return D

#==============================================



#==============================================
#Fluxes calculation

def FluxCal(rho, u, P, E, Area, dAdx, Eqns, i_Max):

    F, H = map(np.zeros, ((Eqns, i_Max), (Eqns, i_Max)))

    F[0] = rho * u * Area
    F[1] = (rho * u * u + P ) * Area
    F[2] = u * ( E + P ) * Area

    H[0] = 0.0
    H[1] = P * dAdx
    H[2] = 0.0
    
    return F, H

#===============================================