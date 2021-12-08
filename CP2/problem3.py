import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from visualization import *

def problem3():

    alpha = 1 / np.pi**2
    xl , xr = -1., 1.
    Lx = xr - xl
    dx = 0.025
    dt = 0.0025
    tmp = 'c2'
    tmax = 2.0
    r = alpha * dt / dx**2

    nx = int(Lx / dx) - 1
    nt = int(tmax / dt) - 1
    x = np.linspace(0, Lx, nx+1)
    t = np.linspace(0, tmax, nt+1)
    u = np.zeros((len(t), len(x)))
    u[0]  = - np.sin(np.pi*x)

    uFT, uBT, uCN, uICP, uExact = map(np.copy, (u, u, u, u, u))
    uFTtmp, uBTtmp, uCNtmp, uICPtmp = map(np.copy, (u[0], u[0], u[0], u[0]))
    for n in range(1, len(t)):
        uExact[n] = exactSol(x, t[n], alpha)
        uFT[n] = FTCS(uFTtmp, r)
        uBT[n] = BTCS(uBTtmp, r)
        uCN[n] = CN(uCNtmp, r)
        uICP[n] = ICP(uICPtmp, r)
        uFTtmp = uFT[n]
        uBTtmp = uBT[n]
        uCNtmp = uCN[n]
        uICPtmp = uICP[n]

    errFT = np.abs(uFT-uExact)
    errBT = np.abs(uBT-uExact)
    errCN = np.abs(uCN-uExact)
    errICP = np.abs(uICP-uExact)

    fileName = 'p3' + tmp
    fileNameErr = 'p3error' + tmp
    plotTitle = 't = 1s ($\Delta t={}$, $\Delta x={}$)'.format(dt, dx)
    plotTitleErr = 'Error at t = 1s ($\Delta t={}$, $\Delta x={}$)'.format(dt, dx)
    label1 = 'FTCS'
    label2 = 'BTCS'
    label3 = 'Crank Nicolson'
    label4 = 'ICP'
    label5 = 'Exact Solution'
    ind = 70
    subplot(1, fileName, plotTitle, label1, label2, label3, label4, label5, x, uFT[ind], uBT[ind], uCN[ind], uICP[ind], uExact[ind])
    plotErr(2, fileNameErr, plotTitleErr, label1, label2, label3, label4, x[1:-1], errFT[ind][1:-1], errBT[ind][1:-1], errCN[ind][1:-1], errICP[ind][1:-1])

def exactSol(x,t, alpha):
    return (-np.sin(np.pi*x)*np.exp(-alpha*t))

def FTCS(u, r):
    for i in range(1, len(u)-1):
        u[i] = u[i] + r * (u[i+1] - 2 * u[i] + u[i-1])
    return u

def BTCS(u, r):
    m = len(u)
    a, c, d = map(np.zeros, (m, m, m))
    b = np.ones((m))
    for i in range(1, m-1):
        a[i] = -r
        b[i] = 1. + 2 * r
        c[i] = -r
        d[i] = u[i]
    u = tdma(a,b,c,d)
    return u

def CN(u, r):
    m = len(u)
    a, c, d = map(np.zeros, (m, m, m))
    b = np.ones((m))
    for i in range(1, m-1):
        a[i] = -0.5*r
        b[i] = 1. + r
        c[i] = -0.5*r
        d[i] = 0.5*r*u[i+1] + (1 - r) * u[i] + 0.5*r*u[i-1]
    u = tdma(a,b,c,d)
    return u

def ICP(u, r):
    m = len(u)
    a, c, d = map(np.zeros, (m, m, m))
    b = np.ones((m))
    for i in range(1, m-1):
        a[i] = 12. * r - 2.
        b[i] = -24. * r - 20.
        c[i] = 12. * r - 2.
        d[i] = - 2. * (u[i+1] + 10. * u[i] + u[i-1]) - 12. * r * (u[i+1] - 2. * u[i] + u[i-1])
    u = tdma(a,b,c,d)
    return u

def tdma(a,b,c,d):
    
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    ne = len(d)
    for i in range(1,ne):
        bc[i] = bc[i] - ac[i]*(cc[i-1]/bc[i-1])
        dc[i] = dc[i] - ac[i]*(dc[i-1]/bc[i-1])
    uc = bc    
    uc[-1] = dc[-1]/bc[-1]
    
    for i in range(ne-2,-1,-1):
        uc[i] = (dc[i] - cc[i]*uc[i+1])/bc[i]
    
    del ac, bc, cc, dc

    return uc

if __name__ == "__main__":
    problem3()