import numpy as np
import matplotlib.pyplot as plt
import time as tm
import utility

def main():

    # nx,ny: number of grid points in x and y direction
    # hx,hy: grid spacing in x and y direction
    # func: source term of the Poisson equation
    Lx , Ly = 1, 1
    nx = 63
    ny = nx

    #grid spacing (spatial)
    hx = Lx/nx
    hy = Ly/ny
    x = np.zeros((nx+1))
    y = np.zeros((ny+1))
    for i in range(nx+1):
        x[i] = i*hx
    for j in range(ny+1):
        y[j] = j*hy
      
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uExact = utility.exactSol(xv, yv)

    clock_time_init = tm.time()
    ft = FIST(x, y)
    phit = SolvePhit(ft, x, y, hx, hy)
    phi = FST(phit, x, y)
    u = sol(phi, x, y)
    total_clock_time = tm.time() - clock_time_init
    print('total_clock_time = ', total_clock_time)
    mse = (np.square(uExact - u)).mean(axis=None)
    print('mean square error = ', mse)

    utility.rmMargine()
    fileName='3DDST'
    title = 'DiscreteSineTransform'
    #utility.contourPlot(xv, yv, u, fileName='DST', figSize=(7, 7))
    #utility.contourPlot(xv, yv, u-uExact, fileName='errorDST', figSize=(7, 7))
    utility.contourPlot3D(xv, yv, u, fileName, title, figSize=(14,7))
    


def FIST(x, y):
    #1. fast inverse discrete sine transform of source term:
    ft = np.zeros((len(x)-1, len(y)-1))
    for k in range(1,len(x)-1):
        for l in range(1,len(y)-1):
            ft[k,l] = 0.0
            for i in range(1,len(x)-1):
                for j in range(1,len(y)-1):
                    ft[k,l] = ft[k,l] + utility.func2(x[i],y[j])*np.sin(np.pi*k*i/(len(x)-1))\
                        *np.sin(np.pi*l*j/(len(y)-1))
    #normalize    
    ft = ft*(2.0/(len(x)-1))*(2.0/(len(y)-1))
    return ft

def SolvePhit(ft, x, y, hx, hy):
    #2. Compute coefficient of solution three diagonal matrix to solve sine transform:
    phit = np.zeros((len(x), len(y)))
    ax = 2.0/(hx*hx)
    ay = 2.0/(hy*hy)
    for k in range(1,len(x)-1):
        for l in range(1,len(y)-1):
            phit[k][l] = ft[k][l] / (ax * (np.cos(np.pi*k/(len(x)-1)) - 1.0) + ay * (np.cos(np.pi*l/(len(y)-1)) - 1.0))
    return phit

def FST(phit, x, y):
    #3. fast forward fourier sine transform to find u:
    phi = np.zeros((len(x), len(y)))
    for i in range(1,len(x)-1):
        for j in range(1,len(y)-1):
            phi[i,j] = 0.0
            for k in range(1,len(x)-1):
                for l in range(1,len(y)-1):
                    phi[i,j] = phi[i,j] + phit[k][l]*np.sin(np.pi*k*i/(len(x)-1))\
                        *np.sin(np.pi*l*j/(len(y)-1))

    return phi

def sol(phi, x, y):
    u = np.zeros((len(x), len(y)))
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            u[i,j] = phi[i,j] - (x[i] - 1.0) * np.sin(2*np.pi*y[j])
    return u

if __name__ == "__main__":
    main()