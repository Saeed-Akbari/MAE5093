import numpy as np
import matplotlib.pyplot as plt
import time as tm
import utility

def main():

    # nx,ny: number of grid points in x and y direction
    # hx,hy: grid spacing in x and y direction
    # func: source term of the Poisson equation
    Lx , Ly = 1, 1
    nx = 255
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
    #utility.contourPlot(xv, yv, u, fileName='DSTFDM', figSize=(7, 7))
    #utility.contourPlot(xv, yv, u-uExact, fileName='errorDSTFDM', figSize=(7, 7))
    fileName='3DDSTFDM'
    title = 'DST & FDM'
    utility.contourPlot3D(xv, yv, u, fileName, title, figSize=(14,7))
    


def FIST(x, y):
    #1. fast inverse discrete sine transform of source term:
    ft = np.zeros((len(x)-1, len(y)-1))
    for k in range(1,len(x)-1):
        for j in range(1,len(y)-1):
            ft[k,j] = 0.0
            for i in range(1,len(x)-1):
                ft[k,j] = ft[k,j] + utility.func2(x[i],y[j])*np.sin(np.pi*k*i/(len(x)-1))
    #normalize    
    ft = ft*(2.0/(len(x)-1))
    return ft

def SolvePhit(ft, x, y, hx, hy):
    #2. Compute coefficient of solution three diagonal matrix to solve sine transform:
    phit = np.zeros((len(x), len(y)))
    a, b, c, d = map(np.ones, (len(y), len(y), len(y), len(y)))
    for k in range(1,len(x)-1):
        for j in range(1,len(y)-1):
            #a[j] = 1.0
            b[j] = 2 * (((hy**2) / (hx**2)) * (np.cos(np.pi*k/(len(x)-1)) - 1.0) - 1.0)
            #c[j] = 1.0
            d[j] = (hy**2) * ft[k,j]
        phit[k][1:-1] = tdma(a[1:-1],b[1:-1],c[1:-1],d[1:-1])

    return phit

def FST(phit, x, y):
    #3. fast forward fourier sine transform to find u:
    phi = np.zeros((len(x), len(y)))
    for i in range(1,len(x)-1):
        for j in range(1,len(y)-1):
            phi[i,j] = 0.0
            for k in range(1,len(x)-1):
                phi[i,j] = phi[i,j] + phit[k][j]*np.sin(np.pi*k*i/(len(x)-1))

    return phi

def sol(phi, x, y):
    u = np.zeros((len(x), len(y)))
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            u[i,j] = phi[i,j] - (x[i] - 1.0) * np.sin(2*np.pi*y[j])
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
    main()