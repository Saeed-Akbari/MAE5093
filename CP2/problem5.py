import numpy as np
import time as tm
from visualization import *
from poisson_solver import cg, mg_n_solver

def main():

    omega = 1.
    Lx , Ly = 2., 2.
    nx = 255
    print('mesh = ', nx+1)
    ny = nx
    hx = Lx/(nx+1)
    hy = Ly/(ny+1)
    x = np.linspace(-1, -1+Lx, nx+1)
    y = np.linspace(-1, -1+Ly, ny+1)
    
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uExact = exactSol(xv, yv)
    fRHS = func(xv, yv)
    
    clock_time_init = tm.time()
    uGS = GS(x, y, hx, hy, omega)
    total_clock_time_GS = tm.time() - clock_time_init
    mse = (np.square(uExact - uGS)).mean(axis=None)
    print('mean square error of GS = ', mse)
    print('=====================================')

    clock_time_init = tm.time()
    uCG, _ = cg(nx,ny,hx,hy,fRHS,1)
    total_clock_time_CG = tm.time() - clock_time_init
    mse = (np.square(uExact - uCG)).mean(axis=None)
    print('mean square error of CG = ', mse)
    print('=====================================')

    clock_time_init = tm.time()
    n_level = 2
    uMG2, _ = mg_n_solver(fRHS, hx, hy, nx, ny, n_level)
    total_clock_time_MG2 = tm.time() - clock_time_init
    mse = (np.square(uExact - uMG2)).mean(axis=None)
    print('mean square error of MG2 = ', mse)
    print('=====================================')

    clock_time_init = tm.time()
    n_level = 4
    uMG4, _ = mg_n_solver(fRHS, hx, hy, nx, ny, n_level)
    total_clock_time_MG4 = tm.time() - clock_time_init
    mse = (np.square(uExact - uMG4)).mean(axis=None)
    print('mean square error of MG4 = ', mse)
    print('=====================================')

    print('=====================================')
    print('total clock time for GS = ', total_clock_time_GS)
    print('=====================================')
    print('total clock time for CG = ', total_clock_time_CG)
    print('=====================================')
    print('total clock time for MG2 = ', total_clock_time_MG2)
    print('=====================================')
    print('total clock time for MG4 = ', total_clock_time_MG4)
    print('=====================================')
    
def GS(x, y, hx, hy, omega):

    u = np.zeros((len(x), len(y)))
    a = -2.0/(hx*hx) - 2.0/(hy*hy)
    mse = 1
    while mse > 1e-10:
        uOld = np.copy(u)
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                u[i,j] = (1 - omega) * uOld[i,j] + omega * (1.0/a)*(func(x[j], y[i]) - (u[i+1][j]+u[i-1][j])/(hx*hx) - (u[i][j+1]+u[i][j-1])/(hy*hy) )
        mse = (np.square(uOld - u)).mean(axis=None)
    return u

def func(x, y):
    return (- 2. * (2. - x**2 - y**2))

def exactSol(x, y):
    return ((x**2 - 1.) * (y**2 - 1.))

if __name__ == "__main__":
    main()
