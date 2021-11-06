import numpy as np
import time as tm
import utility

def main():

    Lx , Ly = 1, 1
    nx = 127
    ny = nx
    hx = Lx/(nx+1)
    hy = Ly/(ny+1)
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    u = np.zeros((len(x), len(y)))
    uOld = np.zeros((len(x), len(y)))
    
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uExact = utility.exactSol(xv, yv)
 
    i = 0
    for j in range(len(y)):
        u[i][j] = np.sin(2*(np.pi)*y[j])
    
    a = -2.0/(hx*hx) - 2.0/(hy*hy)
    mse = 1
    
    clock_time_init = tm.time()
    while mse > 1e-10:
        uOld = np.copy(u)
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                #u[i][j] =  (0.5 / (hx**2 + hy**2)) * ((hy**2) * (u[i+1][j] + u[i-1][j]) + (hx**2) * (u[i][j+1] + u[i][j-1]) - (hx**2) * (hy**2) * func(x[j], y[i]))
                #u[i][j] =  0.25 * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - (hx**2) * func(x[j], y[i]))
                u[i,j] = (1.0/a)*(utility.func(x[j], y[i]) - (u[i+1][j]+u[i-1][j])/(hx*hx) - (u[i][j+1]+u[i][j-1])/(hy*hy) )
        mse = (np.square(uOld - u)).mean(axis=None)
    total_clock_time = tm.time() - clock_time_init
    print('total_clock_time = ', total_clock_time)
    mse = (np.square(uExact - u)).mean(axis=None)
    print('mean square error = ', mse)

    utility.rmMargine()
    fileName='3DGS'
    title = 'Finite Difference Method'
    #utility.contourPlot3D(xv, yv, u, fileName, title, figSize=(14,7))
    fileName='3DExact'
    title = 'Exact Solution'
    utility.contourPlot3D(xv, yv, uExact, fileName, title, figSize=(14,7))


if __name__ == "__main__":
    main()