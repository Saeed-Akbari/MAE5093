import numpy as np
from visualization import *

def P4main():
    omegaList = np.arange(1, 1.96, 0.05)
    counter = np.zeros_like(omegaList)
    i = 0
    for omega in omegaList:
        counter[i] = problem4(omega)
        i += 1

    ind = np.argmin(counter)
    print('The omega corresponding to the minimum iteration is: {0:.2f}'.format(omegaList[ind]))
    print('The number of minimum iteration is: ', int(np.min(counter)))

    figNum = 1
    fileName = 'omega'
    plot1(figNum, fileName, omegaList, counter)

def problem4(omega = 1.):

    #omega = 1.
    x11, x22, x33 = 0, 1, 2
    y11, y22, y33 = 0, 0.5, 1
    nx1, nx2 = 10, 10
    ny2 = 6
    ny1 = ny2 * 2 - 1

    x1 = np.linspace(x11, x22, nx1+1)
    x2 = np.linspace(x22, x33, nx2+1)
    y1 = np.linspace(y11, y33, ny1)
    y2 = np.linspace(y11, y22, ny2)

    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    dy1 = y1[1] - y1[0]
    dy2 = y2[1] - y2[0]

    psi1 = np.zeros((len(x1), len(y1)))
    psi2 = np.zeros((len(x2), len(y2)))
    
    xv1, yv1 = np.meshgrid(x1, y1, indexing='ij')
    xv2, yv2 = np.meshgrid(x2, y2, indexing='ij')

    # boundary conditions
    j1 = 0
    k1 = -1
    for i in range(len(x1)):
        psi1[i][j1] = 0.
        psi1[i][k1] = 1.
    i1 = 0
    l1 = -1
    for j in range(len(y1)):
        psi1[i1][j] = y1[j]
        psi1[l1][j] = 1.
    psi1[l1][i1] = 0.

    j2 = 0
    k2 = -1
    for i in range(len(x2)):
        psi2[i][j2] = 0.
        psi2[i][k2] = 1.

    a1 = - 2. / (dx1**2) - 2. / (dy1**2)
    a2 = - 2. / (dx2**2) - 2. / (dy2**2)
    mse1, mse2 = 1, 1

    counter = 0
    while mse1+mse2 > 1e-10:
        psi_1 = np.copy(psi1)
        psi_2 = np.copy(psi2)
        for i in range(1, len(x1)-1):
            for j in range(1, len(y1)-1):
                psi1[i][j] = (1 - omega) * psi_1[i][j] + omega * (1.0)*(- (psi1[i+1][j]+psi1[i-1][j])/(dx1*dx1*a1) - (psi1[i][j+1]+psi1[i][j-1])/(dy1*dy1*a1) )
        
        i1 = -1
        i2 = 0
        for j in range(1, len(y2)-1):
            psi1[i1][j] = (1 - omega) * psi_1[i1][j] + omega * (1.0)*(- (psi2[i2+1][j]+psi1[i1-1][j])/(dx1*dx1*a1) - (psi1[i1][j+1]+psi1[i1][j-1])/(dy1*dy1*a1) )
            psi2[i2][j] = psi1[i1][j]

        for i in range(1, len(x2)-1):
            for j in range(1, len(y2)-1):
                psi2[i][j] = (1 - omega) * psi_2[i][j] + omega * (1.0)*(- (psi2[i+1][j]+psi2[i-1][j])/(dx2*dx2*a2) - (psi2[i][j+1]+psi2[i][j-1])/(dy2*dy2*a2) )
        i2 = -1
        for j in range(1, len(y2)-1):
            psi2[i2][j] = (1./11) * (18. * psi2[i2-1][j] - 9. * psi2[i2-2][j] + 2. * psi2[i2-3][j])
        mse1 = (np.square(psi_1 - psi1)).mean(axis=None)
        mse2 = (np.square(psi_2 - psi2)).mean(axis=None)
        counter += 1

    #fileName='p4'
    #title = 'stream function'
    #contourPlot2D(xv1, yv1, psi1, xv2, yv2, psi2, fileName, figSize=(14,7))

    return counter
if __name__ == "__main__":
    P4main()
