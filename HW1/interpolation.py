import numpy as np

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

def lag(x, y, z):

    z = z.astype('float64')
    m = len(x)
    n = len(z)
    p = np.empty_like(z)
    for k in range(n):
        p[k] = 0
        for j in range(m):
            tmp = 1
            for i in range(m):
                if i != j:
                    tmp = tmp * (z[k] - x[i]) / (x[j] - x[i])
            p[k] = p[k] + y[j] * tmp
    return p

def gxxCalc(x, y):

    m = len(x)
    a, b, c, d = map(np.zeros, (m, m, m, m))
    b = np.ones((m))

    delta = x[1:m]-x[0:m-1]
    
    for i in range(1, m-1):
        a[i] = delta[i-1] / 6
        b[i] = (delta[i-1] + delta[i]) / 3
        c[i] = delta[i] / 6
        d[i] = (y[i+1] - y[i]) / delta[i] - (y[i] - y[i-1]) / delta[i-1]

    gxx = tdma(a,b,c,d)
    
    return gxx

def spline(x, y, z):

    inds = x.argsort()
    sx = x[inds]
    sy = y[inds]

    m = len(sx)
    for i in range(m):
        if sx[i] == z:
            return sy[i]
    delta = sx[1:m]-sx[0:m-1]

    gxx = gxxCalc(sx, sy)
    A = np.zeros((m, m))
    d = np.zeros((m, 1))
    
    idx1 = (np.abs(x - z)).argmin()

    if x[idx1]>z:
        idx2 = idx1 - 1
    else:
        idx2 = idx1 + 1
    if z > x[-1]:
        idx2 = idx1 - 1
    elif z < x[0]:
        idx2 = idx1 + 1

    i = min(idx1, idx2)

    A = (sx[i+1] - z) / delta[i]
    B = 1 - A
    C = (1.0 / 6.0) * (A**3 - A) * (delta[i]**2)
    D = (1.0 / 6.0) * (B**3 - B) * (delta[i]**2)

    g = A * sy[i] + B * sy[i+1] + C * gxx[i] + D * gxx[i+1]
    
    return g
