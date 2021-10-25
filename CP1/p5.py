import numpy as np
import matplotlib.pyplot as plt

def main():

    x0, y0, z0 = 1.0, 1.0, 1.0

    tl = 0
    tr = 100.001
    sigma = 10.0
    b = 8/3
    r = 28.0
    
    h = 0.001
    t = np.arange(tl, tr, h)
    lent = len(t)
    x, y, z = map(np.zeros, ((lent), (lent), (lent)))
    x[0], y[0], z[0] = x0, y0, z0
    xRK4, yRK4, zRK4 = lorenzRK4(x, y, z, lent, h, sigma, b, r)

    fig = plt.figure(11)
    plt.title('Lorenz r=28')
    plt.plot(t, xRK4)
    plt.ylabel('x')
    plt.savefig('xt')
    
    fig = plt.figure(12)
    plt.title('Lorenz r=28')
    plt.plot(t, yRK4)
    plt.ylabel('y')
    plt.savefig('yt')

    fig = plt.figure(15)
    plt.title('Lorenz r=28')
    plt.plot(t, zRK4)
    plt.savefig('zt')

    fig = plt.figure(16)
    ax = plt.axes(projection='3d')
    ax.plot3D(xRK4, yRK4, zRK4)
    plt.title('Lorenz r=28')
    #ax.legend()
    plt.savefig('3D28')
    ''''''
    
    h = 0.001
    t = np.arange(tl, tr, h)
    lent = len(t)
    
    xRK4, yRK4, zRK4 = map(np.zeros, ((lent), (lent), (lent)))
    xRK4[0], yRK4[0], zRK4[0] = x0, y0, z0
    xRK4, yRK4, zRK4 = lorenzRK4(xRK4, yRK4, zRK4, lent, h, sigma, b, r)
    xEu, yEu, zEu = map(np.zeros, ((lent), (lent), (lent)))
    xEu[0], yEu[0], zEu[0] = x0, y0, z0
    xEu, yEu, zEu = lorenzEuler(xEu, yEu, zEu, lent, h, sigma, b, r)
    xRK2, yRK2, zRK2 = map(np.zeros, ((lent), (lent), (lent)))
    xRK2[0], yRK2[0], zRK2[0] = x0, y0, z0
    xRK2, yRK2, zRK2 = lorenzRK2(xRK2, yRK2, zRK2, lent, h, sigma, b, r)
    #xLF, yLF, zLF = map(np.zeros, ((lent), (lent), (lent)))
    #xLF[0], yLF[0], zLF[0] = x0, y0, z0
    #xLF, yLF, zLF = lorenzLeapfrog(xLF, yLF, zLF, lent, h, sigma, b, r)

    errEuX = np.abs((xRK4[-1]-xEu[-1])/xRK4[-1])
    errEuY = np.abs((yRK4[-1]-yEu[-1])/yRK4[-1])
    errEuZ = np.abs((zRK4[-1]-zEu[-1])/zRK4[-1])
    print('Error for Euler method at {}s for h={} is: '.format(int(t[-1]), h), (errEuX, errEuY, errEuZ))
    errRK2X = np.abs((xRK4[-1]-xRK2[-1])/xRK4[-1])
    errRK2Y = np.abs((yRK4[-1]-yRK2[-1])/yRK4[-1])
    errRK2Z = np.abs((zRK4[-1]-zRK2[-1])/zRK4[-1])
    print('Error for second-order Runge Kutta method at {}s for h={} is: '.format(int(t[-1]), h), (errRK2X, errRK2Y, errRK2Z))
    #errLF = np.abs((xRK4[-1]-xLF[-1])/xRK4[-1])
    #print('Error for Eulter method at {}s for h={} is: '.format(int(t[-1]), h), errLF)

    
    h = 0.1
    t = np.arange(tl, tr, h)
    lent = len(t)

    xRK4, yRK4, zRK4 = map(np.zeros, ((lent), (lent), (lent)))
    xRK4[0], yRK4[0], zRK4[0] = x0, y0, z0
    xRK4, yRK4, zRK4 = lorenzRK4(xRK4, yRK4, zRK4, lent, h, sigma, b, r)
    xEu, yEu, zEu = map(np.zeros, ((lent), (lent), (lent)))
    xEu[0], yEu[0], zEu[0] = x0, y0, z0
    xEu, yEu, zEu = lorenzEuler(xEu, yEu, zEu, lent, h, sigma, b, r)
    xRK2, yRK2, zRK2 = map(np.zeros, ((lent), (lent), (lent)))
    xRK2[0], yRK2[0], zRK2[0] = x0, y0, z0
    xRK2, yRK2, zRK2 = lorenzRK2(xRK2, yRK2, zRK2, lent, h, sigma, b, r)
    #x, y, z = map(np.zeros, ((lent), (lent), (lent)))
    #x[0], y[0], z[0] = x0, y0, z0
    #xLF, yLF, zLF = lorenzLeapfrog(x, y, z, lent, h, sigma, b, r)

    errEuX = np.abs((xRK4[-1]-xEu[-1])/xRK4[-1])
    errEuY = np.abs((yRK4[-1]-yEu[-1])/yRK4[-1])
    errEuZ = np.abs((zRK4[-1]-zEu[-1])/zRK4[-1])
    print('Error for Euler method at {}s for h={} is: '.format(int(t[-1]), h), (errEuX, errEuY, errEuZ))
    errRK2X = np.abs((xRK4[-1]-xRK2[-1])/xRK4[-1])
    errRK2Y = np.abs((yRK4[-1]-yRK2[-1])/yRK4[-1])
    errRK2Z = np.abs((zRK4[-1]-zRK2[-1])/zRK4[-1])
    print('Error for second-order Runge Kutta method at {}s for h={} is: '.format(int(t[-1]), h), (errRK2X, errRK2Y, errRK2Z))
    #errLF = np.abs((xRK4[-1]-xLF[-1])/xRK4[-1])
    #print('Error for Eulter method at {}s for h {} = '.format(int(t[-1]), h), errLF)
    

def lorenzRK4(x, y, z, lent, h, sigma, b, r):
    '''     fourth-order Runge–Kutta       '''
    cte = 0.166666666666666667
    for i in range(lent-1):
        xt, yt, zt = x[i], y[i], z[i]
        fx = sigma * (yt-xt)
        fy = r * xt - yt - xt * zt
        fz = xt * yt - b * zt
        k1x = h * fx
        k1y = h * fy
        k1z = h * fz
        xt, yt, zt = x[i]+0.5*k1x, y[i]+0.5*k1y, z[i]+0.5*k1z
        fx = sigma * (yt-xt)
        fy = r * xt - yt - xt * zt
        fz = xt * yt - b * zt
        k2x = h * fx
        k2y = h * fy
        k2z = h * fz
        xt, yt, zt = x[i]+0.5*k2x, y[i]+0.5*k2y, z[i]+0.5*k2z
        fx = sigma * (yt-xt)
        fy = r * xt - yt - xt * zt
        fz = xt * yt - b * zt
        k3x = h * fx
        k3y = h * fy
        k3z = h * fz
        xt, yt, zt = x[i]+k3x, y[i]+k3y, z[i]+k3z
        fx = sigma * (yt-xt)
        fy = r * xt - yt - xt * zt
        fz = xt * yt - b * zt
        k4x = h * fx
        k4y = h * fy
        k4z = h * fz
        x[i+1] = x[i] + cte*k1x + 2*cte*(k2x+k3x) + cte*k4x
        y[i+1] = y[i] + cte*k1y + 2*cte*(k2y+k3y) + cte*k4y
        z[i+1] = z[i] + cte*k1z + 2*cte*(k2z+k3z) + cte*k4z

    return x, y, z


def lorenzEuler(x, y, z, lent, h, sigma, b, r):
    for i in range(lent-1):
        x[i+1] = h * fx(sigma, x[i], y[i]) + x[i]
        y[i+1] = h * fy(r, x[i], y[i], z[i]) + y[i]
        z[i+1] = h * fz(b, x[i], y[i], z[i]) + z[i]
    return x, y, z

def lorenzEulerInit(x, y, z, h, sigma, b, r):
    i = 0
    x[i+1] = h * fx(sigma, x[i], y[i]) + x[i]
    y[i+1] = h * fy(r, x[i], y[i], z[i]) + y[i]
    z[i+1] = h * fz(b, x[i], y[i], z[i]) + z[i]
    return x, y, z

def lorenzLeapfrog(x, y, z, lent, h, sigma, b, r):
    x, y, z = lorenzEulerInit(x, y, z, h, sigma, b, r)
    for i in range(1, lent-1):
        x[i+1] = 2 * h * fx(sigma, x[i], y[i]) + x[i-1]
        y[i+1] = 2 * h * fy(r, x[i], y[i], z[i]) + y[i-1]
        z[i+1] = 2 * h * fz(b, x[i], y[i], z[i]) + z[i-1]
    return x, y, z


def lorenzRK2(x, y, z, lent, h, sigma, b, r):
    '''     second-order Runge–Kutta       '''
    beta = 0.5
    gamma2 = 0.5 / beta
    gamma1 = 1 - gamma2
    for i in range(lent-1):
        xt, yt, zt = x[i], y[i], z[i]
        fx = sigma * (yt-xt)
        fy = r * xt - yt - xt * zt
        fz = xt * yt - b * zt
        k1x = h * fx
        k1y = h * fy
        k1z = h * fz
        xt, yt, zt = x[i]+beta*k1x, y[i]+beta*k1y, z[i]+beta*k1z
        fx = sigma * (yt-xt)
        fy = r * xt - yt - xt * zt
        fz = xt * yt - b * zt
        k2x = h * fx
        k2y = h * fy
        k2z = h * fz
        x[i+1] = x[i] + gamma1*k1x + gamma2*k2x
        y[i+1] = y[i] + gamma1*k1y + gamma2*k2y
        z[i+1] = z[i] + gamma1*k1z + gamma2*k2z

    return x, y, z


def fx(sigma, xt, yt):
    return sigma * (yt-xt)

def fy(r, xt, yt, zt):
    return r * xt - yt - xt * zt

def fz(b, xt, yt, zt):
    return xt * yt - b * zt

if __name__ == "__main__":
    main()
