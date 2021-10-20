import numpy as np
import matplotlib.pyplot as plt

def main():

    x0, y0, z0 = 1.0, 1.0, 1.0

    tl = 0
    tr = 25
    h = 0.01
    t = np.arange(tl, tr, h)
    lent = len(t)
    x = np.zeros((lent))
    x, y, z = map(np.zeros, ((lent), (lent), (lent)))
    x[0], y[0], z[0] = x0, y0, z0

    sigma = 10.0
    b = 8/3
    r = 20.0
    x20, y20, z20 = lorenzSolver(x, y, z, lent, h, sigma, b, r)

    r = 28.0
    x28, y28, z28 = lorenzSolver(x, y, z, lent, h, sigma, b, r)


    fig = plt.figure(1)
    plt.title('Lorenz r=20')
    plt.plot(x20, y20, label='y')
    plt.plot(x20, z20, label='z')
    plt.legend()
    plt.savefig('onX20')

    fig = plt.figure(2)
    plt.title('Lorenz r=20')
    plt.plot(y20, x20, label='x')
    plt.plot(y20, z20, label='z')
    plt.legend()
    plt.savefig('onY20')

    fig = plt.figure(3)
    plt.title('Lorenz r=20')
    plt.plot(z20, x20, label='x')
    plt.plot(z20, y20, label='y')
    plt.legend()
    plt.savefig('onZ20')

    fig = plt.figure(5)
    plt.title('Lorenz r=20')
    plt.plot(t, x20, label='x')
    plt.plot(t, y20, label='y')
    plt.plot(t, z20, label='z')
    plt.legend()
    plt.savefig('onTime20')

    fig = plt.figure(6)
    ax = plt.axes(projection='3d')
    ax.plot3D(x20, y20, z20)
    plt.title('Lorenz r=20')
    #ax.legend()
    plt.savefig('3D20')


    fig = plt.figure(11)
    plt.title('Lorenz r=28')
    plt.plot(x28, y28, label='y')
    plt.plot(x28, z28, label='z')
    plt.legend()
    plt.savefig('onX28')

    fig = plt.figure(12)
    plt.title('Lorenz r=28')
    plt.plot(y28, x28, label='x')
    plt.plot(y28, z28, label='z')
    plt.legend()
    plt.savefig('onY28')

    fig = plt.figure(13)
    plt.title('Lorenz r=28')
    plt.plot(z28, x28, label='x')
    plt.plot(z28, y28, label='y')
    plt.legend()
    plt.savefig('onZ28')

    fig = plt.figure(15)
    plt.title('Lorenz r=28')
    plt.plot(t, x28, label='x')
    plt.plot(t, y28, label='y')
    plt.plot(t, z28, label='z')
    plt.legend()
    plt.savefig('onTime28')

    fig = plt.figure(16)
    ax = plt.axes(projection='3d')
    ax.plot3D(x28, y28, z28)
    plt.title('Lorenz r=28')
    #ax.legend()
    plt.savefig('3D28')

    r = 28.0
    x0, y0, z0 = 6.0, 6.0, 6.0
    x = np.zeros((lent))
    x, y, z = map(np.zeros, ((lent), (lent), (lent)))
    x[0], y[0], z[0] = x0, y0, z0
    x1, y1, z1 = lorenzSolver(x, y, z, lent, h, sigma, b, r)

    x = np.zeros((lent))
    x, y, z = map(np.zeros, ((lent), (lent), (lent)))
    x[0], y[0], z[0] = x0, y0, z0
    x0, y0, z0 = 6.0, 6.01, 6.0
    x2, y2, z2 = lorenzSolver(x, y, z, lent, h, sigma, b, r)

    fig = plt.figure(21)
    plt.title('Lorenz r=28')
    plt.plot(x1, y1, label='y 6')
    plt.plot(x1, z1, label='z 6')
    plt.plot(x2, y2, label='y 6.01')
    plt.plot(x2, z2, label='z 6.01')
    plt.legend()
    plt.savefig('onXinit')

    fig = plt.figure(22)
    plt.title('Lorenz r=28')
    plt.plot(y1, x1, label='x 6')
    plt.plot(y1, z1, label='z 6')
    plt.plot(y2, x2, label='x 6.01')
    plt.plot(y2, z2, label='z 6.01')
    plt.legend()
    plt.savefig('onYinit')

    fig = plt.figure(23)
    plt.title('Lorenz r=28')
    plt.plot(z1, x1, label='x 6')
    plt.plot(z1, y1, label='y 6')
    plt.plot(z2, x2, label='x 6.01')
    plt.plot(z2, y2, label='y 6.01')
    plt.legend()
    plt.savefig('onZinit')

    fig = plt.figure(25)
    plt.title('Lorenz r=28')
    plt.plot(t, x1, label='x 6')
    plt.plot(t, y1, label='y 6')
    plt.plot(t, z1, label='z 6')
    plt.plot(t, x2, label='x 6.01')
    plt.plot(t, y2, label='y 6.01')
    plt.plot(t, z2, label='z 6.01')
    plt.legend()
    plt.savefig('onTimeinit')

    fig = plt.figure(26)
    plt.title('Lorenz r=28')
    ax = plt.axes(projection='3d')
    ax.plot3D(x1, y1, z1, label='6')
    ax.plot3D(x2, y2, z2, label='6.01')
    ax.legend()
    plt.savefig('3Dinit')

def lorenzSolver(x, y, z, lent, h, sigma, b, r):
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

if __name__ == "__main__":
    main()