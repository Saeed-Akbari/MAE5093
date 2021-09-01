import numpy as np
import matplotlib.pyplot as plt

from interpolation import lag, spline

def main():
    
#%% Q1
    
    x = np.arange(-1, 1.01, 0.2)
    y = 1./(1+25*(x**2))
    q = [0.7]
    z = np.array(q)
    p = lag(x, y, z)
    print('===============================================')
    print('problem 1:')
    print('verifying that P(0.7) = −0.226:      P(0.7) = ', p)

    # (a)
    print('the interpolated value at x = 0.9:      P(0.9) = ', lag(x, y, np.array([0.9])), '\n')
    # (b)
    xn = np.linspace(-1, 1, 21)
    fig = plt.figure(1)
    plt.plot(x, y, 'ko', label='data')
    plt.plot(x, lag(x, y, x), 'b', label='Lagrange 11 points')
    plt.plot(xn, lag(x, y, xn), 'g', label='Lagrange 21 points')
    plt.title( 'Lagrange Polynomial')
    plt.legend(loc=9)
    fig.savefig('problem1')
    ''''''
#%% Q7
    y = np.array([21.300, 23.057, 24.441, 25.917, 27.204, 28.564, 29.847,\
                    31.200, 32.994, 34.800, 36.030])
    x = np.arange(1998, 2009)
    z = np.array([2010])
    print('===============================================')
    print('problem 2:')
    print('predict the tuition in 2010 with Lagrange interpolation:     ', lag(x, y, z))
    print('predict the tuition in 2010 with Spline:     ', [spline(x, y, z[0])], '\n')

    z = np.linspace(1998, 2009, len(x)*2)
    g = np.zeros(len(z))
    for i in range(len(z)):
        g[i] = spline(x, y, z[i])
    
    fig = plt.figure(2)
    plt.plot(x, y, 'o', label='data')
    plt.plot(z, lag(x, y, z), label='Lagrange')
    plt.plot(z, g, label='spline')
    plt.title( 'Tuition per year')
    plt.legend(loc=9)
    fig.savefig('problem7')
    
#%% Q8
    y = np.array([12.0, 12.7, 13.0, 15.2, 18.2, 19.8, 24.1, 28.1])
    x = np.arange(1993, 2009, 2)
    z = np.array([2009])
    print('===============================================')
    print('problem 3:')
    print('the condition of the lakes in 2009 with Lagrange interpolation:      ', lag(x, y, z))
    print('the condition of the lakes in 2009 with spline:      ', [spline(x, y, z[0])], '\n')

    fig = plt.figure(3)
    plt.plot(x, y, 'o')
    plt.plot(x, lag(x, y, x))
    plt.title( 'Toxin Concentration')
    fig.savefig('problem8')

    index = np.argwhere((x==1997) | (x==1999))[:, 0]
    y = np.delete(y , index)
    x = np.delete(x, index)
    z = np.array([1997, 1999])
    print('fill holes with Lagrange interpolation:       ',lag(x, y, z))
    print('fill holes with spline:       ',[spline(x, y, z[0]), spline(x, y, z[1])])

    z = np.linspace(1993, 2009, len(x)*2)
    g = np.zeros(len(z))
    for i in range(len(z)):
        g[i] = spline(x, y, z[i])
    fig = plt.figure(4)
    plt.plot(x, y, 'o', label='data')
    plt.plot(z, lag(x, y, z), label='Lagrange')
    plt.plot(z, g, label='spline')
    plt.title( 'Toxin Concentration')
    plt.legend(loc=9)
    fig.savefig('problem8')
    
if __name__ == "__main__":
    main()
