import numpy as np
import matplotlib.pyplot as plt

def main():
    
#%% Q1
    
    

    n1, n2, n3, n4 = 9, 17, 33, 65
    x = np.linspace(-1, 1, 30)

    fig = plt.figure(1)
    y = func(x)
    plt.plot(x, y, 'ko', label='data')

    x1 = np.linspace(-1, 1, n1)
    p1 = lag(x, y, x1)
    plt.plot(x1, p1, 'b', label=str(n1)+' points')
    
    x2 = np.linspace(-1, 1, n2)
    p2 = lag(x, y, x2)
    plt.plot(x2, p2, 'g', label=str(n2)+' points')

    x3 = np.linspace(-1, 1, n3)
    p3 = lag(x, y, x3)
    plt.plot(x3, p3, 'r', label=str(n3)+' points')

    x4 = np.linspace(-1, 1, n4)
    p4 = lag(x, y, x4)
    plt.plot(x4, p4, 'c', label=str(n4)+' points')

    plt.title( 'Lagrange Polynomial')
    plt.legend(loc=0)
    fig.savefig('p1a')


    fig = plt.figure(2)
    yPrime = -10 * np.sin(10*x)*np.sin(x)+np.cos(10*x)*np.cos(x)
    plt.plot(x, yPrime, 'ko', label='derivative of data')

    p1 = lagPrime(x, y, x1)
    plt.plot(x1, p1, 'b', label=str(n1)+' points')

    p2 = lagPrime(x, y, x2)
    plt.plot(x2, p2, 'g', label=str(n2)+' points')

    p3 = lagPrime(x, y, x3)
    plt.plot(x3, p3, 'r', label=str(n3)+' points')

    p4 = lagPrime(x, y, x4)
    plt.plot(x4, p4, 'c', label=str(n4)+' points')

    plt.title( 'Derivative of Lagrange Polynomial')
    plt.legend(loc=0)
    fig.savefig('p1b')

    fig = plt.figure(3)
    yPrimeC = fxC(func(x1), x1[1]-x1[0])
    plt.plot(x1, yPrimeC, 'orange', label=str(n1)+' points central derivative')
    plt.plot(x, yPrime, 'ko', label='derivative of data')
    plt.plot(x1, p1, 'b', label=str(n1)+' points lagrange')
    plt.title( 'Derivative of Lagrange Polynomial')
    plt.legend(loc=0)
    fig.savefig('p1c'+str(n1))

    fig = plt.figure(4)
    yPrimeC = fxC(func(x2), x2[1]-x2[0])
    plt.plot(x2, yPrimeC, 'orange', label=str(n2)+' points central derivative')
    plt.plot(x, yPrime, 'ko', label='derivative of data')
    plt.plot(x2, p2, 'g', label=str(n2)+' points lagrange')
    plt.title( 'Derivative of Lagrange Polynomial')
    plt.legend(loc=0)
    fig.savefig('p1c'+str(n2))

    fig = plt.figure(5)
    yPrimeC = fxC(func(x3), x3[1]-x3[0])
    plt.plot(x3, yPrimeC, 'orange', label=str(n3)+' points central derivative')
    plt.plot(x, yPrime, 'ko', label='derivative of data')
    plt.plot(x3, p3, 'r', label=str(n3)+' points lagrange')
    plt.title( 'Derivative of Lagrange Polynomial')
    plt.legend(loc=0)
    fig.savefig('p1c'+str(n3))

    fig = plt.figure(6)
    yPrimeC = fxC(func(x4), x4[1]-x4[0])
    plt.plot(x4, yPrimeC, 'orange', label=str(n4)+' points central derivative')
    plt.plot(x, yPrime, 'ko', label='derivative of data')
    plt.plot(x4, p4, 'c', label=str(n4)+' points lagrange')
    plt.title( 'Derivative of Lagrange Polynomial')
    plt.legend(loc=0)
    fig.savefig('p1c'+str(n4))

def func(x):
    return np.cos(10*x)*np.sin(x)


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

def lagPrime(x, y, z):

    z = z.astype('float64')
    m = len(x)
    n = len(z)
    p = np.empty_like(z)
    for k in range(n):
        p[k] = 0
        for j in range(m):
            tmp2 = 0
            for i in range(m):
                if i != j:
                    tmp1 = 1.0 / (x[j] - x[i])
                    tmp = 1
                    for l in range(m):
                        if l != j and l != i:
                            tmp = tmp * (z[k] - x[l]) / (x[j] - x[l])
                    tmp2 = tmp2 + tmp * tmp1
                    
            p[k] = p[k] + y[j] * tmp2
    return p

def fxC(y, h):

    yPrime = np.zeros((len(y)))
    i = 0
    yPrime[i] = (y[i+1] - y[i]) / (h)
    i = len(y) - 1
    yPrime[i] = (y[i] - y[i-1]) / (h)
    for i in range(1, len(y)-1):
        yPrime[i] = (y[i+1] - y[i-1]) / (2*h)
    return yPrime

if __name__ == "__main__":
    main()
