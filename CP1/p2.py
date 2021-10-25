import numpy as np
import matplotlib.pyplot as plt

def main():

    n = [8, 16, 32, 64]
    x1 = np.linspace(-1, 1, n[-1]+1)
    yPrime = funcPr(x1)

    for i in range(len(n)):
        x = np.linspace(-1, 1, n[i]+1)
        fPrime = o4d1(x)
        fig = plt.figure(i+1)
        plt.title('First Derivative with '+str(n[i])+' Points')
        plt.plot(x, fPrime, label='Pade')
        plt.plot(x1, yPrime, 'k.', label='data')
        plt.legend()
        plt.savefig('p2a_'+str(i+1))


    x1 = np.linspace(-1, 1, 30)
    j = 6
    for i in range(len(n)):
        fig = plt.figure(j)
        x = np.linspace(-1, 1, n[i]+1)
        p1 = lagPrime(x1, func(x1), x)
        yPrimeC = fxC(func(x), x[1]-x[0])
        yPrime = funcPr(x1)
        fPrime = o4d1(x)
        plt.plot(x, yPrimeC, 'orange', label=str(n[i])+' points central derivative')
        plt.plot(x1, yPrime, 'ko', label='derivative of data')
        plt.plot(x, p1, 'b', label=str(n[i])+' points lagrange')
        plt.plot(x, fPrime, 'g', label='Pade')
        plt.title( 'Derivative of Lagrange Polynomial')
        plt.legend(loc=0)
        fig.savefig('p2c_'+str(n[i]))
        j = j + 1


    omega = 0.5
    n = range(9, 999, 10)
    error = np.zeros((len(n)))
    h = np.zeros((len(n)))
    for i in range(len(n)):
        x = np.linspace(-1, 1, n[i]+1)
        ind = np.argwhere(x>=omega)[0][0]
        fPrime = o4d1(x)
        #error[i] = np.abs(fPrime[ind] - yPrime[ind1])
        error[i] = np.abs(fPrime[ind] - funcPr(x[ind]))
        h[i] = 2.0 / (n[i] + 1)

    fig = plt.figure(10)
    plt.title('Error for first derivative of Pade method')
    plt.plot(h, error)
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xlabel('delta x')
    plt.ylabel('error of Pade for first derivative')
    plt.savefig('p2b_'+str(len(n)))


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

def func(x):
    return np.cos(10*x)*np.sin(x)

def funcPr(x1):
    return -10 * np.sin(10*x1)*np.sin(x1)+np.cos(10*x1)*np.cos(x1)

def o4d1(x):

    n = len(x)-1
    h = x[1] - x[0]
    a, b, c, d = map(np.zeros, (len(x), len(x), len(x), len(x)))
    b[0], b[n] = 1.0, 1.0
    c[0], a[n] = 2.0, 2.0
    i = 0
    d[i] = (1.0/h)*(-2.5*func(x[i])+2*func(x[i+1])+0.5*func(x[i+2]))
    i = n
    d[i] = (1.0/h)*(2.5*func(x[i])-2*func(x[i-1])-0.5*func(x[i-2]))
    for i in range(1, n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        d[i] = (3.0/h)*(func(x[i+1]) - func(x[i-1]))

    return tdma(a,b,c,d)
    
def fxC(y, h):

    yPrime = np.zeros((len(y)))
    i = 0
    yPrime[i] = (y[i+1] - y[i]) / (h)
    i = len(y) - 1
    yPrime[i] = (y[i] - y[i-1]) / (h)
    for i in range(1, len(y)-1):
        yPrime[i] = (y[i+1] - y[i-1]) / (2*h)
    return yPrime
    
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

if __name__ == "__main__":
    main()
