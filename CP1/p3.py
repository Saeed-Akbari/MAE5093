import numpy as np
import matplotlib.pyplot as plt

def main():

    n = [8, 16, 32, 64]
    x1 = np.linspace(-1, 1, n[-1]+1)
    yPrime = funcP(x1)
    ypp = funcPP(x1)

    for i in range(len(n)):
        x = np.linspace(-1, 1, n[i]+1)
        func1 = func(x)
        fPP = o4d2(x, func1)
        fPrime = o4d1(x, func1)
        fPrimePrime = o4d1(x, fPrime)
        fig = plt.figure(i+1)
        plt.title('Second Derivative with '+str(n[i])+' Points')
        plt.plot(x, fPP, label='second derivative')
        plt.plot(x, fPrimePrime, label='two times first derivative')
        plt.plot(x1, ypp, 'k.', label='data')
        plt.ylabel('values of second derivative')
        plt.legend()
        plt.savefig('problem2c_'+str(i+1))

    omega = 0.5
    n = range(9, 999, 10)
    error = np.zeros((len(n)))
    h = np.zeros((len(n)))
    for i in range(len(n)):
        x = np.linspace(-1, 1, n[i]+1)
        ind = np.argwhere(x>=omega)[0][0]
        fPP = o4d2(x)
        error[i] = np.abs(fPP[ind] - funcPP(x[ind]))
        h[i] = 2.0 / (n[i] + 1)




    fig = plt.figure(5)
    plt.title('Error for second derivative of Pade method')
    plt.plot(h, error)
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xlabel('delta x')
    plt.savefig('problem2c_'+str(len(n)))

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

def funcP(x1):
    return -10 * np.sin(10*x1)*np.sin(x1)+np.cos(10*x1)*np.cos(x1)

def funcPP(x1):
    return -20 * np.sin(10*x1)*np.cos(x1)-101*np.cos(10*x1)*np.sin(x1)

def o4d2(x, func):

    n = len(x)-1
    h = x[1] - x[0]
    a, b, c, d = map(np.zeros, (len(x), len(x), len(x), len(x)))
    b[0], b[n] = 1.0, 1.0
    c[0], a[n] = 11.0, 11.0
    i = 0
    d[i] = (1.0/(h**2))*(13*func[i]-27*func[i+1]+15*func[i+2]-func[i+3])
    i = n
    d[i] = (1.0/(h**2))*(13*func[i]-27*func[i-1]+15*func[i-2]-func[i-3])
    for i in range(1, n):
        a[i] = 1.0
        b[i] = 10.0
        c[i] = 1.0
        d[i] = (12.0/(h**2))*(func[i+1] - 2 * func[i] + func[i-1])

    return tdma(a,b,c,d)

def o4d1(x, func):

    n = len(x)-1
    h = x[1] - x[0]
    a, b, c, d = map(np.zeros, (len(x), len(x), len(x), len(x)))
    b[0], b[n] = 1.0, 1.0
    c[0], a[n] = 2.0, 2.0
    i = 0
    d[i] = (1.0/h)*(-2.5*func[i]+2*func[i+1]+0.5*func[i+2])
    i = n
    d[i] = (1.0/h)*(2.5*func[i]-2*func[i-1]-0.5*func[i-2])
    for i in range(1, n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        d[i] = (3.0/h)*(func[i+1] - func[i-1])

    return tdma(a,b,c,d)

if __name__ == "__main__":
    main()