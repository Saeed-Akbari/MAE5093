import numpy as np
import matplotlib.pyplot as plt


def main():

    N = 24
    L = 1
    h = L / N
    x = np.zeros((N))
    for i in range(N):
        x[i] = (i + 0.5) * h
    
    C = np.zeros((N, N))
    B = np.zeros((N, N))
    A = np.zeros((N, N))

    i = 0
    A[i][i] = 23
    A[i][i+1] = -11
    A[i][i+2] = 0
    B[i][i] = -3 * 12/ h**2
    B[i][i+1] = 4 * 12/ h**2
    B[i][i+2] = -1 * 12/ h**2

    for i in range(1, N-1):
        A[i][i-1] = 1
        A[i][i] = 10
        A[i][i+1] = 1
        B[i][i-1] = 1 * 12/ h**2
        B[i][i] = -2 * 12/ h**2
        B[i][i+1] = 1 * 12/ h**2

    i = N-1
    A[i][i] = 23
    A[i][i-1] = -11
    A[i][i-2] = 0
    B[i][i] = -3 * 12/ h**2
    B[i][i-1] = 4 * 12/ h**2
    B[i][i-2] = -1 * 12/ h**2

    C = A + B
    sol = np.linalg.inv(C) @ (A @ (np.expand_dims(x**3, axis=1)))
    
    fig = plt.figure(1)
    plt.plot(x, sol, label='4th order Pade')
    plt.plot(x, ExactSol(x), 'k.', label='exact solution')
    #plt.ylabel('values of second derivative')
    plt.legend()
    plt.savefig('p8')

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

def ExactSol(x):
    return (((3 * (2 * np.cos(1) - 1)) / np.sin(1)) * np.cos(x) + 6 * np.sin(x) + x**3 - 6 * x)

def func(x):
    return x**3

if __name__ == "__main__":
    main()
