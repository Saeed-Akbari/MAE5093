import numpy as np
import matplotlib.pyplot as plt

def main():

    Ta, TA, TB = 0, 5, 4
    L = 2
    n = 20
    h = L/(n+1)
    x = np.linspace(0, L, n+1)
    #T = map(np.zeros, ((len(x))))
    a, b, c, d = map(np.zeros, (len(x), len(x), len(x), len(x)))
    b[0], b[-1] = 1.0, 1.0
    d[0], d[-1] = TA, TB
    for i in range(1, n):
        a[i] = 1.0 - 0.5*h*cal_a(x[i])
        b[i] = (h**2) * cal_b(x[i]) - 2.0
        c[i] = 1.0 + 0.5*h*cal_a(x[i])
        d[i] = cal_f(x[i], cal_b(x[i])) * h**2

    T = tdma(a,b,c,d)

    with open('FDM.npy', 'wb') as npFile:
        np.save(npFile, T)

    fig = plt.figure(1)
    plt.title('Problem 26 Central Difference')
    plt.plot(x, T, marker='D', linestyle = 'None', label='temperature')
    plt.legend()
    plt.savefig('problem26b')


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

def cal_a(x):
    return - (x + 3.0) / (x + 1.0)
def cal_b(x):
    return (x + 3.0) / (x + 1.0)**2
def cal_f(x, b):
    return 2*(x + 1.0) + 3.0*b

if __name__ == "__main__":
    main()