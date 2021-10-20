import numpy as np
import matplotlib.pyplot as plt

def main():

    Ta, TA, TB = 0, 5, 4
    L = 2
    n = 99
    h = L/(n+1)
    A1 = 50
    A2 = -28
    x = np.linspace(0, L, n+1)
    u, v = map(np.zeros, ((len(x)), (len(x))))

    u[0] = TA
    v[0] = A1
    u, v = rk4(u, v, x, h)
    B1 = u[-1]

    u[0] = TA
    v[0] = A2
    u, v = rk4(u, v, x, h)
    B2 = u[-1]

    j = 0
    while np.abs(TB - B2) > 1e-6:
        guess = A2 + (TB - B2)/((B2-B1)/(A2-A1))
        u[0] = TA
        v[0] = guess
        u, v = rk4(u, v, x, h)
        B1 = B2
        B2 = u[-1]
        A1 = A2
        A2 = guess
        j = j + 1

    with open('shooting.npy', 'wb') as npFile:
        np.save(npFile, u)

    fig = plt.figure(1)
    plt.title('Problem 26')
    plt.plot(x, u, label='temperature')
    plt.legend()
    plt.savefig('problem26a1')
    
def rk4(u, v, x, h):

    '''     fourth-order Runge–Kutta       '''
    cte = 0.166666667
    for i in range(len(x)-1):
        a, b = cal_a(x[i]), cal_b(x[i])
        f = cal_f(x[i], b)
        ut, vt = u[i], v[i]
        fu = vt
        fv = -a * vt - b * ut + f
        k1u = h * fu
        k1v = h * fv
        a, b = cal_a(x[i]+0.5*h), cal_b(x[i]+0.5*h)
        f = cal_f(x[i]+0.5*h, b)
        ut, vt = u[i]+0.5*k1u, v[i]+0.5*k1v
        fu = vt
        fv = -a * vt - b * ut + f
        k2u = h * fu
        k2v = h * fv
        a, b = cal_a(x[i]+0.5*h), cal_b(x[i]+0.5*h)
        f = cal_f(x[i]+0.5*h, b)
        ut, vt = u[i]+0.5*k2u, v[i]+0.5*k2v
        fu = vt
        fv = -a * vt - b * ut + f
        k3u = h * fu
        k3v = h * fv
        a, b = cal_a(x[i]+h), cal_b(x[i]+h)
        f = cal_f(x[i]+h, b)
        ut, vt = u[i]+k3u, v[i]+k3v
        fu = vt
        fv = -a * vt - b * ut + f
        k4u = h * fu
        k4v = h * fv
        u[i+1] = u[i] + cte*k1u + 2*cte*(k2u+k3u) + cte*k4u
        v[i+1] = v[i] + cte*k1v + 2*cte*(k2v+k3v) + cte*k4v

    return u, v

def cal_a(x):
    return - (x + 3.0) / (x + 1.0)
def cal_b(x):
    return (x + 3.0) / (x + 1.0)**2
def cal_f(x, b):
    return 2*(x + 1.0) + 3.0*b

if __name__ == "__main__":
    main()