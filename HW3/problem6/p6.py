import numpy as np
import matplotlib.pyplot as plt

def main():
    
    l = 0
    r = 15

    h = 0.01
    x = np.arange(l, r, h)
    lenx = len(x)
    y = np.zeros((lenx))
    y = (1 + 3*x + 3 * x**2 + x**3) / np.exp(x)


    v0 = 1.0
    dt = [0.2, 0.8, 1.1]
    '''     forward Euler       '''
    fig = plt.figure(1)
    plt.title('forwardEuler')
    for h in dt:
        t = np.arange(l, r, h)
        lent = len(t)
        v = np.zeros((lent))
        v[0] = v0
        for i in range(lent-1):
            alpha = (3*t[i]) / (1.0+t[i])
            beta = 2 * np.exp(-t[i]) * (1.0 + t[i])**3
            f = - alpha * v[i] + beta
            v[i+1] = v[i] + h * f
        plt.plot(t, v, label=f'h={h}')
    plt.plot(x, y, label='Analytical')
    plt.gca().set_ylim([-0.5, 5])
    plt.legend()
    plt.savefig('forwardEuler')
    
    '''     backward Euler       '''
    fig = plt.figure(2)
    plt.title('backwardEuler')
    for h in dt:
        t = np.arange(l, r, h)
        lent = len(t)
        v = np.zeros((lent))
        v[0] = v0
        for i in range(lent-1):
            alpha = (3*t[i+1]) / (1.0+t[i+1])
            beta = 2 * np.exp(-t[i+1]) * (1.0 + t[i+1])**3
            v[i+1] = (v[i] + h * beta) / (1.0 + h * alpha)
        plt.plot(t, v, label=f'h={h}')
    plt.plot(x, y, label='Analytical')
    plt.gca().set_ylim([-0.5, 4])    
    plt.legend()
    plt.savefig('backwardEuler')

    '''     trapezoidal       '''
    fig = plt.figure(3)
    plt.title('trapezoidal')
    for h in dt:
        t = np.arange(l, r, h)
        lent = len(t)
        v = np.zeros((lent))
        v[0] = v0
        for i in range(lent-1):
            alpha = (3*t[i]) / (1.0+t[i])
            beta = 2 * np.exp(-t[i]) * (1.0 + t[i])**3
            fl = - alpha * v[i] + beta
            alpha = (3*t[i+1]) / (1.0+t[i+1])
            beta = 2 * np.exp(-t[i+1]) * (1.0 + t[i+1])**3
            v[i+1] = (v[i] + (0.5*h) * (fl + beta)) / (1.0 + (0.5*h) * alpha )
        plt.plot(t, v, label=f'h={h}')
    plt.plot(x, y, label='Analytical')
    plt.gca().set_ylim([-0.5, 4])
    plt.legend()
    plt.savefig('trapezoidal')

    '''     second-order Runge–Kutta       '''
    fig = plt.figure(4)
    plt.title('second-order Runge–Kutta')
    for h in dt:
        t = np.arange(l, r, h)
        lent = len(t)
        v = np.zeros((lent))
        v[0] = v0
        for i in range(lent-1):
            alpha = (3*t[i]) / (1.0+t[i])
            beta = 2 * np.exp(-t[i]) * (1.0 + t[i])**3
            f = - alpha * v[i] + beta
            temp = v[i] + (0.5*h) * f
            alpha = (3*(t[i]+0.5*h)) / (1.0+(t[i]+0.5*h))
            beta = 2 * np.exp(-(t[i]+0.5*h)) * (1.0 + (t[i]+0.5*h))**3
            f = - alpha * temp + beta
            v[i+1] = v[i] + h * f
        plt.plot(t, v, label=f'h={h}')
    plt.plot(x, y, label='Analytical')
    plt.gca().set_ylim([-0.5, 4])
    plt.legend()
    plt.savefig('secondRunge–Kutta')


    '''     fourth-order Runge–Kutta       '''
    fig = plt.figure(5)
    plt.title('fourth-order Runge–Kutta')
    cte = 0.166666667
    for h in dt:
        t = np.arange(l, r, h)
        lent = len(t)
        v = np.zeros((lent))
        v[0] = v0
        for i in range(lent-1):
            alpha = (3*t[i]) / (1.0+t[i])
            beta = 2 * np.exp(-t[i]) * (1.0 + t[i])**3
            f = - alpha * v[i] + beta
            k1 = h * f
            alpha = (3*(t[i]+0.5*h)) / (1.0+(t[i]+0.5*h))
            beta = 2 * np.exp(-(t[i]+0.5*h)) * (1.0 + (t[i]+0.5*h))**3
            f = - alpha * (v[i]+0.5*k1) + beta
            k2 = h * f
            alpha = (3*(t[i]+0.5*h)) / (1.0+(t[i]+0.5*h))
            beta = 2 * np.exp(-(t[i]+0.5*h)) * (1.0 + (t[i]+0.5*h))**3
            f = - alpha * (v[i]+0.5*k2) + beta
            k3 = h * f
            alpha = (3*(t[i]+h)) / (1.0+(t[i]+h))
            beta = 2 * np.exp(-(t[i]+h)) * (1.0 + (t[i]+h))**3
            f = - alpha * (v[i]+k3) + beta
            k4 = h * f
            v[i+1] = v[i] + cte*k1 + 2*cte*(k2+k3) + cte*k4
        plt.plot(t, v, label=f'h={h}')
    plt.plot(x, y, label='Analytical')
    plt.gca().set_ylim([-0.5, 4])
    plt.legend()
    plt.savefig('fourthRunge–Kutta')

if __name__ == "__main__":
    main()
