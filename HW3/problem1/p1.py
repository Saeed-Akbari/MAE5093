import numpy as np
import matplotlib.pyplot as plt

def main():

    l = 0
    r = 15
    y0 = 4
    dx = [0.1, 0.5, 1.0]
    '''     forward Euler       '''
    for h in dx:
        x = np.arange(l, r, h)
        lenx = len(x)
        y = np.zeros((lenx))
        y[0] = y0
        for i in range(lenx-1):
            y[i+1] = y[i] - h * (2.0 + 0.01 * x[i]**2) * y[i]
        fig = plt.figure(1)
        plt.title('forwardEuler')
        plt.gca().set_ylim([-0.5, 5])
        plt.plot(x, y, label=f'h={h}')
    plt.legend()
    plt.savefig('forwardEuler')
    
    '''     backward Euler       '''
    for h in dx:
        x = np.arange(l, r, h)
        lenx = len(x)
        y = np.zeros((lenx))
        y[0] = y0
        for i in range(lenx-1):
            y[i+1] = y[i] / (1 + h * (2.0 + 0.01 * x[i]**2))
        fig = plt.figure(2)
        plt.title('backwardEuler')
        plt.gca().set_ylim([-0.5, 5])
        plt.plot(x, y, label=f'h={h}')
    plt.legend()
    plt.savefig('backwardEuler')

    '''     trapezoidal       '''
    for h in dx:
        x = np.arange(l, r, h)
        lenx = len(x)
        y = np.zeros((lenx))
        y[0] = y0
        for i in range(lenx-1):
            fl = - (2.0 + 0.01 * x[i]**2) * y[i]
            fr = - (2.0 + 0.01 * x[i+1]**2)
            y[i+1] = (y[i] + (0.5*h) * fl) / (1.0 - (0.5*h) * fr )
            fig = plt.figure(3)
        plt.title('trapezoidal')
        plt.gca().set_ylim([-0.5, 5])
        plt.plot(x, y, label=f'h={h}')
    plt.legend()
    plt.savefig('trapezoidal')

    '''     second-order Runge–Kutta       '''
    for h in dx:
        x = np.arange(l, r, h)
        lenx = len(x)
        y = np.zeros((lenx))
        y[0] = y0
        for i in range(lenx-1):
            f = - (2.0 + 0.01 * x[i]**2) * y[i]
            temp = y[i] + (0.5*h) * f
            f = - (2.0 + 0.01 * (0.5*(x[i]+x[i+1]))**2) * temp
            y[i+1] = y[i] + h * f
            fig = plt.figure(4)
        plt.title('second-order Runge–Kutta')
        plt.gca().set_ylim([-0.5, 5])
        plt.plot(x, y, label=f'h={h}')
    plt.legend()
    plt.savefig('secondRunge–Kutta')


    '''     fourth-order Runge–Kutta       '''
    cte = 0.166666667
    for h in dx:
        x = np.arange(l, r, h)
        lenx = len(x)
        y = np.zeros((lenx))
        y[0] = y0
        for i in range(lenx-1):
            f = - (2.0 + 0.01 * x[i]**2) * y[i]
            k1 = h * f
            f = - (2.0 + 0.01 * (x[i]+0.5*h)**2) * (0.5*k1 + y[i])
            k2 = h * f
            f = - (2.0 + 0.01 * (x[i]+0.5*h)**2) * (0.5*k2 + y[i])
            k3 = h * f
            f = - (2.0 + 0.01 * (x[i]+h)**2) * (k3 + y[i])
            k4 = h * f
            y[i+1] = y[i] + cte*k1 + 2*cte*(k2+k3) + cte*k4
        fig = plt.figure(5)
        plt.title('fourth-order Runge–Kutta')
        plt.gca().set_ylim([-0.5, 5])
        plt.plot(x, y, label=f'h={h}')
    plt.legend()
    plt.savefig('fourthRunge–Kutta')

if __name__ == "__main__":
    main()
