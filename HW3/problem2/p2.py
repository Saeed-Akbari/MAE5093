import numpy as np
import matplotlib.pyplot as plt

def main():

    l = 0
    r = 7
    v0 = 1
    dt = [0.2, 0.05, 0.025, 0.006, 0.002, 0.001, 0.0005]
    '''     explicit Euler       '''
    fig = plt.figure(1)
    plt.title('Problem 2')
    plt.gca().set_xlim([l, r])
    plt.gca().set_ylim([l, 1.4])
    for h in dt:
        t = np.arange(l, r, h)
        lent = len(t)
        v = np.zeros((lent))
        v[0] = v0
        for i in range(lent-1):
            f = -0.2*v[i] - 2 * np.cos(2*t[i]) * v[i]**2
            v[i+1] = v[i] + h * f
        plt.plot(t, v, label=f'h={h}')
    '''t = np.arange(l, r, 0.01)
    lent = len(t)
    v = np.zeros((lent))
    for i in range(len(t)):
        v[i] = (101.0) / (100 * np.sin(2*t[i]) - 10*np.cos(2*t[i])+111*np.exp(0.2*t[i]))
    plt.plot(t, v, label='Analytical')
    '''
    plt.legend()
    plt.savefig('problem2')

if __name__ == "__main__":
    main()