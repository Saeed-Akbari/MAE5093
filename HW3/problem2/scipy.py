import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def main():

    l = 0
    r = 3
    v0 = 1
    dt = [0.001]

    F = lambda t, s: -0.2*s-2*s*s*np.cos(2*t)

    t_eval = np.arange(0, 7, 0.01)
    sol = solve_ivp(F, [0, 7], [1], t_eval=t_eval)

    plt.figure(figsize = (12, 4))
    plt.plot(sol.t, sol.y[0])
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()