import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def main():

    L = 2
    n1 = 20
    n2 = 20
    h1 = L/(n1+1)
    h2 = L/(n2+1)
    x1 = np.linspace(0, L, n1+1)
    x2 = np.linspace(0, L, n2+1)

    with open('FDM.npy', 'rb') as npFile:
        T1 = np.load(npFile)
    with open('shooting.npy', 'rb') as npFile:
        T2 = np.load(npFile)

    fig = plt.figure(1)
    plt.title('Temperature 21 grid')
    plt.plot(x1, T1, marker='D', linestyle = 'None', label='FDM')
    plt.plot(x2, T2, label='Shooting method')
    plt.legend()
    plt.savefig('problem26b2')


if __name__ == "__main__":
    main()