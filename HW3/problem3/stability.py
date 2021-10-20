import numpy as np
import matplotlib.pyplot as plt

list1 = np.arange(0,1.1,0.2)
sr = np.arange(-1.5, 0.5, 0.01)
si = np.arange(-1, 1, 0.01)
[Sr, Si] = np.meshgrid(sr,si)
S = Sr + Si*1j
A2 = 0.5 * (1.0 + 1.5 * S + np.sqrt((1.0+1.5*S)**2-2*S))
A3 = 0.5 * (1.0 + 1.5 * S - np.sqrt((1.0+1.5*S)**2-2*S))
# plot
fig, ax = plt.subplots(1, 1)
cont2 = ax.contour(Sr, Si, np.abs(A2), [1], colors='b')
cont3 = ax.contour(Sr, Si, np.abs(A3), [1], colors='g')
ax.set_aspect('equal')
cont2.clabel()
cont3.clabel()
plt.savefig('rungeKutta')

fig = plt.figure(2)
cont2 = plt.contour(Sr, Si, np.abs(A2), list1, colors='b')
plt.gca().set_aspect('equal')
cont2.clabel()
plt.savefig('rungeKutta2')

fig = plt.figure(3)
cont3 = plt.contour(Sr, Si, np.abs(A3), list1, colors='g')
plt.gca().set_aspect('equal')
cont3.clabel()
plt.savefig('rungeKutta3')