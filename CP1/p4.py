import numpy as np
import matplotlib.pyplot as plt

list1 = np.arange(0,1.1,0.2)
sr = np.arange(-3, 1, 0.1)
si = np.arange(-4, 4, 0.1)
[Sr, Si] = np.meshgrid(sr,si)
S = Sr + Si*1j
A2 = 1 + S + 0.5*S**2
A3 = 1 + S + 0.5*S**2 + (1/6)*(S**3)
A4 = 1 + S + 0.5*S**2 + (1/6)*(S**3) + (1/24)*(S**4)
# plot
fig, ax = plt.subplots(1, 1)
cont2 = ax.contour(Sr, Si, np.abs(A2), [1], colors='r')
cont3 = ax.contour(Sr, Si, np.abs(A3), [1], colors='g')
cont4 = ax.contour(Sr, Si, np.abs(A4), [1], colors='b')
ax.set_aspect('equal')
cont2.clabel()
cont3.clabel()
cont4.clabel()
plt.savefig('rungeKutta')

fig = plt.figure(2)
cont2 = plt.contour(Sr, Si, np.abs(A2), list1, colors='r')
plt.gca().set_aspect('equal')
cont2.clabel()
plt.savefig('rungeKutta2')

fig = plt.figure(3)
cont3 = plt.contour(Sr, Si, np.abs(A3), list1, colors='g')
plt.gca().set_aspect('equal')
cont3.clabel()
plt.savefig('rungeKutta3')

fig = plt.figure(4)
cont4 = plt.contour(Sr, Si, np.abs(A4), list1, colors='b')
plt.gca().set_aspect('equal')
cont4.clabel()
plt.savefig('rungeKutta4')