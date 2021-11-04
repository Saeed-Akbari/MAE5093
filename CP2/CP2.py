import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():

    a = 1.0
    tmax = 10
    Lx = 1
    nx = 99
    hx = Lx/(nx+1)
    #ht = 0.005
    ht = 0.01
    nt = int(tmax / ht) + 1
    x = np.linspace(0, Lx, nx+1)
    c = a * ht / hx
    u0 = np.sin(2*np.pi*x)

    uEuler, uLax, uRK3 = map(np.copy, (u0, u0, u0))
    u2DEuler, u2DLax, u2DRK3, uExact = map(np.zeros, ((nt+1, nx+1), (nt+1, nx+1), (nt+1, nx+1), (nt+1, nx+1)))
    u2DEuler[0], u2DLax[0], u2DRK3[0] = map(np.copy, (u0, u0, u0))
    
    for n in range(nt):
        uEuler = Euler(uEuler, nx, c)
        uLax = LaxWendroff(uLax, nx, c)
        uRK3 = rungeKutta(uRK3, nx, c)
        u2DEuler[n+1] = uEuler
        u2DLax[n] = uLax
        u2DRK3[n] = uRK3
    
    time = np.arange(0, tmax+ht/10, ht)
    for n in range(nt):
        uExact[n] = exactSol(x,time[n],a)

    figNum = 1
    fileName = 't1c1'
    plotTitle = 't = 1s and c = 1'
    label1 = 'Euler'
    label2 = 'Lax Wendroff'
    label3 = 'Runge Kutta 3th'
    label4 = 'Exact Solution'
    ind = 100
    plot(figNum, fileName, plotTitle, label1, label2, label3, label4, x, u2DEuler[ind], u2DLax[ind], u2DRK3[ind], uExact[ind])

    figNum = 2
    fileName = 't10c1'
    plotTitle = 't = 10s and c = 1'
    ind = 1000
    plot(figNum, fileName, plotTitle, label1, label2, label3, label4, x, u2DEuler[ind], u2DLax[ind], u2DRK3[ind], uExact[ind])

    #animPlot(x, u2DEuler)
    #animPlot(x, u2DLax)
    #animPlot(x, u2DRK3)
    #animPlot(x, uExact)
    animPlotE(x, u2DRK3, uExact)

def exactSol(x,t,a):
    return (np.sin(2*np.pi*(x-a*t)))

def Euler(u, nx, c):
    
    un = np.copy(u)
    for i in range(nx, 0, -1):
        u[i] = un[i] - c * (un[i] - un[i-1])
    u[0] = u[nx]
    return u


def LaxWendroff(u, nx, c):

    un = np.copy(u)
    i = nx
    u[i] = un[i] - 0.5 * c * (un[0] - un[i-1]) + 0.5 * (c**2) * (un[0] - 2 * un[i] + un[i-1])
    for i in range(nx-1, 0, -1):
        u[i] = un[i] - 0.5 * c * (un[i+1] - un[i-1]) + 0.5 * (c**2) * (un[i+1] - 2 * un[i] + un[i-1])
    u[0] = u[nx]
    return u


def rungeKutta(u, nx, c):
    
    un = np.copy(u)
    u1, u2 = map(np.zeros_like, (u, u))
    
    for i in range(nx-1, -1, -1):
        u1[i] = un[i] - 0.5 * c * (un[i+1] - un[i-1])
    i = nx
    u1[i] = un[i] - 0.5 * c * (un[0] - un[i-1])

    for i in range(nx-1, -1, -1):        
        u2[i] = 0.75 * un[i] + 0.25 * u1[i] - 0.125 * c * (u1[i+1] - u1[i-1])
    i = nx
    u2[i] = 0.75 * un[i] + 0.25 * u1[i] - 0.125 * c * (u1[0] - u1[i-1])

    for i in range(nx-1, -1, -1):
        u[i] = (un[i] + 2 * u2[i] - c * (u2[i+1] - u2[i-1])) / 3 
    i = nx
    u[i] = (un[i] + 2 * u2[i] - c * (u2[0] - u2[i-1])) / 3 

    return u

def animPlot(x, u):

    fig, ax = plt.subplots()


    line, = ax.plot(x, u[0], marker = 'o', linestyle = 'None')


    def animate(i):
        line.set_ydata(u[i])  # update the data.
        return line,


    ani = animation.FuncAnimation(
        fig, animate, interval=10, blit=True, save_count=50)

    # To save the animation, use e.g.
    #
    #ani.save("RK3.mp4")
    #
    # or
    #
    #writer = animation.FFMpegWriter(
    #    fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save("RK3.mp4", writer=writer)

    #plt.show()


def animPlotE(x, u, uE):

    fig, ax = plt.subplots()

    line, = ax.plot(x, u[0], 'b', label='RK3')
    lineE, = ax.plot(x, uE[0], 'r', label='Exact', marker = 'o', markerfacecolor='None', linestyle = 'None')


    def animate(i):
        line.set_ydata(u[i])
        lineE.set_ydata(uE[i])
        return line, lineE

    plt.title('Runge Kutta 3th order')
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m.s^-1)')
    plt.legend()

    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, save_count=50)

    # To save the animation, use e.g.
    #
    #ani.save("RK3.mp4")
    #
    # or
    #
    writer = animation.FFMpegWriter(
        fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("RK3.mp4", writer=writer)

    #plt.show()

def plot(figNum, fileName, plotTitle, label1, label2, label3, label4, x, phi1, phi2, phi3, phi4):

    plt.figure(figNum)
    plt.plot(x, phi1, 'b', label=label1)
    plt.plot(x, phi2, 'r', label=label2)
    plt.plot(x, phi3, 'g', label=label3)
    plt.plot(x, phi4, 'k', label=label4, marker = 'o', markerfacecolor='none', linestyle = 'None')
    plt.title(plotTitle)
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m.s^-1)')
    plt.legend()
    filename = fileName
    plt.savefig(filename, dpi = 200)



if __name__ == "__main__":
    main()