import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    plt.show()


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

def plot(figNum, fileName, plotTitle, label1, label2, label3, label4, label5, x, phi1, phi2, phi3, phi4, phi5):

    plt.figure(figNum)
    plt.plot(x, phi1, 'b', label=label1)
    plt.plot(x, phi2, 'r', label=label2)
    plt.plot(x, phi3, 'g', label=label3)
    plt.plot(x, phi4, 'c', label=label4)
    plt.plot(x, phi5, 'k', label=label5, marker = 'o', markerfacecolor='none', linestyle = 'None')
    plt.title(plotTitle)
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m.s^-1)')
    plt.legend()
    plt.savefig(fileName, dpi = 200)

def plotErr(figNum, fileName, plotTitle, label1, label2, label3, label4, x, phi1, phi2, phi3, phi4):

    plt.figure(figNum)
    plt.plot(x, phi1, 'b', label=label1)
    plt.plot(x, phi2, 'r', label=label2, marker = '*', markerfacecolor='none', linestyle = 'None')
    plt.plot(x, phi3, 'g', label=label3, marker = 'o', markerfacecolor='none', linestyle = 'None')
    plt.plot(x, phi4, 'c', label=label4)
    plt.title(plotTitle)
    plt.xlabel('Distance')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fileName, dpi = 200)

def plot1(figNum, fileName, x, phi1):

    plt.figure(figNum)
    plt.plot(x, phi1, 'b')
    plt.xlabel('$\omega$')
    plt.ylabel('iteration')
    plt.savefig(fileName, dpi = 200)

def plot2(figNum, fileName, plotTitle, label1, label2, x, phi1, phi2):

    plt.figure(figNum)
    plt.plot(x, phi1, 'b', label=label1)
    plt.plot(x, phi2, 'k', label=label2, marker = 'o', markerfacecolor='none', linestyle = 'None')
    plt.xlabel('$\omega$')
    plt.ylabel('iteration')
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(fileName, dpi = 200)

def subplot(figNum, fileName, plotTitle, label1, label2, label3, label4, labelExt, x, phi1, phi2, phi3, phi4, Ext):

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(plotTitle)

    axs.flat[0].plot(x, phi1, 'b', label=label1)
    axs.flat[0].plot(x, Ext, 'k', label=labelExt, marker = 'o', markersize=4, markerfacecolor='none', linestyle = 'None')
    axs.flat[0].legend([label1, labelExt], loc=0, prop={'size': 5})

    axs.flat[1].plot(x, phi2, 'r', label=label2)
    axs.flat[1].plot(x, Ext, 'k', label=labelExt, marker = 'o', markersize=4, markerfacecolor='none', linestyle = 'None')
    axs.flat[1].legend([label2, labelExt], loc=0, prop={'size': 5})

    axs.flat[2].plot(x, phi3, 'g', label=label3)
    axs.flat[2].plot(x, Ext, 'k', label=labelExt, marker = 'o', markersize=4, markerfacecolor='none', linestyle = 'None')
    axs.flat[2].legend([label3, labelExt], loc=0, prop={'size': 5})

    axs.flat[3].plot(x, phi4, 'c', label=label4)
    axs.flat[3].plot(x, Ext, 'k', label=labelExt, marker = 'o', markersize=4, markerfacecolor='none', linestyle = 'None')
    axs.flat[3].legend([label4, labelExt], loc=0, prop={'size': 5})
    
    plt.legend()
    plt.savefig(fileName, dpi = 200)
    fig.clear(True)

def subplotErr(figNum, fileName, plotTitle, label1, label2, label3, label4, x, phi1, phi2, phi3, phi4):

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(plotTitle)

    axs.flat[0].plot(x, phi1, 'b', label=label1)
    axs.flat[1].plot(x, phi2, 'r', label=label2)
    axs.flat[2].plot(x, phi3, 'g', label=label3)
    axs.flat[3].plot(x, phi4, 'c', label=label4)

    for i in range(4):
        axs.flat[i].set_yscale('log')
    
    plt.savefig(fileName, dpi = 200)
    fig.clear(True)


def contourPlot(X, Y, phi, fileName='Gauss_Seidel', figSize=(14,7)):

    fig, axs = plt.subplots(1,1,figsize=figSize)

    cs = axs.contour(X,Y,phi,colors='black')
    #cs = axs.imshow(phi.T,extent=[0, 1, 0, 1], origin='lower',
    #        interpolation='bicubic',cmap='RdBu_r', alpha=1.0,)
    fig.colorbar(cs, ax=axs, orientation='vertical')
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)


def contourPlot2D(X1, Y1, phi1, X2, Y2, phi2, fileName, figSize=(14,7)):

    fig, axs = plt.subplots(1,1,figsize=figSize)

    cs1 = axs.contour(X1,Y1,phi1,levels=np.arange(0,1.1,0.1),colors='blue')
    cs2 = axs.contour(X2,Y2,phi2,levels=np.arange(0,1.1,0.1),colors='blue')
    plt.plot(X1,Y1, marker='.', color='r', linestyle='none')
    plt.plot(X2,Y2, marker='.', color='g', linestyle='none')
    cs1.clabel()
    cs2.clabel()
    plt.xlim(-0.05,2.1)
    plt.ylim(-0.05,1.05)
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)


def contourPlot3D(X, Y, phi, fileName, title, figSize=(14,7)):
    
    phiGrad = np.gradient(phi)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if len(X) == 64:
        rx = 2
    elif len(X) == 128:
        rx = 4
    elif len(X) == 256:
        rx = 8
    else:
        rx = 1

    if len(Y) == 64:
        ry = 2
    elif len(Y) == 128:
        ry = 4
    elif len(Y) == 256:
        ry = 8
    else:
        ry = 1

    ax.plot_surface(X, Y, phi, color='white', rstride=rx, cstride=ry, antialiased=False,linewidth=0.01, edgecolors = '#393536', shade =False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(title+str(len(X)))
    ax.view_init(25, -135)
    #ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
    ax.xaxis._axinfo["grid"]['color'] = 'gray'
    ax.yaxis._axinfo["grid"]['color'] = 'gray'
    ax.zaxis._axinfo["grid"]['color'] = 'gray'
    ax.xaxis._axinfo["grid"]['linestyle'] = (0, (1, 2))
    ax.yaxis._axinfo["grid"]['linestyle'] = (0, (1, 2))
    ax.zaxis._axinfo["grid"]['linestyle'] = (0, (1, 2))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #x = X[:,0]
    #y = Y[0,:]
    fig.savefig(fileName+str(len(X)), bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)


def rmMargine():
    from mpl_toolkits.mplot3d.axis3d import Axis
    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs
        Axis._get_coord_info_old = Axis._get_coord_info  
        Axis._get_coord_info = _get_coord_info_new