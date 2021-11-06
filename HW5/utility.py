import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    return ( 30 * (x * (x - 1) + y * (y - 1)))


def func2(x, y):
    return ( 30 * (x * (x - 1) + y * (y - 1))) - 4. * (np.pi**2) * (x - 1.) * np.sin(2*(np.pi)*y)


def exactSol(x, y):
    return ( 15 * (x * (x - 1) * y * (y - 1)) - np.sin(2*np.pi*y) * ( (np.sinh(2*np.pi*(x-1))) / (np.sinh(2*np.pi))))


def contourPlot(X, Y, phi, fileName='Gauss_Seidel', figSize=(14,7)):

    fig, axs = plt.subplots(1,1,figsize=figSize)

    cs = axs.contour(X,Y,phi,colors='black')
    cs = axs.imshow(phi.T,extent=[0, 1, 0, 1], origin='lower',
            interpolation='bicubic',cmap='RdBu_r', alpha=1.0,)
    fig.colorbar(cs, ax=axs, orientation='vertical')
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)


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
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(-1,1)
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