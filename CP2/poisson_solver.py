# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:37:05 2021

@author: osan
"""
import numpy as np
from scipy.fftpack import dst, idst
import matplotlib.pyplot as plt 


#%%
def fst(nx,ny,dx,dy,f):
    
    data = f[1:-1,1:-1]
        
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dst.html
    data = dst(data, axis = 1, type = 1)
    data = dst(data, axis = 0, type = 1)
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
        
    data1 = np.zeros((nx-1,ny-1))

    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    data1 = data/alpha
    
    data1 = idst(data1, axis = 1, type = 1)
    data1 = idst(data1, axis = 0, type = 1)
    
    u = data1/((2.0*nx)*(2.0*ny))
    
    un = np.zeros((nx+1,ny+1))
    un[1:-1,1:-1] = u
    
    return un

def jacobi(nx,ny,dx,dy,f):
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    ii,jj = np.meshgrid(ii,jj,indexing='ij')
    
    den = -2.0/dx**2 - 2.0/dy**2
    omega = 1.0

    un = np.zeros((nx+1,ny+1))
    if ic == 1:
        un[ii,jj] = np.random.randn(nx-1,ny-1)
        
    rt = np.zeros((nx+1,ny+1))
    
    l2_norm_history = []
    l2_norm_n = 1.0
    counter = 0
    
    rt[ii,jj] = f[ii,jj] - (un[ii+1,jj] - 2.0*un[ii,jj] + un[ii-1,jj])/dx**2 - (un[ii,jj+1] - 2.0*un[ii,jj] + un[ii,jj-1])/dy**2
    l2_norm_0 = np.sqrt(np.mean(rt[ii,jj]**2))
    
    while l2_norm_n > TOL:
        if vector:
            rt[ii,jj] = f[ii,jj] - \
                        (un[ii+1,jj] - 2.0*un[ii,jj] + un[ii-1,jj])/dx**2 -\
                        (un[ii,jj+1] - 2.0*un[ii,jj] + un[ii,jj-1])/dy**2
            
            un[ii,jj] = un[ii,jj] + omega*rt[ii,jj]/den
            
        else:
            for j in range(1,nx):
                for i in range(1,ny):
                    rt[i,j] = f[i,j] - \
                             (un[i+1,j] - 2.0*un[i,j] + un[i-1,j])/dx**2 - \
                             (un[i,j+1] - 2.0*un[i,j] + un[i,j-1])/dy**2
            
            for j in range(1,nx):
                for i in range(1,ny):
                    un[i,j] = un[i,j] + omega*rt[i,j]/den
        
        rt[ii,jj] = f[ii,jj] - (un[ii+1,jj] - 2.0*un[ii,jj] + un[ii-1,jj])/dx**2 - (un[ii,jj+1] - 2.0*un[ii,jj] + un[ii,jj-1])/dy**2
        l2_norm = np.sqrt(np.mean(rt[ii,jj]**2))
        
        l2_norm_n = l2_norm/l2_norm_0
        print('%0.2i %0.5e %0.5e' % (counter, l2_norm, l2_norm_n))
        l2_norm_history.append([counter,l2_norm, l2_norm_n])
        counter += 1
        
        if counter > MAX_ITER:
            break
    
    print(f'Number of iterations = {counter}')
    return un, l2_norm_history


def gs(nx,ny,dx,dy,f):
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    ii,jj = np.meshgrid(ii,jj,indexing='ij')
    
    den = -2.0/dx**2 - 2.0/dy**2
    omega = 1.0
    
    un = np.zeros((nx+1,ny+1))
    if ic == 1:
        un[ii,jj] = np.random.randn(nx-1,ny-1)

    rt = np.zeros((nx+1,ny+1))

    l2_norm_history = []
    l2_norm_n = 1.0
    counter = 0
    
    rt[ii,jj] = f[ii,jj] - (un[ii+1,jj] - 2.0*un[ii,jj] + un[ii-1,jj])/dx**2 - (un[ii,jj+1] - 2.0*un[ii,jj] + un[ii,jj-1])/dy**2
    l2_norm_0 = np.sqrt(np.mean(rt[ii,jj]**2))

    while l2_norm_n > TOL:
        for j in range(1,nx):
            for i in range(1,ny):
                rt[i,j] = f[i,j] - \
                         (un[i+1,j] - 2.0*un[i,j] + un[i-1,j])/dx**2 - \
                         (un[i,j+1] - 2.0*un[i,j] + un[i,j-1])/dy**2
  
                un[i,j] = un[i,j] + omega*rt[i,j]/den
        
        rt[ii,jj] = f[ii,jj] - (un[ii+1,jj] - 2.0*un[ii,jj] + un[ii-1,jj])/dx**2 - (un[ii,jj+1] - 2.0*un[ii,jj] + un[ii,jj-1])/dy**2
        l2_norm = np.sqrt(np.mean(rt[ii,jj]**2))
    
        l2_norm_n = l2_norm/l2_norm_0
        print('%0.2i %0.5e %0.5e' % (counter, l2_norm, l2_norm_n))
        l2_norm_history.append([counter,l2_norm, l2_norm_n])
        counter += 1

        if counter > MAX_ITER:
            break
    
    print(f'Number of iterations = {counter}')
    return un, l2_norm_history

def cg(nx,ny,dx,dy,f,ic):
    TOL = 1e-10
    tiny = 1e-12
    MAX_ITER = 1e3
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    ii,jj = np.meshgrid(ii,jj,indexing='ij')
    
    un = np.zeros((nx+1,ny+1))
    if ic == 1:
        un[ii,jj] = np.random.randn(nx-1,ny-1)

    rt = np.zeros((nx+1,ny+1))
    d = np.zeros((nx+1,ny+1))
    
    l2_norm_history = []
    l2_norm_n = 1.0
    counter = 0
    
    rt[ii,jj] = f[ii,jj] - (un[ii+1,jj] - 2.0*un[ii,jj] + un[ii-1,jj])/dx**2 - (un[ii,jj+1] - 2.0*un[ii,jj] + un[ii,jj-1])/dy**2
    l2_norm_0 = np.sqrt(np.mean(rt[ii,jj]**2))
    
    p = np.copy(rt)
     
    while l2_norm_n > TOL:
        d[ii,jj] = (p[ii+1,jj] - 2.0*p[ii,jj] + p[ii-1,jj])/dx**2 + \
            (p[ii,jj+1] - 2.0*p[ii,jj] + p[ii,jj-1])/dy**2
        
        aa = np.sum(rt[ii,jj]**2)
        bb = np.sum(d[ii,jj]*p[ii,jj])
        
        cc = aa/(bb + tiny)
        
        un[ii,jj] = un[ii,jj] + cc*p[ii,jj]
        
        rt[ii,jj] = rt[ii,jj] - cc*d[ii,jj]
        
        ee = np.sum(rt[ii,jj]**2)  
        
        cc = ee/(aa + tiny)
        
        p[ii,jj] = rt[ii,jj] + cc*p[ii,jj]
        
        l2_norm = np.sqrt(np.mean(rt[ii,jj]**2))
    
        l2_norm_n = l2_norm/l2_norm_0
        #print('%0.2i %0.5e %0.5e' % (counter, l2_norm, l2_norm_n))
        l2_norm_history.append([counter,l2_norm, l2_norm_n])
        counter += 1
        
        if counter > MAX_ITER:
            break
        
    print(f'Number of iterations for CG = {counter}')
    return un, l2_norm_history

#%%
def compute_residual(nx, ny, dx, dy, f, u_n):
    r = np.zeros((nx+1, ny+1))
    d2udx2 = np.zeros((nx+1, ny+1))
    d2udy2 = np.zeros((nx+1, ny+1))
    ii = np.arange(1,nx)
    jj = np.arange(1,ny)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    d2udx2[i,j] = (u_n[i+1,j] - 2*u_n[i,j] + u_n[i-1,j])/(dx**2)
    d2udy2[i,j] = (u_n[i,j+1] - 2*u_n[i,j] + u_n[i,j-1])/(dy**2)
    r[i,j] = f[i,j]  - d2udx2[i,j] - d2udy2[i,j]
    
    del d2udx2, d2udy2
    
    return r
    
def restriction(nxf, nyf, nxc, nyc, r):
    ec = np.zeros((nxc+1, nyc+1))
    center = np.zeros((nxc+1, nyc+1))
    grid = np.zeros((nxc+1, nyc+1))
    corner = np.zeros((nxc+1, nyc+1))
    
    ii = np.arange(1,nxc)
    jj = np.arange(1,nyc)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    # grid index for fine grid for the same coarse point
    center[i,j] = 4.0*r[2*i, 2*j]
    
    # E, W, N, S with respect to coarse grid point in fine grid
    grid[i,j] = 2.0*(r[2*i, 2*j+1] + r[2*i, 2*j-1] +
                r[2*i+1, 2*j] + r[2*i-1, 2*j])
    
    # NE, NW, SE, SW with respect to coarse grid point in fine grid
    corner[i,j] = 1.0*(r[2*i+1, 2*j+1] + r[2*i+1, 2*j-1] +
                  r[2*i-1, 2*j+1] + r[2*i-1, 2*j-1])
    
    # restriction using trapezoidal rule
    ec[i,j] = (center[i,j] + grid[i,j] + corner[i,j])/16.0
    
    del center, grid, corner
    
    i = np.arange(0,nxc+1)
    ec[i,0] = r[2*i, 0]
    ec[i,nyc] = r[2*i, nyf]
    
    j = np.arange(0,nyc+1)
    ec[0,j] = r[0, 2*j]
    ec[nxc,j] = r[nxf, 2*j]
    
    return ec

def prolongation(nxc, nyc, nxf, nyf, unc):
    ef = np.zeros((nxf+1, nyf+1))
    ii = np.arange(0,nxc)
    jj = np.arange(0,nyc)
    i,j = np.meshgrid(ii,jj,indexing='ij')
    
    ef[2*i, 2*j] = unc[i,j]
    # east neighnour on fine grid corresponding to coarse grid point
    ef[2*i, 2*j+1] = 0.5*(unc[i,j] + unc[i,j+1])
    # north neighbout on fine grid corresponding to coarse grid point
    ef[2*i+1, 2*j] = 0.5*(unc[i,j] + unc[i+1,j])
    # NE neighbour on fine grid corresponding to coarse grid point
    ef[2*i+1, 2*j+1] = 0.25*(unc[i,j] + unc[i,j+1] + unc[i+1,j] + unc[i+1,j+1])
    
    i = np.arange(0,nxc+1)
    ef[2*i,nyf] = unc[i,nyc]
    
    j = np.arange(0,nyc+1)
    ef[nxf,2*j] = unc[nxc,j]
    
    return ef

def gauss_seidel_mg(nx, ny, dx, dy, f, un, V, solver=1):
    rt = np.zeros((nx+1,ny+1))
    den = -2.0/dx**2 - 2.0/dy**2
    omega = 1.0
    unr = np.copy(un)
    
    if solver == 1:
        for k in range(V):
            for j in range(1,nx):
                for i in range(1,ny):
                    rt[i,j] = f[i,j] - \
                    (unr[i+1,j] - 2.0*unr[i,j] + unr[i-1,j])/dx**2 - \
                    (unr[i,j+1] - 2.0*unr[i,j] + unr[i,j-1])/dy**2
      
                    unr[i,j] = unr[i,j] + omega*rt[i,j]/den
                    
    elif solver == 2:
        ii = np.arange(1,nx)
        jj = np.arange(1,ny)
        i,j = np.meshgrid(ii,jj,indexing='ij')
        
        for k in range(V):
            rt[i,j] = f[i,j] - \
                      (unr[i+1,j] - 2.0*unr[i,j] + unr[i-1,j])/dx**2 - \
                      (unr[i,j+1] - 2.0*unr[i,j] + unr[i,j-1])/dy**2
                      
            unr[i,j] = unr[i,j] + omega*rt[i,j]/den
    
    return unr


def mg_n_solver(f, dx, dy, nx, ny, n_level=4, iprint=False):
    
    tiny = 1e-12
    max_iterations = 500
    v1 = 2
    v2 = 2
    v3 = 2
    tolerance = 1e-10
    solver = 1

    un = np.zeros((nx+1,ny+1))    
    u_mg = []
    f_mg = []    
    
    u_mg.append(un)
    f_mg.append(f)
    
    r = compute_residual(nx, ny, dx, dy, f_mg[0], u_mg[0])
    
    rms = np.linalg.norm(r)/np.sqrt((nx-1)*(ny-1))
    init_rms = np.copy(rms)
    
    if iprint:
        print('%0.2i %0.5e %0.5e' % (0, rms, rms/init_rms))
    
    if nx < 2**n_level:
        print("Number of levels exceeds the possible number.\n")
    
    lnx = np.zeros(n_level, dtype='int')
    lny = np.zeros(n_level, dtype='int')
    ldx = np.zeros(n_level)
    ldy = np.zeros(n_level)
    
    
    # initialize the mesh details at fine level
    lnx[0] = nx
    lny[0] = ny
    ldx[0] = dx
    ldy[0] = dy
    
    for i in range(1,n_level):
        lnx[i] = int(lnx[i-1]/2)
        lny[i] = int(lny[i-1]/2)
        ldx[i] = ldx[i-1]*2
        ldy[i] = ldy[i-1]*2
        
        fc = np.zeros((lnx[i]+1, lny[i]+1))
        unc = np.zeros((lnx[i]+1, lny[i]+1))
        
        u_mg.append(unc)
        f_mg.append(fc)
    
    # allocate matrix for storage at fine level
    # residual at fine level is already defined at global level
    prol_fine = np.zeros((lnx[1]+1, lny[1]+1))    
    
    # temporaty residual which is restricted to coarse mesh error
    # the size keeps on changing
    temp_residual = np.zeros((lnx[1]+1, lny[1]+1))    
        
    
    l2_norm_history = []
    
    # start main iteration loop
    for iteration_count in range(max_iterations):  
        k = 0
        u_mg[k][:,:] = gauss_seidel_mg(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k], v1, solver=solver)
        
        r = compute_residual(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k])
        
        rms = np.linalg.norm(r)/np.sqrt((nx-1)*(ny-1))
        
        l2_norm_history.append([iteration_count+1, rms, rms/init_rms])
        
        if iprint:
            print('%0.2i %0.5e %0.5e' % (iteration_count+1, rms, rms/init_rms))
        
        if rms/init_rms <= tolerance:
            break
        
        for k in range(1,n_level):
            temp_residual = compute_residual(lnx[k-1], lny[k-1], ldx[k-1], ldy[k-1], 
                                                 f_mg[k-1], u_mg[k-1])
                
            f_mg[k] = restriction(lnx[k-1], lny[k-1], lnx[k], lny[k], temp_residual)
            
            # solution at kth level to zero
            u_mg[k][:,:] = 0.0
            
            if k < n_level-1:
                u_mg[k][:,:] = gauss_seidel_mg(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k], v1, solver=solver)
            elif k == n_level-1:
                u_mg[k][:,:] = gauss_seidel_mg(lnx[k], lny[k], ldx[k], ldy[k], f_mg[k], u_mg[k], v2, solver=solver)
        
        for k in range(n_level-1,0,-1):
            prol_fine = prolongation(lnx[k], lny[k], lnx[k-1], lny[k-1], u_mg[k])
            
            ii = np.arange(1,lnx[k-1])
            jj = np.arange(1,lny[k-1])
            i,j = np.meshgrid(ii,jj,indexing='ij')
            
            u_mg[k-1][i,j] = u_mg[k-1][i,j] + prol_fine[i,j]
            
            u_mg[k-1][:,:] = gauss_seidel_mg(lnx[k-1], lny[k-1], ldx[k-1], ldy[k-1], f_mg[k-1], u_mg[k-1], v3, solver=solver)
           
            
    return u_mg[0], l2_norm_history


#%% 
if __name__ == "__main__":
    
    TOL = 1.0e-6
    MAX_ITER = 400
    vector = False
    tiny = 1.0e-12
    
    ic = 0 # [0] Zero initial condition, [1] Random numbers
    ipr = 3 # Different Poisson equation problems
    isolver = 6 # [1] FST, [2] Jacobi, [3] Gauss-Seidel, [4] MG-N, [5] CG, [6] Comparison       
    nx = 64
    ny = 64
    
    if ipr == 1:
        x_l = -1.0
        x_r = 1.0
        y_b = -1.0
        y_t = 1.0
        
        dx = (x_r-x_l)/nx
        dy = (y_t-y_b)/ny
        
        x = np.linspace(x_l, x_r, nx+1)
        y = np.linspace(y_b, y_t, ny+1)
        
        xm, ym = np.meshgrid(x,y, indexing='ij')
        
        ue = (xm**2 - 1.0)*(ym**2 - 1.0)
        f = -2.0*(2.0 - xm**2 - ym**2)

    elif ipr == 2:
        x_l = 0.0
        x_r = 1.0
        y_b = 0.0
        y_t = 1.0
        
        dx = (x_r-x_l)/nx
        dy = (y_t-y_b)/ny
        
        x = np.linspace(x_l, x_r, nx+1)
        y = np.linspace(y_b, y_t, ny+1)
        
        xm, ym = np.meshgrid(x,y, indexing='ij')
        
        km = 16.0
        c1 = (1.0/km)**2
        c2 = -2.0*np.pi**2
        
        f = c2*np.sin(np.pi*xm)*np.sin(np.pi*ym) + \
            c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
            
        ue = np.sin(np.pi*xm)*np.sin(np.pi*ym) + \
              c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
        
    elif ipr == 3:
        x_l = 0.0
        x_r = 1.0
        y_b = 0.0
        y_t = 1.0
        
        dx = (x_r-x_l)/nx
        dy = (y_t-y_b)/ny
        
        x = np.linspace(x_l, x_r, nx+1)
        y = np.linspace(y_b, y_t, ny+1)
        
        xm, ym = np.meshgrid(x,y, indexing='ij')
        
        km = 16.0
        c1 = (1.0/km)**2
        c2 = -2.0*np.pi**2
                                 
        ue = np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
              c1*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)
        
        f = 4.0*c2*np.sin(2.0*np.pi*xm)*np.sin(2.0*np.pi*ym) + \
            c2*np.sin(km*np.pi*xm)*np.sin(km*np.pi*ym)

    elif ipr == 4:
        x_l = 0.0
        x_r = 1.0
        y_b = 0.0
        y_t = 1.0
        
        dx = (x_r-x_l)/nx
        dy = (y_t-y_b)/ny
        
        x = np.linspace(x_l, x_r, nx+1)
        y = np.linspace(y_b, y_t, ny+1)
        
        xm, ym = np.meshgrid(x,y, indexing='ij')
        
        ue = np.sin(np.pi*xm)*np.sin(np.pi*ym) 
        f = -2.0*np.pi*np.pi*np.sin(np.pi*xm)*np.sin(np.pi*ym) 
            
    if isolver == 1:
        un = fst(nx,ny,dx,dy,f)    
    elif isolver == 2:
        un, l2_norm_history = jacobi(nx,ny,dx,dy,f)    
    elif isolver == 3:
        un, l2_norm_history = gs(nx,ny,dx,dy,f)    
    elif isolver == 4:    
        input_data = {}
        input_data['nlevel'] = 4
        input_data['pmax'] = MAX_ITER
        input_data['v1'] = 2 # during restriction
        input_data['v2'] = 2 # bottom-most level
        input_data['v3'] = 2 # during prolongation
        input_data['tolerance'] = TOL
        input_data['solver'] = 1 # [1] : Gauss-Seidel, [2] Jacobi
        un, l2_norm_history = mg_n_solver(f, dx, dy, nx, ny, input_data, iprint=True)  
    elif isolver == 5:
        un, l2_norm_history = cg(nx,ny,dx,dy,f) 
        
    elif isolver == 6:
        un_j, l2_norm_history_j = jacobi(nx,ny,dx,dy,f) 
        un_gs, l2_norm_history_gs = gs(nx,ny,dx,dy,f) 
        un_cg, l2_norm_history_cg = cg(nx,ny,dx,dy,f) 
        
        input_data = {}
        input_data['nlevel'] = 4
        input_data['pmax'] = MAX_ITER
        input_data['v1'] = 2 # during restriction
        input_data['v2'] = 2 # bottom-most level
        input_data['v3'] = 2 # during prolongation
        input_data['tolerance'] = TOL
        input_data['solver'] = 1 # [1] : Gauss-Seidel, [2] Jacobi
        un_mg, l2_norm_history_mg = mg_n_solver(f, dx, dy, nx, ny, input_data, iprint=True)  
        
    if isolver != 6:
        fig, axs = plt.subplots(1,2,figsize=(14,5))
        cs = axs[0].contourf(xm, ym, ue, 60,cmap='jet')
        #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
        fig.colorbar(cs, ax=axs[0], orientation='vertical')
        
        cs = axs[1].contourf(xm, ym, un,60,cmap='jet')
        #cax = fig.add_axes([1.05, 0.25, 0.05, 0.5])
        fig.colorbar(cs, ax=axs[1], orientation='vertical')
        
        axs[0].set_title('Exact')
        axs[1].set_title('Numerical')

        plt.show()
        fig.tight_layout()
    
    
        fig, axs = plt.subplots(1,2,subplot_kw=dict(projection='3d'),figsize=(14,5))

        surf = axs[0].plot_surface(xm, ym, ue, rstride=1, cstride=1, linewidth=0, cmap='jet', alpha=0.8,
                       antialiased=False, shade=False)
        axs[0].set_zlim([0,1.0])

        surf = axs[1].plot_surface(xm, ym, un, rstride=1, cstride=1, linewidth=0, cmap='jet', alpha=0.8,
                       antialiased=False, shade=False)
        axs[1].set_zlim([0,1.0])

        axs[0].set_title('Exact')
        axs[1].set_title('Numerical')

        fig.tight_layout()
        plt.show()
    
        print(np.linalg.norm(un-ue)/np.sqrt((nx*ny)))    
        
        
    if isolver != 1 and isolver != 6:
        l2_norm_history = np.array(l2_norm_history)
        fig, axs = plt.subplots(1,1,figsize=(6,5))
        axs.semilogy(l2_norm_history[:,0], l2_norm_history[:,1], 'k-')
        axs.set_xlabel('$k$')
        axs.set_ylabel('RMSE')
        fig.tight_layout()
        plt.show()
        
     
    elif isolver == 6:
        l2_norm_history_j = np.array(l2_norm_history_j)
        l2_norm_history_gs = np.array(l2_norm_history_gs)
        l2_norm_history_mg = np.array(l2_norm_history_mg)
        l2_norm_history_cg = np.array(l2_norm_history_cg)
        fig, axs = plt.subplots(1,1,figsize=(6,5))
        axs.semilogy(l2_norm_history_j[:,0], l2_norm_history_j[:,2], 'g', label='Jacobi')
        axs.semilogy(l2_norm_history_gs[:,0], l2_norm_history_gs[:,2], 'b', label='Gauss-Seidel')
        axs.semilogy(l2_norm_history_cg[:,0], l2_norm_history_cg[:,2], 'm', label='Conjugate-Gradient')
        axs.semilogy(l2_norm_history_mg[:,0], l2_norm_history_mg[:,2], 'ro-', label='Multigrid')
        axs.legend()
        axs.set_xlabel('$k$')
        axs.set_ylabel('RMSE/RMSE($k_0$)')
        fig.tight_layout()
        plt.show()


    