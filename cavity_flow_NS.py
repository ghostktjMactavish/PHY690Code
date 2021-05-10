import numpy as np


def build_up_b(b,u,v, rho, dt, dx, dy):
    
     # Below procedure is optimized python code, or in short I used array operations to avoid any kind of loop hustles which may take years to compile :p

    b[1:-1, 1:-1] = (rho*(1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b


def pressure_poisson(u,v,p,b, nit, dx, dy):

    pn = np.empty_like(p)
    pn = p.copy()
    # pn has all boundary related elemnts after 1st time loop
    # Below loop helps us to achieve pressure terms from boundary to whole surface via differential schemes in time.
    # Indeed we used boundary conditions and  get the whole surface discretely

    for q in range(nit):

        # (n-1)th time presssure part is used to calculate nth time pressure.
        pn = p.copy()

        # optimized python code for our PP equation  
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])
    
      # boundary conditions 
      # utilized backward difference scheme and equated it to 0 
      
        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2  
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
        
    return p


def cavity_flow(u, v, p, b, vis, rho, nx, ny, nit, dt, dx, dy):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx), dtype=np.float64)

    un = u.copy()
    vn = v.copy()
    b = build_up_b(b, u, v, rho, dt, dx, dy)
    p = pressure_poisson(un,vn,p,b, nit, dx, dy)
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                     un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     vis * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                   (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                   (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    vis * (dt / dx**2 *
                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 1    # set velocity on cavity lid equal to 1
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0

    return u,v,p 