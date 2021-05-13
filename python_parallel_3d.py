import numpy as np
from mpi4py import MPI
import time


def build_b(u,v,w):
    dz2y2 = (dz*dy)**2
    dx2y2 = (dx*dy)**2
    dz2x2 = (dz*dx)**2
    dx_div = (dx2y2+dz2x2+dz2y2)

    uu = ((u[1:-1,1:-1,2:]-u[1:-1,1:-1,:-2])/dx)**2 + ((v[1:-1,2:,1:-1]-v[1:-1,:-2,1:-1])/dy)**2 + ((w[2:,1:-1,1:-1]-w[:-2,1:-1,1:-1])/dz)**2
    uv = 2*((u[1:-1,2:,1:-1]-u[1:-1,:-2,1:-1])/dy)*((v[1:-1,1:-1,2:]-v[1:-1,1:-1,:-2])/dx)
    vw = 2*((v[2:,1:-1,1:-1]-v[:-2,1:-1,1:-1])/dz)*((w[1:-1,2:,1:-1]-w[1:-1,:-2,1:-1])/dy)
    wu = 2*((w[1:-1,1:-1,2:]-w[1:-1,1:-1,:-2])/dx)*((u[2:,1:-1,1:-1]-u[:-2,1:-1,1:-1])/dz)

    tterms = 0.25*rho*((dx*dy*dz)**2)*((u[1:-1,1:-1,2:]-u[1:-1,1:-1,:-2])/dx+(v[1:-1,2:,1:-1]-v[1:-1,:-2,1:-1])/dy+(w[2:,1:-1,1:-1]-w[:-2,1:-1,1:-1])/dz)/(dx_div*dt)
    uterms = 0.125*rho*((dx*dy*dz)**2)*(uu+uv+vw+wu)/(dx_div)
    b = -tterms + uterms
    
    return b


def build_p(u,v,w,pn,b):
    
    dz2y2 = (dz*dy)**2
    dx2y2 = (dx*dy)**2
    dz2x2 = (dz*dx)**2
    dx_div = (dx2y2+dz2x2+dz2y2)
    
    pterms = 0.5*((pn[1:-1,1:-1,2:]+pn[1:-1,1:-1,:-2])*(dz2y2)+(pn[1:-1,2:,1:-1]+pn[1:-1,:-2,1:-1])*(dz2x2)+(pn[2:,1:-1,1:-1]+pn[:-2,1:-1,1:-1])*(dx2y2))/(dx_div)
    
    #tterms = 0.25*rho*((dx*dy*dz)**2)*((u[1:-1,1:-1,2:]-u[1:-1,1:-1,:-2])/dx+(v[1:-1,2:,1:-1]-v[1:-1,:-2,1:-1])/dy+(w[2:,1:-1,1:-1]-w[:-2,1:-1,1:-1])/dz)/(dx_div*dt)
    #uterms = 0.125*rho*((dx*dy*dz)**2)*(uu+uv+vw+wu)/(dx_div)

    p = pterms+b[1:-1,1:-1,1:-1]
    return p


def pressure_poisson(u,v,w,p,b,nit):
    #print("Pressure Poisson",pn.shape)
    
    for q in range(nit):
        pn = np.copy(p)
        p[1:-1,1:-1,1:-1] = build_p(u,v,w,pn,b)
        #Apply BCs
        p[:,:,-1] = p[:,:,-2]
        p[:,:,0]  = p[:,:,1]
        p[:,0,:]  = p[:,1,:]
        p[0,:,:]  = p[1,:,:]
        p[-1,:,:] = p[-2,:,:]
        p[:,-1,:] = 0
        
    return p


def cavity_flow(nt,u,v,w,p,b,vis,nit):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    wn = np.empty_like(w)
    us = np.empty_like(u)
    vs = np.empty_like(v)
    ws = np.empty_like(w)

    for n in range(1):
        un = u.copy()
        vn = v.copy()
        wn = w.copy()

        us = np.sign(un[1:-1,1:-1,1:-1])
        vs = np.sign(vn[1:-1,1:-1,1:-1])
        ws = np.sign(wn[1:-1,1:-1,1:-1])
        

        #Calculating Pressure
        #print(n)
        b[1:-1,1:-1,1:-1] = build_b(un,vn,wn)
        p = pressure_poisson(un,vn,wn,p,b,nit)
        #print(p.shape)
        #Calculating U velocity using NS
        
        pu = -1*(0.5*dt)*(p[1:-1,1:-1,2:]-p[1:-1,1:-1,:-2])/(rho*dx)
        pv = -1*(0.5*dt)*(p[1:-1,2:,1:-1]-p[1:-1,:-2,1:-1])/(rho*dy)
        pw = -1*(0.5*dt)*(p[2:,1:-1,1:-1]-p[:-2,1:-1,1:-1])/(rho*dz)
        
        uij =  un[1:-1,1:-1,1:-1]
        vij =  vn[1:-1,1:-1,1:-1]
        wij =  wn[1:-1,1:-1,1:-1]

        ui1j = un[1:-1,1:-1,2:]
        vi1j = vn[1:-1,1:-1,2:]
        wi1j = wn[1:-1,1:-1,2:]

        uin1j = un[1:-1,1:-1,:-2]
        vin1j = vn[1:-1,1:-1,:-2]
        win1j = wn[1:-1,1:-1,:-2]

        uij1 = un[1:-1,2:,1:-1]
        vij1 = vn[1:-1,2:,1:-1]
        wij1 = wn[1:-1,2:,1:-1]

        uijn1 = un[1:-1,:-2,1:-1]
        vijn1 = vn[1:-1,:-2,1:-1]
        wijn1 = wn[1:-1,:-2,1:-1]

        uik1 = un[2:,1:-1,1:-1]
        vik1 = vn[2:,1:-1,1:-1]
        wik1 = wn[2:,1:-1,1:-1]

        uikn1 = un[:-2,1:-1,1:-1]
        vikn1 = vn[:-2,1:-1,1:-1]
        wikn1 = wn[:-2,1:-1,1:-1]

        velu = -1*dt*(0.5*(1+us)*uij*(uij-uin1j)/dx + 0.5*(-1+us)*uij*(ui1j-uij)/dx + 0.5*(1+vs)*vij*(uij-uijn1)/dy + 0.5*(-1+vs)*vij*(uij1-uij)/dy + 0.5*(1+ws)*wij*(uij-uikn1)/dz + 0.5*(-1+ws)*wij*(uik1-uij)/dz)
        velv = -1*dt*(0.5*(1+us)*uij*(vij-vin1j)/dx + 0.5*(-1+us)*uij*(vi1j-vij)/dx + 0.5*(1+vs)*vij*(vij-vijn1)/dy + 0.5*(-1+vs)*vij*(vij1-vij)/dy + 0.5*(1+ws)*wij*(vij-vikn1)/dz + 0.5*(-1+ws)*wij*(vik1-vij)/dz)
        velw = -1*dt*(0.5*(1+us)*uij*(wij-win1j)/dx + 0.5*(-1+us)*uij*(wi1j-wij)/dx + 0.5*(1+vs)*vij*(wij-wijn1)/dy + 0.5*(-1+vs)*vij*(wij1-wij)/dy + 0.5*(1+ws)*wij*(wij-wikn1)/dz + 0.5*(-1+ws)*wij*(wik1-wij)/dz)

        visu = vis*dt*((ui1j+uin1j-2*uij)/(dx*dx) + (uij1+uijn1-2*uij)/(dy*dy) + (uik1+uikn1-2*uij)/(dz*dz))
        visv = vis*dt*((vi1j+vin1j-2*vij)/(dx*dx) + (vij1+vijn1-2*vij)/(dy*dy) + (vik1+vikn1-2*vij)/(dz*dz))
        visw = vis*dt*((wi1j+win1j-2*wij)/(dx*dx) + (wij1+wijn1-2*wij)/(dy*dy) + (wik1+wikn1-2*wij)/(dz*dz))

        u[1:-1,1:-1,1:-1] = un[1:-1,1:-1,1:-1]+velu+pu+visu
        v[1:-1,1:-1,1:-1] = vn[1:-1,1:-1,1:-1]+velv+pv+visv
        w[1:-1,1:-1,1:-1] = wn[1:-1,1:-1,1:-1]+velw+pw+visw

        v[0,:,: ]  = 0
        v[-1,:,:]  = 0
        v[:,0,: ]  = 0
        v[:,-1,:]  = 0
        v[:,:,0 ]  = 0
        v[:,:,-1]  = 0

        w[0,:,: ]  = 0
        w[-1,:,:]  = 0
        w[:,0,: ]  = 0
        w[:,-1,:]  = 0
        w[:,:,0 ]  = 0
        w[:,:,-1]  = 0

        u[:,-1,:]  = 1
        u[0,:,: ]  = 0
        u[-1,:,:]  = 0
        u[:,:, 0]  = 0
        u[:,:,-1]  = 0
        u[:,0,: ]  = 0
       
    return u,v,w,p


if __name__ == "__main__":
    ## Parallel Declaration
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initial Conditions and Variables
    nx  = 64
    ny  = 64
    nz  = 64
    nt  = 1000
    nit = 500

    count = nz // size                   # number of catchments for each process to analyze
    remainder = nz % size                # extra catchments if n is not a multiple of size

    if rank < remainder:                          # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)                # index of first catchment to analyze
        stop = start + count + 1                  # index of last catchment to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    number_of_rows = stop-start

    if rank==0:
        new_nz = number_of_rows+1
    elif rank==size-1:
        new_nz = number_of_rows+1
    else:
        new_nz = number_of_rows+2

    Re = 100
    vis = 2/Re
    rho = 1.0
    Lx  = 2
    Ly  = 2
    Lz  = 2
    dx  = Lx/(nx-1)
    dy  = Ly/(ny-1)
    dz  = Lz/(nz-1)
    dt  = 0.001
    i_dt = 1/dt
    i_dx = 1/dx
    i_dy = 1/dy
    i_dz = 1/dz
    # x   = np.arange(0,Lx,dx)
    # y   = np.arange(0,Ly,dy)
    # z   = np.arange(0,Lz,dz)
    u   = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    v   = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    w   = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    p   = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    b   = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    un  = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    vn  = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    wn  = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )
    pn  = np.zeros((new_nz,ny,nx))#, dtype = np.float32 )

    if rank == 0:
        start_time = time.time()

    for _ in range(nt):
        u,v,w,p = cavity_flow(nt,u,v,w,p,b,vis,nit)

        if rank != size-1:
            send_data = np.empty([4,ny,nx], dtype=np.float64)
            send_data[0,:,:] = u[-2,:,:]
            send_data[1,:,:] = v[-2,:,:]
            send_data[2,:,:] = w[-2,:,:]
            send_data[3,:,:] = p[-2,:,:]
            comm.send(send_data, dest = rank+1, tag=1)
        if rank != 0:
            recv_data = comm.recv(source = rank-1, tag=1)
            u[0,:,:] = recv_data[0,:,:]
            v[0,:,:] = recv_data[1,:,:]
            w[0,:,:] = recv_data[2,:,:]
            p[0,:,:] = recv_data[3,:,:]

        ## Now sending the first row and reciving it
        if rank !=0:
            send_data = np.empty([4,ny,nx], dtype=np.float64)
            send_data[0,:,:] = u[1,:,:]
            send_data[1,:,:] = v[1,:,:]
            send_data[2,:,:] = w[1,:,:]
            send_data[3,:,:] = p[1,:,:]
            comm.send(send_data, dest = rank-1, tag=2)
        if rank != size-1:
            recv_data = comm.recv(source = rank+1, tag=2)
            u[-1,:,:] = recv_data[0,:,:]
            v[-1,:,:] = recv_data[1,:,:]
            w[-1,:,:] = recv_data[2,:,:]
            p[-1,:,:] = recv_data[3,:,:]

    if rank == 0:
        u = u[0:-1,:,:]
        v = v[0:-1,:,:]
        w = w[0:-1,:,:]
        p = p[0:-1,:,:]
    elif rank == size-1:
        u = u[1:,:,:]
        v = v[1:,:,:]
        w = w[1:,:,:]
        p = p[1:,:,:]
    else:
        u = u[1:-1,:,:]
        v = v[1:-1,:,:]
        w = w[1:-1,:,:]
        p = p[1:-1,:,:]

    recv_array_u = comm.gather(u, root = 0)
    recv_array_v = comm.gather(v, root = 0)
    recv_array_w = comm.gather(w, root = 0)
    recv_array_p = comm.gather(p, root = 0)

    if rank == 0:
        final_u, final_v, final_w, final_p = None, None, None, None
        for i in range(len(recv_array_u)):
            if i==0:
                final_u = recv_array_u[i]
                final_v = recv_array_v[i]
                final_w = recv_array_w[i]
                final_p = recv_array_p[i]
            else:
                final_u = np.vstack((final_u, recv_array_u[i]))
                final_v = np.vstack((final_v, recv_array_v[i]))
                final_w = np.vstack((final_w, recv_array_w[i]))
                final_p = np.vstack((final_p, recv_array_p[i]))
        print("DONE!!!!!!")

    if rank == 0:
        total_time = round(time.time() - start_time,2)
        print('Time taken to run the parallel code is ', total_time, 'seconds')


