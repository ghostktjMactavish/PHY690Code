import numpy as np
import cupy as cp
from mpi4py import MPI
import time
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.cm as cm


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Parameters
nx  = 210                          # Number of nodes in the x-direction
ny  = 210                          # Number of nodes in the y-direction
Lx  = 2                            # Length in the x-direction
Ly  = 2                            # Length in the y-direction
dx  = Lx/(nx-1)                    # Grid spacing in the x-direction
dy  = Ly/(ny-1)                    # Grid spacing in the y-direction

nt  = 10                       # Number of time steps
nit = 50                           # Number of artificial time steps
dt  = 0.001*dx                        # time-step size

vis = 0.1                         # Viscosity
rho = 1.0                          # Density


count = ny // size                   # number of catchments for each process to analyze
remainder = ny % size                # extra catchments if n is not a multiple of size


########################### HELPER FUNCTIONS ###################################
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

    pn = cp.empty_like(p)
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
    un = cp.empty_like(u)
    vn = cp.empty_like(v)
    b = cp.zeros((ny, nx), dtype=np.float64)

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

#########################################################################



if rank < remainder:                          # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)                # index of first catchment to analyze
    stop = start + count + 1                  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count
number_of_rows = stop-start

if rank==0:
    new_ny = number_of_rows+1
    u   = cp.zeros((new_ny,nx), dtype=np.float64)
    v   = cp.zeros((new_ny,nx), dtype=np.float64)
    p   = cp.zeros((new_ny,nx), dtype=np.float64)
    b   = cp.zeros((new_ny,nx), dtype=np.float64)
elif rank==size-1:
    new_ny = number_of_rows+1
    u   = cp.zeros((new_ny,nx), dtype=np.float64)
    v   = cp.zeros((new_ny,nx), dtype=np.float64)
    p   = cp.zeros((new_ny,nx), dtype=np.float64)
    b   = cp.zeros((new_ny,nx), dtype=np.float64)
else:
    new_ny = number_of_rows+2
    u   = cp.zeros((new_ny,nx), dtype=np.float64)
    v   = cp.zeros((new_ny,nx), dtype=np.float64)
    p   = cp.zeros((new_ny,nx), dtype=np.float64)
    b   = cp.zeros((new_ny,nx), dtype=np.float64)



## We will make rank=0 processer as master and rest of them as slave
## After performing all the time steps, the data will be sent to rank=0 processor
## Perform Calculations

## Tags defined for internal use
## for sending last row -> 1
## for sending first row -> 2
## for sending full u,v,p -> 14

#recv_data = np.empty([3,nx], dtype=np.float64)

if rank == 0:
    start_time = time.time()

for _ in range(nt):
    u,v,p = cavity_flow(u, v, p, b, vis, rho, nx, new_ny, nit, dt, dx, dy)

    ## Sending the data to other processors
    ## First sending the last row and recieving it

    if rank != size-1:
        send_data = np.empty([3,nx], dtype=np.float64)
        send_data[0,:] = u[-2,:]
        send_data[1,:] = v[-2,:]
        send_data[2,:] = p[-2,:]
        comm.send(send_data, dest = rank+1, tag=1)
    if rank != 0:
        recv_data = comm.recv(source = rank-1, tag=1)
        u[0,:] = recv_data[0,:]
        v[0,:] = recv_data[1,:]
        p[0,:] = recv_data[2,:]

    ## Now sending the first row and reciving it
    if rank !=0:
        send_data = np.empty([3,nx], dtype=np.float64)
        send_data[0,:] = u[1,:]
        send_data[1,:] = v[1,:]
        send_data[2,:] = p[1,:]
        comm.send(send_data, dest = rank-1, tag=2)
    if rank != size-1:
        recv_data = comm.recv(source = rank+1, tag=2)
        u[-1,:] = recv_data[0,:]
        v[-1,:] = recv_data[1,:]
        p[-1,:] = recv_data[2,:]

## all data should be transfered to rank zero processor
## We will use comm.gather to gather data from each processor

if rank == 0:
    u = u[0:-1,:]
    v = v[0:-1,:]
    p = p[0:-1,:]
elif rank == size-1:
    u = u[1:,:]
    v = v[1:,:]
    p = p[1:,:]
else:
    u = u[1:-1,:]
    v = v[1:-1,:]
    p = p[1:-1,:]

recv_array_u = comm.gather(u, root = 0)
recv_array_v = comm.gather(v, root = 0)
recv_array_p = comm.gather(p, root = 0)

#u_cpu = cp.asnumpy(recv_array_u)
#v_cpu = cp.asnumpy(recv_array_v)
#p_cpu = cp.asnumpy(recv_array_p)

if rank == 0:
    final_u, final_v, final_p = None, None, None
    for i in range(len(recv_array_u)):
        if i==0:
            #final_u = u_cpu[i]
            #final_v = v_cpu[i]
            #final_p = p_cpu[i]
            final_u = recv_array_u[i]
            final_v = recv_array_v[i]
            final_p = recv_array_p[i]
        else:
            #final_u = np.vstack((final_u, u_cpu[i]))
            #final_v = np.vstack((final_v, v_cpu[i]))
            #final_p = np.vstack((final_p, p_cpu[i]))
            final_u = np.vstack((final_u, recv_array_u[i]))
            final_v = np.vstack((final_v, recv_array_v[i]))
            final_p = np.vstack((final_p, recv_array_p[i]))
    print("DONE!!!!!!")


# if rank > 0:
#     if rank == size-1:
#         send_data_full = u[1:,:]
#         send_data_full = np.vstack((send_data_full, v[1:,:]))
#         send_data_full = np.vstack((send_data_full, p[1:,:]))
#         comm.Send(send_data_full, dest=0, tag=14)  # send results to process 0
#     else:
#         send_data_full = u[1:-1,:]
#         send_data_full = np.vstack((send_data_full, v[1:-1,:]))
#         send_data_full = np.vstack((send_data_full, p[1:-1,:]))
#         comm.Send(send_data_full, dest=0, tag=14)
# else:
#     final_u = np.copy(u)                             # initialize final results with results from process 0
#     final_u = final_u[0:-1,:]
#     final_v = np.copy(v)
#     final_v = final_v[0:-1,:]
#     final_p = np.copy(p)
#     final_p = final_p[0:-1,:]
#     for i in range(1, size):                         # determine the size of the array to be received from each process
#         # if i < remainder:
#         #     rank_size = count + 1
#         # else:
#         #     rank_size = count
#         #tmp = np.empty((3*rank_size, final_u.shape[1]), dtype=np.float64)  # create empty array to receive results
#         tmp = comm.recv(source=i, tag=14)
#         temp_rows = tmp.shape[0]/3
#         final_u = np.vstack((final_u, tmp[0:temp_rows,:]))
#         final_v = np.vstack((final_v, tmp[temp_rows:2*temp_rows,:]))
#         final_p = np.vstack((final_p, tmp[2*temp_rows:3*temp_rows,:]))

if rank == 0:
    total_time = round(time.time() - start_time,2)
    print('Time taken to run the parallel code is ', total_time, 'seconds')

    # x   = np.arange(0,Lx+dx,dx)
    # y   = np.arange(0,Ly+dy,dy)
    # x,y = np.meshgrid(x,y)
    # with PdfPages('plot_parallel.pdf') as pdf:
    #     plt.contourf(x, y, final_p, alpha=0.5, cmap=cm.viridis)
    #     plt.colorbar()
    #     plt.contour(x, y, final_p, cmap=cm.viridis)
    #     plt.streamplot(x, y, final_u, final_v)
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     pdf.savefig(dpi = 300)
    #     plt.close()


