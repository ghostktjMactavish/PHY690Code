import numpy as np 
from cavity_flow_NS import cavity_flow
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

nt  = 10000                        # Number of time steps
nit = 50                           # Number of artificial time steps
dt  = 0.001*dx                        # time-step size

vis = 0.1                         # Viscosity
rho = 1.0                          # Density


count = ny // size                   # number of catchments for each process to analyze
remainder = ny % size                # extra catchments if n is not a multiple of size

if rank < remainder:                          # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)                # index of first catchment to analyze
    stop = start + count + 1                  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count
number_of_rows = stop-start

if rank==0:
    new_ny = number_of_rows+1
    u   = np.zeros((new_ny,nx), dtype=np.float64)
    v   = np.zeros((new_ny,nx), dtype=np.float64)
    p   = np.zeros((new_ny,nx), dtype=np.float64)
    b   = np.zeros((new_ny,nx), dtype=np.float64)
elif rank==size-1:
    new_ny = number_of_rows+1
    u   = np.zeros((new_ny,nx), dtype=np.float64)
    v   = np.zeros((new_ny,nx), dtype=np.float64)
    p   = np.zeros((new_ny,nx), dtype=np.float64)
    b   = np.zeros((new_ny,nx), dtype=np.float64)
else:
    new_ny = number_of_rows+2
    u   = np.zeros((new_ny,nx), dtype=np.float64)
    v   = np.zeros((new_ny,nx), dtype=np.float64)
    p   = np.zeros((new_ny,nx), dtype=np.float64)
    b   = np.zeros((new_ny,nx), dtype=np.float64)



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
        u[-1,:] = recv_data[0,:]
        v[-1,:] = recv_data[1,:]
        p[-1,:] = recv_data[2,:]

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

if rank == 0:
    final_u, final_v, final_p = None, None, None
    for i in range(len(recv_array_u)):
        if i==0:
            final_u = recv_array_u[i]
            final_v = recv_array_v[i]
            final_p = recv_array_p[i]
        else:
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


