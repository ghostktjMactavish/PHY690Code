import multiprocessing as mp
from multiprocessing import Lock
import SharedArray
from util_python_parallel_multip_2d import cavity_flow
import numpy as np
import time

nProcs = 4

nx = 256
ny = 256
Lx = 2
Ly = 2
dx = Lx/(nx-1)
dy = Ly/(ny-1)

nt = 10000
nit = 50
dt = 0.001*dx

vis = 0.1
rho = 1.0

size = ny // nProcs
remainder = ny % nProcs


def mp_NVS():
    try:
        SharedArray.delete("shm://u")
        SharedArray.delete("shm://v")
        SharedArray.delete("shm://p")
    except:
        pass
    us = SharedArray.create("shm://u", (ny, nx), dtype=np.float64)
    vs = SharedArray.create("shm://v", (ny, nx), dtype=np.float64)
    ps = SharedArray.create("shm://p", (ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            us[i,j] = 0
            vs[i,j] = 0
            ps[i,j] = 0
            # bs[i,j] = 0

    #us,vs,ps = cavity_flow(us, vs, ps, bs, vis, rho, nx, ny, nit, dt, dx, dy)

    def mps(i):
        if i < remainder:
            start = i*(size+1)
            stop = start + (size+1)
        else:
            start = i*size + remainder
            stop = start + size
        rows = stop-start

        print(i,start,stop,rows)
        print("\n")

        #if i == 0:
         #   new_ny = rows+1
         #   u = np.zeros((new_ny, nx), dtype=np.float64)
         #   v = np.zeros((new_ny, nx), dtype=np.float64)
         #   p = np.zeros((new_ny, nx), dtype=np.float64)
        #elif i == nProcs-1:
        #    new_ny = rows+1
        #    u = np.zeros((new_ny, nx), dtype=np.float64)
        #    v = np.zeros((new_ny, nx), dtype=np.float64)
        #    p = np.zeros((new_ny, nx), dtype=np.float64)
        #else:
        #    new_ny = rows+2
        #    u = np.zeros((new_ny, nx), dtype=np.float64)
        #    v = np.zeros((new_ny, nx), dtype=np.float64)
        #    p = np.zeros((new_ny, nx), dtype=np.float64)

        if i == 0:
            new_ny = rows+1
            #print("NT {0}".format(nt))
            u = np.zeros((new_ny, nx), dtype=np.float64)
            v = np.zeros((new_ny, nx), dtype=np.float64)
            p = np.zeros((new_ny, nx), dtype=np.float64)
            for _ in range(nt):
                u, v, p = us[start:stop+1, :], vs[start:stop+1, :], ps[start:stop+1, :]
                u, v, p = u.astype(np.float64), v.astype(np.float64), p.astype(np.float64)
                u,v,p = cavity_flow(u, v, p, vis, rho, nx, new_ny, nit, dt, dx, dy)
                lock.acquire()
                us[start:stop, :], vs[start:stop, :], ps[start:stop, :] = u[0:-1,:], v[0:-1,:] , p[0:-1,:]
                lock.release()
        elif i == nProcs-1:
            new_ny = rows+1
            u = np.zeros((new_ny, nx), dtype=np.float64)
            v = np.zeros((new_ny, nx), dtype=np.float64)
            p = np.zeros((new_ny, nx), dtype=np.float64)
            for _ in range(nt):
                u, v, p = us[start-1:stop, :], vs[start-1:stop, :], ps[start-1:stop, :]
                u, v, p = u.astype(np.float64), v.astype(np.float64), p.astype(np.float64)
                u,v,p = cavity_flow(u, v, p, vis, rho, nx, new_ny, nit, dt, dx, dy)
                lock.acquire()
                us[start:stop, :], vs[start:stop, :], ps[start:stop, :] = u[1:,:], v[1:,:], p[1:,:]
                lock.release()
        else:
            new_ny = rows+2
            u = np.zeros((new_ny, nx), dtype=np.float64)
            v = np.zeros((new_ny, nx), dtype=np.float64)
            p = np.zeros((new_ny, nx), dtype=np.float64)
            for _ in range(nt):
                u, v, p = us[start-1:stop+1, :], vs[start-1:stop+1, :], ps[start-1:stop+1, :]
                u, v, p = u.astype(np.float64), v.astype(np.float64), p.astype(np.float64)
                u,v,p = cavity_flow(u, v, p, vis, rho, nx, new_ny, nit, dt, dx, dy)
                #print(us[start-1:stop+1, :], "\n")
                #print(u,"\n")
                lock.acquire()
                us[start:stop, :], vs[start:stop, :], ps[start:stop, :] = u[1:-1,:], v[1:-1,:], p[1:-1,:]
                lock.release()
                #print(us[start-1:stop+1, :], "\n")


    processes = []
    lock = Lock()
    for i in range(nProcs):
        process = mp.Process(target=mps, args=(i,))
        print(i, process)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    SharedArray.delete("shm://u")
    SharedArray.delete("shm://v")
    SharedArray.delete("shm://p")


if __name__ == "__main__":
    #SharedArray.delete("shm://u")
    #SharedArray.delete("shm://v")
    #SharedArray.delete("shm://p")
    #SharedArray.delete("shm://b")

    start = time.time()
    mp_NVS()
    end = time.time()

    print("Time taken to run: {0} seconds".format(round(end-start,2)))







