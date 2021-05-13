import multiprocessing as mp
from multiprocessing import Lock
import SharedArray
from util_python_parallel_multip_3d import cavity_flow
import numpy as np
import time

nProcs = 4

nx = 64
ny = 64
nz = 64
Lx = 2
Ly = 2
Lz = 2
dx = Lx/(nx-1)
dy = Ly/(ny-1)
dz = Lz/(nz-1)

nt = 10
nit = 500
dt = 0.001*dx

vis = 0.1
rho = 1.0

size = ny // nProcs
remainder = ny % nProcs


def mp_NVS():
    try:
        SharedArray.delete("shm://u")
        SharedArray.delete("shm://v")
        SharedArray.delete("shm://w")
        SharedArray.delete("shm://p")
    except:
        pass
    us = SharedArray.create("shm://u", (nz, ny, nx), dtype=np.float64)
    vs = SharedArray.create("shm://v", (nz, ny, nx), dtype=np.float64)
    ws = SharedArray.create("shm://w", (nz, ny, nx), dtype=np.float64)
    ps = SharedArray.create("shm://p", (nz, ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            for k in range(nz):
                us[i,j,k] = 0
                vs[i,j,k] = 0
                ws[i,j,k] = 0
                ps[i,j,k] = 0

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


        if i == 0:
            new_nz = rows+1
            #print("NT {0}".format(nt))
            u = np.zeros((new_nz, ny, nx), dtype=np.float64)
            v = np.zeros((new_nz, ny, nx), dtype=np.float64)
            w = np.zeros((new_nz, ny, nx), dtype=np.float64)
            p = np.zeros((new_nz, ny, nx), dtype=np.float64)
            for _ in range(nt):
                u, v, w, p = us[start:stop+1, :], vs[start:stop+1, :], ws[start:stop+1, :], ps[start:stop+1, :]
                u, v, w, p = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64), p.astype(np.float64)
                u, v, w, p = cavity_flow(u, v, w, p, vis, rho, nx, ny, new_nz, nit, dt, dx, dy, dz)
                lock.acquire()
                us[start:stop, :], vs[start:stop, :], ws[start:stop, :], ps[start:stop, :] = u[0:-1,:], v[0:-1,:], w[0:-1,:], p[0:-1,:]
                lock.release()
        elif i == nProcs-1:
            new_nz = rows+1
            u = np.zeros((new_nz, ny, nx), dtype=np.float64)
            v = np.zeros((new_nz, ny, nx), dtype=np.float64)
            w = np.zeros((new_nz, ny, nx), dtype=np.float64)
            p = np.zeros((new_nz, ny, nx), dtype=np.float64)
            for _ in range(nt):
                u, v, w, p = us[start-1:stop, :], vs[start-1:stop, :], ws[start-1:stop, :], ps[start-1:stop, :]
                u, v, w, p = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64), p.astype(np.float64)
                u, v, w, p = cavity_flow(u, v, w, p, vis, rho, nx, ny, new_nz, nit, dt, dx, dy, dz)
                lock.acquire()
                us[start:stop, :], vs[start:stop, :], ws[start:stop, :], ps[start:stop, :] = u[1:,:], v[1:,:], w[1:,:], p[1:,:]
                lock.release()
        else:
            new_nz = rows+2
            u = np.zeros((new_nz, ny, nx), dtype=np.float64)
            v = np.zeros((new_nz, ny, nx), dtype=np.float64)
            w = np.zeros((new_nz, ny, nx), dtype=np.float64)
            p = np.zeros((new_nz, ny, nx), dtype=np.float64)
            for _ in range(nt):
                u, v, w, p = us[start-1:stop+1, :], vs[start-1:stop+1, :], ws[start-1:stop+1, :], ps[start-1:stop+1, :]
                u, v, w, p = u.astype(np.float64), v.astype(np.float64), w.astype(np.float64), p.astype(np.float64)
                u, v, w, p = cavity_flow(u, v, w, p, vis, rho, nx, ny, new_nz, nit, dt, dx, dy, dz)
                #print(us[start-1:stop+1, :], "\n")
                #print(u,"\n")
                lock.acquire()
                us[start:stop, :], vs[start:stop, :], ws[start:stop, :], ps[start:stop, :] = u[1:-1,:], v[1:-1,:], w[1:-1,:], p[1:-1,:]
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
    SharedArray.delete("shm://w")
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







