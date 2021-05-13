#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <openacc.h>
#include <stdlib.h>

#define IX(x,y) ((x)+N*(y))

// xxxx
// y
// y
// y

using namespace std;
using namespace std::chrono;

const int N   = 41;
const int nx  = N;
const int ny  = N;
const int nt  = 10000;
const int nit = 50;

const float Lx = 2.0;
const float Ly = 2.0;

const float dx = Lx/(nx-1);
const float dy = Ly/(ny-1);

// Chosen as 1% of the courant number limit for dt
const float dt =  0.001;//0.01*(dx+dy);
//const float courant = 2.0*dt/dx;

const float Re = 10;
const float vis = (Lx*1.0)/Re;
const float rho = 1.0;


void init(float * u, float *v, float *p, float *b)
{
    for(int i=0;i<nx;i++)
    {
        for(int j=0;j<ny;j++)
        {
            if(j==(N-1))
                u[IX(i,j)]= 1.0f;
            else
                u[IX(i,j)]= 0.0f;
            v[IX(i,j)]= 0.0f;
            b[IX(i,j)]= 0.0f;
            p[IX(i,j)]= 0.0f;
        }
    }
}

void print_mat(float *u)
{
    for(int j=0;j<ny;j++)
    {
        for(int i=0;i<nx;i++)
        {
            printf("%f ",u[IX(i,j)]);
        }
        printf("\n");
    }
}


float* tempcpy(float *u)
{
    float *ucpy = (float*)calloc(nx*ny,sizeof(float));
#pragma data pcopyin(u[0:nx][0:ny]) copyout(ucpy[0:nx][0:ny])
    {
#pragma acc kernels
    {
    #pragma acc loop independent vector
    for(int i=0;i<nx;i++)
    {
        #pragma acc loop independent vector
        for(int j=0;j<ny;j++)
        {
            ucpy[IX(i,j)]=u[IX(i,j)];
        }
    }
    }
    }

    return ucpy;
}

//#pragma data copy(u[0:nx][0:ny], v[0:nx][0:ny], p[0:nx][0:ny], b[0:nx][0:ny])
void build_b(float *u, float *v,float *p,float *b)
{
#pragma data pcopyin(u[0:nx][0:ny], v[0:nx][0:ny]) copyout(b[0:nx][0:ny])
    {
#pragma acc kernels
    {
    #pragma acc loop independent vector
    for(int i=1;i<(nx-1);i++)
    {
        #pragma acc loop independent vector
        for(int j=1;j<(ny-1);j++)
        {
            b[IX(i,j)] = (1/dt)*((u[IX(i+1,j)]-u[IX(i-1,j)])/(2*dx) + (v[IX(i,j+1)]-v[IX(i,j-1)])/(2*dy)) - std::pow(((u[IX(i+1,j)] - u[IX(i-1,j)])/(2*dx)),2) - std::pow(((v[IX(i,j+1)] - v[IX(i,j-1)])/(2*dy)),2) -2*((u[IX(i,j+1)] - u[IX(i,j-1)])/(2*dy))*((v[IX(i+1,j)] - v[IX(i-1,j)])/(2*dx));
        }
    }
    }
    }
}

//#pragma data copy(u[0:nx][0:ny], v[0:nx][0:ny], p[0:nx][0:ny], b[0:nx][0:ny])
void p_poisson(float *u, float *v,float *p,float *b)
{
    float * __restrict__ pn;

    for(int n=0;n<nit;n++)
    {
        pn = tempcpy(p);

#pragma data pcopyin(pn[0:nx][0:ny]) copyout(p[0:nx][0:ny])
    {
    #pragma acc kernels
        {
        #pragma acc loop independent vector
        for(int i=1;i<(nx-1);i++)
        {
           #pragma acc loop independent vector
            for(int j=1;j<(ny-1);j++)
            {
                p[IX(i,j)] = ((pn[IX(i+1,j)]+pn[IX(i-1,j)])*(dy*dy) + (pn[IX(i,j+1)]+pn[IX(i,j-1)])*(dx*dx))/(2*(dx*dx+dy*dy)) - (dx*dx*dy*dy)*b[IX(i,j)]/(2*(dx*dx+dy*dy));
            }
        }

        #pragma acc loop independent vector
        for(int j = 0;j<ny;j++)
        {
            p[IX(0,j)]    = p[IX(1,j)];
            p[IX(nx-1,j)] = p[IX(nx-2,j)];
        }

        #pragma acc loop independent vector
        for(int i = 0;i<nx;i++)
        {
            p[IX(i,0)]    = p[IX(i,1)];
            p[IX(i,ny-1)] = p[IX(i,ny-2)];
        }
        }
    }
        free(pn);
    }
}

//#pragma data copy(u[0:nx][0:ny], v[0:nx][0:ny], p[0:nx][0:ny], b[0:nx][0:ny], un[0:nx][0:ny], vn[0:nx][0:ny])
void set_u(float *u, float *v,float *p,float *b,float *un,float *vn)
{
#pragma data pcopy(p[0:nx][0:ny], un[0:nx][0:ny], vn[0:nx][0:ny]) copyout(u[0:nx][0:ny])
    {
#pragma acc kernels
    {
    #pragma acc loop independent vector
    for(int i=1;i<(nx-1);i++)
    {
        #pragma acc loop independent vector
        for(int j =1; j<(ny-1);j++)
        {
            u[IX(i,j)] = un[IX(i,j)] - (dt/dx)*un[IX(i,j)]*(un[IX(i,j)]-un[IX(i-1,j)]) - (dt/dy)*vn[IX(i,j)]*(un[IX(i,j)]-un[IX(i,j-1)]) -(dt/(2*rho*dx))*(p[IX(i+1,j)]-p[IX(i-1,j)])+vis*(dt/(dx*dx))*(un[IX(i+1,j)]+un[IX(i-1,j)]-2*un[IX(i,j)]) + vis*(dt/(dy*dy))*(un[IX(i,j+1)]+un[IX(i,j-1)]-2*un[IX(i,j)]) ;
        }
    }
    }
    }
}

//#pragma data copy(u[0:nx][0:ny], v[0:nx][0:ny], p[0:nx][0:ny], b[0:nx][0:ny], un[0:nx][0:ny], vn[0:nx][0:ny])
void set_v(float *u, float *v,float *p,float *b,float *un,float *vn)
{
#pragma data pcopy(p[0:nx][0:ny], vn[0:nx][0:ny]) copyout(v[0:nx][0:ny])
    {
#pragma acc kernels
    {
    #pragma acc loop independent vector
    for(int i=1;i<(nx-1);i++)
    {
        #pragma acc loop independent vector
        for(int j =1; j<(ny-1);j++)
        {
            v[IX(i,j)] = vn[IX(i,j)] - (dt/dx)*un[IX(i,j)]*(vn[IX(i,j)]-vn[IX(i-1,j)]) - (dt/dy)*vn[IX(i,j)]*(vn[IX(i,j)]-vn[IX(i,j-1)]) -(dt/(2*rho*dx))*(p[IX(i,j+1)]-p[IX(i,j-1)])+vis*(dt/(dx*dx))*(vn[IX(i+1,j)]+vn[IX(i-1,j)]-2*vn[IX(i,j)]) + vis*(dt/(dy*dy))*(vn[IX(i,j+1)]+vn[IX(i,j-1)]-2*vn[IX(i,j)]) ;
        }
    }
    }
    }
}
void set_bc(float *u, float *v)
{
    for(int i=0;i<nx;i++)
    {
        u[IX(i,0)]      =   0;
        u[IX(i,nx-1)]   =   1;
        v[IX(i,0)]      =   0;
        v[IX(i,ny-1)]   =   0;
    }

    for(int j=0;j<ny;j++)
    {
        u[IX(0,j)]      =   0;
        u[IX(nx-1,j)]   =   0;
        v[IX(0,j)]      =   0;
        v[IX(ny-1,j)]   =   0;
    }
}


void cavity_flow(float *u, float *v, float *p, float *b)
{
    float * __restrict__ un, * __restrict__ vn;
    for(int n = 0; n<nt;n++)
    {
        //cout << "Iteration : " << n << "/" << nt << endl;
        un = tempcpy(u);
        vn = tempcpy(v);

        build_b(un,vn,p,b);
        p_poisson(un,vn,p,b);
        set_u(u,v,p,b,un,vn);
        set_v(u,v,p,b,un,vn);
        set_bc(u,v);

        //Free temp copies
        free(un);
        free(vn);
    }
}

void save(float *u, float *v,float *p)
{
   ofstream fout;
   ifstream fin;

   fout.open("u.txt");
   for(int j=0;j<ny;j++)
   {
      for(int i=0;i<nx;i++)
      {
         fout << u[IX(i,j)] << " " ;
      }
      fout << endl;
   }
   fout.close();

   fout.open("v.txt");
   for(int j=0;j<ny;j++)
   {
      for(int i=0;i<nx;i++)
      {
         fout << v[IX(i,j)] << " " ;
      }
      fout << endl;
   }
   fout.close();

   fout.open("p.txt");
   for(int j=0;j<ny;j++)
   {
      for(int i=0;i<nx;i++)
      {
         fout << p[IX(i,j)] << " " ;
      }
      fout << endl;
   }
   fout.close();
}


int main()
{
    float * __restrict__ u = (float*)calloc(nx*ny,sizeof(float));
    float * __restrict__ v = (float*)calloc(nx*ny,sizeof(float));
    float * __restrict__ p = (float*)calloc(nx*ny,sizeof(float));
    float * __restrict__ b = (float*)calloc(nx*ny,sizeof(float));

    init(u,v,p,b);
    cout << "Initialization done" << endl;
    // print_mat(u);
    auto start = high_resolution_clock::now();
    cavity_flow(u,v,p,b);
    auto stop  = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start)/1e6;
    cout << "Time Taken to run" << endl;
    cout << duration.count() << endl;

    save(u,v,p);
    cout << "Data Saved" << endl;

    return 0;
}
