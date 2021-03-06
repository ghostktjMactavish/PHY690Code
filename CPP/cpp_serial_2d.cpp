#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>

#define IX(x,y) ((x)+N*(y))

// xxxx
// y
// y
// y

using namespace std;
using namespace std::chrono;

int N   = 41;
int nx  = N;
int ny  = N;
int nt  = 10000;
int nit = 50;

float Lx = 2.0;
float Ly = 2.0;

float dx = Lx/(nx-1);
float dy = Ly/(ny-1);

// Chosen as 1% of the courant number limit for dt
float dt =  0.001;//0.01*(dx+dy); 
float courant = 2.0*dt/dx;

float Re = 10;
float vis = (Lx*1.0)/Re;
float rho = 1.0;


void init(float *u,float *v,float *p,float *b)
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
    for(int i=0;i<nx;i++)
    {
        for(int j=0;j<ny;j++)
        {
            ucpy[IX(i,j)]=u[IX(i,j)];
        }
    }

    return ucpy;
}

void build_b(float *u, float *v,float *p,float *b)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j=1;j<(ny-1);j++)
        {
            b[IX(i,j)] = (1/dt)*((u[IX(i+1,j)]-u[IX(i-1,j)])/(2*dx) + (v[IX(i,j+1)]-v[IX(i,j-1)])/(2*dy)) - std::pow(((u[IX(i+1,j)] - u[IX(i-1,j)])/(2*dx)),2) - std::pow(((v[IX(i,j+1)] - v[IX(i,j-1)])/(2*dy)),2) -2*((u[IX(i,j+1)] - u[IX(i,j-1)])/(2*dy))*((v[IX(i+1,j)] - v[IX(i-1,j)])/(2*dx));
        }
    }
}

void p_poisson(float *u, float *v,float *p,float *b)
{
    float *pn;
    for(int n=0;n<nit;n++)
    {
        pn = tempcpy(p);
        for(int i=1;i<(nx-1);i++)
        {
            for(int j=1;j<(ny-1);j++)
            {
                p[IX(i,j)] = ((pn[IX(i+1,j)]+pn[IX(i-1,j)])*(dy*dy) + (pn[IX(i,j+1)]+pn[IX(i,j-1)])*(dx*dx))/(2*(dx*dx+dy*dy)) - (dx*dx*dy*dy)*b[IX(i,j)]/(2*(dx*dx+dy*dy));
            }
        }
        for(int j = 0;j<ny;j++)
        {
            p[IX(0,j)]    = p[IX(1,j)];
            p[IX(nx-1,j)] = p[IX(nx-2,j)]; 
        }
        for(int i = 0;i<nx;i++)
        {
            p[IX(i,0)]    = p[IX(i,1)];
            p[IX(i,ny-1)] = p[IX(i,ny-2)]; 
        }

        free(pn);
    }
}
void set_u(float *u, float *v,float *p,float *b,float *un,float *vn)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j =1; j<(ny-1);j++)
        {
            u[IX(i,j)] = un[IX(i,j)] - (dt/dx)*un[IX(i,j)]*(un[IX(i,j)]-un[IX(i-1,j)]) - (dt/dy)*vn[IX(i,j)]*(un[IX(i,j)]-un[IX(i,j-1)]) -(dt/(2*rho*dx))*(p[IX(i+1,j)]-p[IX(i-1,j)])+vis*(dt/(dx*dx))*(un[IX(i+1,j)]+un[IX(i-1,j)]-2*un[IX(i,j)]) + vis*(dt/(dy*dy))*(un[IX(i,j+1)]+un[IX(i,j-1)]-2*un[IX(i,j)]) ;
        }
    }
}
void set_v(float *u, float *v,float *p,float *b,float *un,float *vn)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j =1; j<(ny-1);j++)
        {
            v[IX(i,j)] = vn[IX(i,j)] - (dt/dx)*un[IX(i,j)]*(vn[IX(i,j)]-vn[IX(i-1,j)]) - (dt/dy)*vn[IX(i,j)]*(vn[IX(i,j)]-vn[IX(i,j-1)]) -(dt/(2*rho*dx))*(p[IX(i,j+1)]-p[IX(i,j-1)])+vis*(dt/(dx*dx))*(vn[IX(i+1,j)]+vn[IX(i-1,j)]-2*vn[IX(i,j)]) + vis*(dt/(dy*dy))*(vn[IX(i,j+1)]+vn[IX(i,j-1)]-2*vn[IX(i,j)]) ;
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
    float *un, *vn;
    for(int n = 0; n<nt;n++)
    {
        cout << "Iteration : " << n << "/" << nt << endl; 
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
    float *u = (float*)calloc(nx*ny,sizeof(float));
    float *v = (float*)calloc(nx*ny,sizeof(float));
    float *p = (float*)calloc(nx*ny,sizeof(float));
    float *b = (float*)calloc(nx*ny,sizeof(float));

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