#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>

#define IX(x,y,z) ((x)+N*(y)+N*N*(z))

// xxxx
// y
// y
// y

using namespace std;
using namespace std::chrono;

int N   = 41;
int nx  = N;
int ny  = N;
int nz  = N;
int nt  = 10;
int nit = 50;

float Lx = 2.0;
float Ly = 2.0;
float Lz = 2.0;

float dx = Lx/(nx-1);
float dy = Ly/(ny-1);
float dz = Lz/(nz-1);

// Chosen as 1% of the courant number limit for dt
float dt =  0.01*(dx+dy);//0.001; 
float courant = 3.0*dt/dx;

float Re = 10;
float vis = (Lx*1.0)/Re;
float rho = 1.0;


void init(float *u,float *v,float *w,float *p,float *b)
{
    for(int i=0;i<nx;i++)
    {
        for(int j=0;j<ny;j++)
        {
            for(int k=0;k<nz;k++)
            {
                if(j==(N-1))
                    u[IX(i,j,k)]= 1.0f;
                else
                    u[IX(i,j,k)]= 0.0f;
                
                v[IX(i,j,k)]= 0.0f;
                w[IX(i,j,k)]= 0.0f;
                b[IX(i,j,k)]= 0.0f;
                p[IX(i,j,k)]= 0.0f;
            }
        }
    }
}

void print_mat(float *u)
{
    for(int j=0;j<ny;j++)
    {
        for(int i=0;i<nx;i++)
        {
            printf("%f ",u[IX(i,j,0)]);
        }
        printf("\n");
    }
}


float* tempcpy(float *u)
{
    float *ucpy = (float*)calloc(nx*ny*nz,sizeof(float));
    for(int i=0;i<nx;i++)
    {
        for(int j=0;j<ny;j++)
        {
            for(int k=0;k<nz;k++)
            {
                ucpy[IX(i,j,k)]=u[IX(i,j,k)];
            }
        }
    }

    return ucpy;
}

void build_b(float *u, float *v,float*w,float *p,float *b)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j=1;j<(ny-1);j++)
        {   
            for(int k=1;k<(nz-1);k++)
            {
                b[IX(i,j,k)] = rho*(dx*dx*dy*dy*dz*dz)*( 
                                (u[IX(i+1,j,k)]-u[IX(i-1,j,k)])/(2*dx) 
                              + (v[IX(i,j+1,k)]-v[IX(i,j-1,k)])/(2*dy)
                              + (w[IX(i,j,k+1)]-w[IX(i,j,k-1)])/(2*dz)
                              ) / ( 2*(dx*dx+dy*dy+dz*dz) * dt)
                              + rho*(dx*dx*dy*dy*dz*dz)*(
                              - std::pow(((u[IX(i+1,j,k)] - u[IX(i-1,j,k)])/(2*dx)),2) 
                              - std::pow(((v[IX(i,j+1,k)] - v[IX(i,j-1,k)])/(2*dy)),2)
                              - std::pow(((w[IX(i,j,k+1)] - w[IX(i,j,k-1)])/(2*dz)),2) 
                              - 2*( (u[IX(i,j+1,k)] - u[IX(i,j-1,k)])/(2*dy) ) * ( (v[IX(i+1,j,k)] - v[IX(i-1,j,k)])/(2*dx) )
                              - 2*( (u[IX(i,j,k+1)] - u[IX(i,j,k-1)])/(2*dz) ) * ( (w[IX(i+1,j,k)] - w[IX(i-1,j,k)])/(2*dx) )
                              - 2*( (v[IX(i,j,k+1)] - v[IX(i,j,k-1)])/(2*dz) ) * ( (w[IX(i,j+1,k)] - w[IX(i,j-1,k)])/(2*dy) )
                              ) / ( 2*(dx*dx+dy*dy+dz*dz) * dt);   
            }
        }
    }
}

void p_poisson(float *u, float *v,float *w,float *p,float *b)
{
    float *pn;
    for(int n=0;n<nit;n++)
    {
        pn = tempcpy(p);
        for(int i=1;i<(nx-1);i++)
        {
            for(int j=1;j<(ny-1);j++)
            {   
                for(int k=1;k<(nz-1);k++)
                {
                    p[IX(i,j,k)] =  (
                                      ( pn[IX(i+1,j,k)] + pn[IX(i-1,j,k)] )*(dy*dy*dz*dz) 
                                    + ( pn[IX(i,j+1,k)] + pn[IX(i,j-1,k)] )*(dx*dx*dz*dz)
                                    + ( pn[IX(i,j,k+1)] + pn[IX(i,j,k-1)] )*(dx*dx*dy*dy)
                                    )   /  ( 2 * ( dx*dx + dy*dy + dz*dz ))
                                    - b[IX(i,j,k)] ;
                }
            }
        }

        //Set BCs

        // j = ny-1
        for(int i = 0;i<nx;i++)
        {   
            for(int k = 0;k<nz;k++)
            {
                p[IX(i,ny-1,k)] = p[IX(i,ny-2,k)];
            } 
        }
        // j = 0
        for(int i = 0;i<nx;i++)
        {   
            for(int k = 0;k<nz;k++)
            {
                p[IX(i,0,k)] = p[IX(i,1,k)];
            } 
        }
        // k = 0
        for(int i = 0;i<nx;i++)
        {   
            for(int j = 0;j<ny;j++)
            {
                p[IX(i,j,0)] = p[IX(i,j,1)];
            } 
        }
        // k = nz-1
        for(int i = 0;i<nx;i++)
        {   
            for(int j = 0;j<ny;j++)
            {
                p[IX(i,j,nz-1)] = p[IX(i,j,nz-2)];
            } 
        }

        // i = 0
        for(int k = 0;k<nz;k++)
        {   
            for(int j = 0;j<ny;j++)
            {
                p[IX(0,j,k)] = p[IX(1,j,k)];
            } 
        }

        // i = nx-1
        for(int k = 0;k<nx;k++)
        {   
            for(int j = 0;j<ny;j++)
            {
                p[IX(nx-1,j,k)] = p[IX(nx-2,j,k)];
            } 
        }

        free(pn);
    }
}
void set_u(float *u, float *v,float *w,float *p,float *b,float *un,float *vn, float *wn)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j=1; j<(ny-1);j++)
        {
            for( int k=1; k<(nz-1);k++)
            {
                u[IX(i,j,k)] = un[IX(i,j,k)] 
                            - (dt/dx)*un[IX(i,j,k)]*(un[IX(i,j,k)]-un[IX(i-1,j,k)])
                            - (dt/dy)*vn[IX(i,j,k)]*(un[IX(i,j,k)]-un[IX(i,j-1,k)])
                            - (dt/dz)*wn[IX(i,j,k)]*(un[IX(i,j,k)]-un[IX(i,j,k-1)])
                            - (dt/(2*rho*dx))*(p[IX(i+1,j,k)]-p[IX(i-1,j,k)])
                            + vis*(dt/(dx*dx))*(un[IX(i+1,j,k)]+un[IX(i-1,j,k)]-2*un[IX(i,j,k)])
                            + vis*(dt/(dy*dy))*(un[IX(i,j+1,k)]+un[IX(i,j-1,k)]-2*un[IX(i,j,k)])
                            + vis*(dt/(dz*dz))*(un[IX(i,j,k+1)]+un[IX(i,j,k-1)]-2*un[IX(i,j,k)]) ;
            }
        }
    }
}
void set_v(float *u, float *v,float *w,float *p,float *b,float *un,float *vn,float *wn)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j=1; j<(ny-1);j++)
        {
            for( int k=1; k<(nz-1);k++)
            {
                v[IX(i,j,k)] = vn[IX(i,j,k)] 
                            - (dt/dx)*un[IX(i,j,k)]*(vn[IX(i,j,k)]-vn[IX(i-1,j,k)])
                            - (dt/dy)*vn[IX(i,j,k)]*(vn[IX(i,j,k)]-vn[IX(i,j-1,k)])
                            - (dt/dz)*wn[IX(i,j,k)]*(vn[IX(i,j,k)]-vn[IX(i,j,k-1)])
                            - (dt/(2*rho*dy))*(p[IX(i,j+1,k)]-p[IX(i,j-1,k)])*10
                            + vis*(dt/(dx*dx))*(vn[IX(i+1,j,k)]+vn[IX(i-1,j,k)]-2*vn[IX(i,j,k)])
                            + vis*(dt/(dy*dy))*(vn[IX(i,j+1,k)]+vn[IX(i,j-1,k)]-2*vn[IX(i,j,k)])
                            + vis*(dt/(dz*dz))*(vn[IX(i,j,k+1)]+vn[IX(i,j,k-1)]-2*vn[IX(i,j,k)]) ;
            }
        }
    }
}

void set_w(float *u, float *v,float *w,float *p,float *b,float *un,float *vn,float *wn)
{
    for(int i=1;i<(nx-1);i++)
    {
        for(int j=1; j<(ny-1);j++)
        {
            for( int k=1; k<(nz-1);k++)
            {
                w[IX(i,j,k)] = wn[IX(i,j,k)] 
                            - (dt/dx)*un[IX(i,j,k)]*(wn[IX(i,j,k)]-wn[IX(i-1,j,k)])
                            - (dt/dy)*vn[IX(i,j,k)]*(wn[IX(i,j,k)]-wn[IX(i,j-1,k)])
                            - (dt/dz)*wn[IX(i,j,k)]*(wn[IX(i,j,k)]-wn[IX(i,j,k-1)])
                            - (dt/(2*rho*dz))*(p[IX(i,j,k+1)]-p[IX(i,j,k-1)])
                            + vis*(dt/(dx*dx))*(wn[IX(i+1,j,k)]+wn[IX(i-1,j,k)]-2*wn[IX(i,j,k)])
                            + vis*(dt/(dy*dy))*(wn[IX(i,j+1,k)]+wn[IX(i,j-1,k)]-2*wn[IX(i,j,k)])
                            + vis*(dt/(dz*dz))*(wn[IX(i,j,k+1)]+wn[IX(i,j,k-1)]-2*wn[IX(i,j,k)]) ;
            }
        }
    }
}


void set_bc(float *u, float *v,float *w)
{
        
        // j = 0
        for(int i = 0;i<nx;i++)
        {   
            for(int k = 0;k<nz;k++)
            {
                u[IX(i,0,k)] = 0;
                v[IX(i,0,k)] = 0;
                w[IX(i,0,k)] = 0;
            } 
        }
        // j = ny-1
        for(int i = 0;i<nx;i++)
        {   
            for(int k = 0;k<nz;k++)
            {
                u[IX(i,ny-1,k)] = 1;
                v[IX(i,ny-1,k)] = 0;
                w[IX(i,ny-1,k)] = 0;
            } 
        }
        // k = 0
        for(int i = 0;i<nx;i++)
        {   
            for(int j = 0;j<ny;j++)
            {
                u[IX(i,j,0)] = 0;
                v[IX(i,j,0)] = 0;
                w[IX(i,j,0)] = 0;
            } 
        }
        // k = nz-1
        for(int i = 0;i<nx;i++)
        {   
            for(int j = 0;j<ny;j++)
            {
                u[IX(i,j,nz-1)] = 0;
                v[IX(i,j,nz-1)] = 0;
                w[IX(i,j,nz-1)] = 0;
            } 
        }

        // i = 0
        for(int k = 0;k<nz;k++)
        {   
            for(int j = 0;j<ny;j++)
            {
                u[IX(0,j,k)] = 0;
                v[IX(0,j,k)] = 0;
                w[IX(0,j,k)] = 0;
            } 
        }

        // i = nx-1
        for(int k = 0;k<nx;k++)
        {   
            for(int j = 0;j<ny;j++)
            {
                u[IX(nx-1,j,k)] = 0;
                v[IX(nx-1,j,k)] = 0;
                w[IX(nx-1,j,k)] = 0;
            } 
        }

}


void cavity_flow(float *u, float *v,float *w, float *p, float *b)
{
    float *un, *vn, *wn;
    for(int n = 0; n<nt;n++)
    {
        cout << "Iteration : " << n << "/" << nt << endl; 
        un = tempcpy(u);
        vn = tempcpy(v);
        wn = tempcpy(w);

        build_b(un,vn,wn,p,b);
        p_poisson(un,vn,wn,p,b);
        set_u(u,v,w,p,b,un,vn,wn);
        set_v(u,v,w,p,b,un,vn,wn);
        set_w(u,v,w,p,b,un,vn,wn);
        set_bc(u,v,w);
        
        //Free temp copies
        free(un);
        free(vn);
        free(wn);
    }
}

void save(float *u, float *v,float *w, float *p)
{
   ofstream fout;
   ifstream fin;

   fout.open("u3d.txt");
   for(int k=0;k<nz;k++)
   {
        for(int j=0;j<ny;j++)
        {
            for(int i=0;i<nx;i++)
            {
                fout << u[IX(i,j,k)] << " " ;
            }
            fout << endl;
        }
        fout << endl;
   }
   fout.close();

   fout.open("v3d.txt");
   for(int k=0;k<nz;k++)
   {
        for(int j=0;j<ny;j++)
        {
            for(int i=0;i<nx;i++)
            {
                fout << v[IX(i,j,k)] << " " ;
            }
            fout << endl;
        }
        fout << endl;
   }
   fout.close();
   
   fout.open("w3d.txt");
   for(int k=0;k<nz;k++)
   {
        for(int j=0;j<ny;j++)
        {
            for(int i=0;i<nx;i++)
            {
                fout << w[IX(i,j,k)] << " " ;
            }
            fout << endl;
        }
        fout << endl;
   }
   fout.close();

   fout.open("p3d.txt");
   for(int k=0;k<nz;k++)
   {
        for(int j=0;j<ny;j++)
        {
            for(int i=0;i<nx;i++)
            {
                fout << p[IX(i,j,k)] << " " ;
            }
            fout << endl;
        }
        fout << endl;
   }
   fout.close();
}


int main()
{
    float *u = (float*)calloc(nx*ny*nz,sizeof(float));
    float *v = (float*)calloc(nx*ny*nz,sizeof(float));
    float *w = (float*)calloc(nx*ny*nz,sizeof(float));
    float *p = (float*)calloc(nx*ny*nz,sizeof(float));
    float *b = (float*)calloc(nx*ny*nz,sizeof(float));

    init(u,v,w,p,b);
    cout << "Initialization done" << endl;
    // print_mat(u);
    auto start = high_resolution_clock::now();
    cavity_flow(u,v,w,p,b);
    auto stop  = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start)/1e6;
    cout << "Time Taken to run" << endl;
    cout << duration.count() << endl;

    save(u,v,w,p);
    cout << "Data Saved" << endl;
    
    return 0;
}