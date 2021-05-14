#include <iostream>
#include <blitz/array.h>
#include <cmath>
#include <fstream>
#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace blitz;

#define R(x,y) Range(x,y)
// (k,j,i) indexing

Array<double,3> build_up_b( Array<double,3> b, Array<double,3> u, Array<double,3> v, Array<double,3> w, float rho, double dt, double dx, double dy, double dz, int nx, int ny, int nz)
{
	b(R(1,nz-2),R(1,ny-2),R(1,nx-2)) = (dx*dx*dy*dy*dz*dz) * (rho*(  
                                            (1/dt) *
                                            (
                                                    ( u(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - u(R(1,nz-2),R(1,ny-2),R(0,nx-3)) ) / (2 * dx) 
                                                +   ( v(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - v(R(1,nz-2),R(0,ny-3),R(1,nx-2)) ) / (2 * dy)
                                                +   ( w(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - w(R(0,nz-3),R(1,ny-2),R(1,nx-2)) ) / (2 * dz)
                                            )
                                       -2*( (u(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - u(R(1,nz-2),R(0,ny-3),R(1,nx-2))) / (2 * dy) )*( (v(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - v(R(1,nz-2),R(1,ny-2),R(0,nx-3)) ) / (2 * dx))
                                       -2*( (w(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - u(R(1,nz-2),R(0,ny-3),R(1,nx-2))) / (2 * dy) )*( (v(R(2,nx-1),R(1,ny-2),R(1,nx-2)) - v(R(0,nz-3),R(1,ny-2),R(1,nx-2)) ) / (2 * dz))
                                       -2*( (w(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - w(R(1,nz-2),R(1,ny-2),R(0,nx-3))) / (2 * dy) )*( (u(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - u(R(0,nz-3),R(1,ny-2),R(1,nx-2)) ) / (2 * dz)) 
                                       -  ( (u(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - u(R(1,nz-2),R(1,ny-2),R(0,nx-3))) / (2 * dx) )*( (u(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - u(R(1,nz-2),R(1,ny-2),R(0,nx-3)) ) / (2 * dx))
                                       -  ( (w(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - w(R(0,nz-3),R(1,ny-2),R(1,nx-2))) / (2 * dz) )*( (w(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - w(R(0,nz-3),R(1,ny-2),R(1,nx-2)) ) / (2 * dz)) 
                                       -  ( (v(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - v(R(1,nz-2),R(0,ny-3),R(1,nx-2))) / (2 * dy) )*( (v(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - v(R(1,nz-2),R(0,ny-3),R(1,nx-2)) ) / (2 * dy))))
                                       / (2*dx*dx+2*dy*dy+2*dz*dz);
	return b;
}


Array<double,3> pressure_poisson(Array<double,3> un, Array<double,3> vn, Array<double,3> wn, Array<double,3> p, Array<double,3> b, int nit, double dx, double dy, double dz, int nx, int ny,int nz){

    // pn has all boundary related elemnts after 1st time loop
    // Below loop helps us to achieve pressure terms from boundary to whole surface via differential schemes in time.
    // Indeed we used boundary conditions and  get the whole surface discretely

    for(int q=0;q<nit;q++){
    	// (n-1)th time presssure part is used to calculate nth time pressure.
    	Array<double,3> pn = p;
    	// optimized python code for our PP equation  
    	p(R(1,nz-2),R(1,ny-2),R(1,nx-2)) = (
                                            (
                                            (pn(R(1,nz-2),R(1,ny-2),R(2,nx-1)) + pn(R(1,nz-2),R(1,ny-2),R(0,nx-3))) * (dy*dy*dz*dz) + 
                                            (pn(R(1,nz-2),R(2,ny-1),R(1,nx-2)) + pn(R(1,nz-2),R(0,ny-3),R(1,nx-2))) * (dx*dx*dz*dz) + 
                                            (pn(R(2,nz-1),R(1,ny-2),R(1,nx-2)) + pn(R(0,nz-3),R(1,ny-2),R(1,nx-2))) * (dx*dx*dy*dy)
                                            ) / (2 * (dx*dx + dy*dy + dz*dz)) 
                                            - (dx*dx * dy*dy*dz*dz) / (2 * (dx*dx + dy*dy+dz*dz)) * b(R(1,nz-2),R(1,ny-2),R(1,nx-2)));
    	// boundary conditions 
        // utilized backward difference scheme and equated it to 0 
    	p(R(0,nz-1),R(0,ny-1),Range(nx-1)) = p(R(0,nz-1),R(0,ny-1),Range(nx-2));
        p(R(0,nz-1),R(0,ny-1),Range(0))    = p(R(0,nz-1),R(0,ny-1),Range(1));
        p(R(0,nz-1),Range(ny-1),R(0,nx-1)) = 0;
        p(R(0,nz-1),Range(0),R(0,nx-1))    = p(R(0,nz-1),Range(1),R(0,nx-1));
        p(Range(nz-1),R(0,ny-1),R(0,nx-1)) = p(Range(nz-2),R(0,ny-1),R(0,nx-1));
        p(Range(0),R(0,ny-1),R(0,nx-1))    = p(Range(1),R(0,ny-1),R(0,nx-1));               
    }
        
    return p;
}



Array<double,3> cavity_flow( Array<double,3> u, Array<double,3> v, Array<double,3> w, Array<double,3> p, Array<double,3> b,
                            float vis, float rho, int nx, int ny, int nz, int nit, double dt, double dx, double dy, double dz)
{
 	Array<double,3> un = u;
 	Array<double,3> vn = v;
    Array<double,3> wn = w;

 	b = build_up_b(b, u, v, w, rho, dt, dx, dy, dz, nx, ny, nz);
 	p = pressure_poisson( un, vn, wn, p, b, nit, dx, dy, dz, nx, ny, nz);

 	u(R(1,nz-2),R(1,ny-2),R(1,nx-2)) = ( 
                                         un(R(1,nz-2),R(1,ny-2),R(1,nx-2))-
                                         un(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( un(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - un(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dx 
                                       - vn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( un(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - un(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dy
                                       - wn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( un(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - un(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dz 
                                       - dt * ( p(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - p(R(1,nz-2),R(1,ny-2),R(0,nx-3)) ) / (2 * rho * dx) 
                                       + vis * (
                                         dt * (un(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - 2 * un(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + un(R(1,nz-2),R(1,ny-2),R(0,nx-3)))/(dx*dx)
                                       + dt * (un(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - 2 * un(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + un(R(1,nz-2),R(0,ny-3),R(1,nx-2)))/(dy*dy)
                                       + dt * (un(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - 2 * un(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + un(R(0,nz-3),R(1,ny-2),R(1,nx-2)))/(dz*dz)
                                       )
                                       );

 	v(R(1,nz-2),R(1,ny-2),R(1,nx-2)) = ( 
                                         vn(R(1,nz-2),R(1,ny-2),R(1,nx-2))-
                                         un(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( vn(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - vn(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dx 
                                       - vn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( vn(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - vn(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dy
                                       - wn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( vn(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - vn(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dz 
                                       - dt * ( p(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - p(R(1,nz-2),R(0,ny-3),R(1,nx-2)) ) / (2 * rho * dy) 
                                       + vis * (
                                         dt * (vn(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - 2 * vn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + vn(R(1,nz-2),R(1,ny-2),R(0,nx-3)))/(dx*dx)
                                       + dt * (vn(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - 2 * vn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + vn(R(1,nz-2),R(0,ny-3),R(1,nx-2)))/(dy*dy)
                                       + dt * (vn(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - 2 * vn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + vn(R(0,nz-3),R(1,ny-2),R(1,nx-2)))/(dz*dz)
                                       )
                                       );

    w(R(1,nz-2),R(1,ny-2),R(1,nx-2)) = ( 
                                         wn(R(1,nz-2),R(1,ny-2),R(1,nx-2))-
                                         un(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( wn(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - wn(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dx 
                                       - vn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( wn(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - wn(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dy
                                       - wn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) * dt *
                                       ( wn(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - wn(R(1,nz-2),R(1,ny-2),R(1,nx-2))) / dz 
                                       - dt * ( p(R(2,nz-1),R(1,ny-2),R(1,ny-2)) - p(R(0,nz-3),R(1,ny-2),R(1,nx-2)) ) / (2 * rho * dz) 
                                       + vis * (
                                         dt * (wn(R(1,nz-2),R(1,ny-2),R(2,nx-1)) - 2 * wn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + wn(R(1,nz-2),R(1,ny-2),R(0,nx-3)))/(dx*dx)
                                       + dt * (wn(R(1,nz-2),R(2,ny-1),R(1,nx-2)) - 2 * wn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + wn(R(1,nz-2),R(0,ny-3),R(1,nx-2)))/(dy*dy)
                                       + dt * (wn(R(2,nz-1),R(1,ny-2),R(1,nx-2)) - 2 * wn(R(1,nz-2),R(1,ny-2),R(1,nx-2)) + wn(R(0,nz-3),R(1,ny-2),R(1,nx-2)))/(dz*dz)
                                       )
                                       );


 	u(R(0,nz-1),R(0,ny-1),Range(nx-1)) = 0;
    u(R(0,nz-1),R(0,ny-1),Range(0))    = 0;
    u(R(0,nz-1),Range(ny-1),R(0,nx-1)) = 1;
    u(R(0,nz-1),Range(0),R(0,nx-1))    = 0;
    u(Range(nz-1),R(0,ny-1),R(0,nx-1)) = 0;
    u(Range(0),R(0,ny-1),R(0,nx-1))    = 0;

 	v(R(0,nz-1),R(0,ny-1),Range(nx-1)) = 0;
    v(R(0,nz-1),R(0,ny-1),Range(0))    = 0;
    v(R(0,nz-1),Range(ny-1),R(0,nx-1)) = 0;
    v(R(0,nz-1),Range(0),R(0,nx-1))    = 0;
    v(Range(nz-1),R(0,ny-1),R(0,nx-1)) = 0;
    v(Range(0),R(0,ny-1),R(0,nx-1))    = 0;

    w(R(0,nz-1),R(0,ny-1),Range(nx-1)) = 0;
    w(R(0,nz-1),R(0,ny-1),Range(0))    = 0;
    w(R(0,nz-1),Range(ny-1),R(0,nx-1)) = 0;
    w(R(0,nz-1),Range(0),R(0,nx-1))    = 0;
    w(Range(nz-1),R(0,ny-1),R(0,nx-1)) = 0;
    w(Range(0),R(0,ny-1),R(0,nx-1))    = 0;

 	Array<double,3> returned_array(nz,4*ny,nx);
 	returned_array(R(0,nz-1),R(0,ny-1),R(0,nx-1))       = u;
 	returned_array(R(0,nz-1),R(ny,2*ny-1),R(0,nx-1))    = v;
 	returned_array(R(0,ny-1),R(2*ny,3*ny-1),R(0,nx-1))  = w;
    returned_array(R(0,ny-1),R(3*ny,4*ny-1),R(0,nx-1))  = p;
 	return returned_array;
}

//Check this function

void save(Array<double,3> final_u, Array<double,3> final_v,Array<double,3> final_w,Array<double,3> final_p)
{
	ofstream fout;
    ifstream fin;

    fout.open("u.txt");
    fout<<final_u;
    fout.close();

    fout.open("v.txt");
    fout<<final_v;
    fout.close();

    fout.open("p.txt");
    fout<<final_p;
    fout.close();
}


int main()
{

	// Parameters
	int nx  = 41;                           // Number of nodes in the x-direction 
	int ny  = 41;                           // Number of nodes in the y-direction
    int nz  = 41;                           // Number of nodes in the z-direction 
	float Lx  = 2;                           // Length in the x-direction
	float Ly  = 2;                           // Length in the y-direction
    float Lz  = 2;                           // Length in the z-direction
	double dx  = Lx/(nx-1);                  // Grid spacing in the x-direction 
	double dy  = Ly/(ny-1);                  // Grid spacing in the y-direction
    double dz  = Lz/(nz-1);                  // Grid spacing in the z-direction

	int nt  = 10;                         // Number of time steps
	int nit = 50;                            // Number of artificial time steps
	double dt  = 0.001*dx;                   // time-step size

	float vis = 0.1;                         // Viscosity
	float rho = 1.0;                         // Density

	int new_ny=ny;

	Array<double,3> u(nz,ny,nx);
	Array<double,3> v(nz,ny,nx);
    Array<double,3> w(nz,ny,nx);
	Array<double,3> p(nz,ny,nx);
	Array<double,3> b(nz,ny,nx);

	u(R(0,nz-1),R(0,ny-1),R(0,nx-1)) = 0;
	v(R(0,nz-1),R(0,ny-1),R(0,nx-1)) = 0;
    w(R(0,nz-1),R(0,ny-1),R(0,nx-1)) = 0;
	p(R(0,nz-1),R(0,ny-1),R(0,nx-1)) = 0;
	b(R(0,nz-1),R(0,ny-1),R(0,nx-1)) = 0;

	auto start_time = high_resolution_clock::now();

    // Running the Code
	for(int n=0;n<nt;n++)
    {
		//if(n%100 == 0)
			cout<<"Iteration : "<<n<<"/"<<nt<<endl;
		
        Array<double,3> returned_array = cavity_flow(u, v, w, p, b, vis, rho, nx, ny, nz, nit, dt, dx, dy, dz);
		u = returned_array(R(0,nz-1),R(0,ny-1),R(0,nx-1));
		v = returned_array(R(0,nz-1),R(ny,2*ny-1),R(0,nx-1));
        w = returned_array(R(0,nz-1),R(2*ny,3*ny-1),R(0,nx-1));
		p = returned_array(R(0,nz-1),R(3*ny,4*ny-1),R(0,nx-1));
	}
		
    save(u,v,w,p);
   	cout << "Data Saved" << endl;

   	auto stop_time  = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop_time - start_time)/1e6;
	cout<<"Time Taken to run: "<<duration.count()<<" seconds"<<endl;

	return 0;
}