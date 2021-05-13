#include<iostream>
#include<blitz/array.h>
#include <cmath>
#include <fstream>
#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace blitz;



Array<double,2> build_up_b(Array<double,2> b, Array<double,2> u, Array<double,2> v, float rho, double dt, double dx, double dy, int nx, int ny){
	b(Range(1,ny-2),Range(1,nx-2)) = (rho*(1 / dt * 
                    ((u(Range(1,ny-2),Range(2,nx-1)) - u(Range(1,ny-2),Range(0,nx-3))) / 
                     (2 * dx) + (v(Range(2,ny-1),Range(1,nx-2)) - v(Range(0,ny-3),Range(1,nx-2))) / (2 * dy)) -
                    ((u(Range(1,ny-2),Range(2,nx-1)) - u(Range(1,ny-2),Range(0,nx-3))) / (2 * dx))*((u(Range(1,ny-2),Range(2,nx-1)) - u(Range(1,ny-2),Range(0,nx-3))) / (2 * dx)) -
                      2 * ((u(Range(2,ny-1),Range(1,nx-2)) - u(Range(0,ny-3),Range(1,nx-2))) / (2 * dy) *
                           (v(Range(1,ny-2),Range(2,nx-1)) - v(Range(1,ny-2),Range(0,nx-3))) / (2 * dx))-
                          ((v(Range(2,ny-1),Range(1,nx-2)) - v(Range(0,ny-3),Range(1,nx-2))) / (2 * dy))*((v(Range(2,ny-1),Range(1,nx-2)) - v(Range(0,ny-3),Range(1,nx-2))) / (2 * dy))));
	return b;
}


Array<double,2> pressure_poisson(Array<double,2> un, Array<double,2> vn, Array<double,2> p, Array<double,2> b, int nit, double dx, double dy, int nx, int ny){

    // pn has all boundary related elemnts after 1st time loop
    // Below loop helps us to achieve pressure terms from boundary to whole surface via differential schemes in time.
    // Indeed we used boundary conditions and  get the whole surface discretely

    for(int q=0;q<nit;q++){
    	// (n-1)th time presssure part is used to calculate nth time pressure.
    	Array<double,2> pn = p;
    	// optimized python code for our PP equation  
    	p(Range(1,ny-2),Range(1,nx-2)) = (((pn(Range(1,ny-2),Range(2,nx-1)) + pn(Range(1,ny-2),Range(0,nx-3))) * dy*dy + 
                          (pn(Range(2,ny-1),Range(1,nx-2)) + pn(Range(0,ny-3),Range(1,nx-2))) * dx*dx) /
                          (2 * (dx*dx + dy*dy)) -
                          dx*dx * dy*dy / (2 * (dx*dx + dy*dy)) * 
                          b(Range(1,ny-2),Range(1,nx-2)));
    	// boundary conditions 
        // utilized backward difference scheme and equated it to 0 
    	p(Range(0,ny-1),Range(nx-1)) = p(Range(0,ny-1),Range(nx-2));               // dp/dx = 0 at x = 2 
    	p(Range(0),Range(0,nx-1)) = p(Range(1),Range(0,nx-1));                     // dp/dy = 0 at y = 0
    	p(Range(0,ny-1),Range(0)) = p(Range(0,ny-1),Range(1));                     // dp/dx = 0 at x = 0                                          
    	p(Range(ny-1),Range(0,nx-1)) = 0;                                          // p = 0 at y = 2
    }
        
    return p;
}



Array<double,2> cavity_flow(Array<double,2> u, Array<double,2> v, Array<double,2> p, Array<double,2> b,float vis, float rho, int nx, int ny, int nit, double dt, double dx, double dy){
 	Array<double,2> un = u;
 	Array<double,2> vn = v;

 	b = build_up_b(b, u, v, rho, dt, dx, dy, nx, ny);
 	p = pressure_poisson(un,vn,p,b, nit, dx, dy, nx, ny);

 	u(Range(1,ny-2),Range(1,nx-2)) = (un(Range(1,ny-2),Range(1,nx-2))-
                     un(Range(1,ny-2),Range(1,nx-2)) * dt / dx *
                    (un(Range(1,ny-2),Range(1,nx-2)) - un(Range(1,ny-2),Range(0,nx-3))) -
                     vn(Range(1,ny-2),Range(1,nx-2)) * dt / dy *
                    (un(Range(1,ny-2),Range(1,nx-2)) - un(Range(0,ny-3),Range(1,nx-2))) -
                     dt / (2 * rho * dx) * (p(Range(1,ny-2),Range(2,nx-1)) - p(Range(1,ny-2),Range(0,nx-3))) +
                     vis * (dt / dx*dx *
                    (un(Range(1,ny-2),Range(2,nx-1)) - 2 * un(Range(1,ny-2),Range(1,nx-2)) + un(Range(1,ny-2),Range(0,nx-3))) +
                     dt / dy*dy *
                    (un(Range(2,ny-1),Range(1,nx-2)) - 2 * un(Range(1,ny-2),Range(1,nx-2)) + un(Range(0,ny-3),Range(1,nx-2)))));

 	v(Range(1,ny-2),Range(1,nx-2)) = (vn(Range(1,ny-2),Range(1,nx-2)) -
                    un(Range(1,ny-2),Range(1,nx-2)) * dt / dx *
                   (vn(Range(1,ny-2),Range(1,nx-2)) - vn(Range(1,ny-2),Range(0,nx-3))) -
                    vn(Range(1,ny-2),Range(1,nx-2)) * dt / dy *
                   (vn(Range(1,ny-2),Range(1,nx-2)) - vn(Range(0,ny-3),Range(1,nx-2))) -
                    dt / (2 * rho * dy) * (p(Range(2,ny-1),Range(1,nx-2)) - p(Range(0,ny-3),Range(1,nx-2))) +
                    vis * (dt / dx*dx *
                   (vn(Range(1,ny-2),Range(2,nx-1)) - 2 * vn(Range(1,ny-2),Range(1,nx-2)) + vn(Range(1,ny-2),Range(0,nx-3))) +
                    dt / dy*dy *
                   (vn(Range(2,ny-1),Range(1,nx-2)) - 2 * vn(Range(1,ny-2),Range(1,nx-2)) + vn(Range(0,ny-3),Range(1,nx-2)))));

 	u(Range(0),Range(0,nx-1)) = 0;
 	u(Range(0,ny-1),Range(0)) = 0;
 	u(Range(0,ny-1),Range(nx-1)) = 0;
 	u(Range(ny-1),Range(0,nx-1)) = 1;                    // set velocity on cavity lid equal to 1

 	v(Range(0),Range(0,nx-1)) = 0;
 	v(Range(ny-1),Range(0,nx-1)) = 0;
 	v(Range(0,ny-1),Range(0)) = 0;
 	v(Range(0,ny-1),Range(nx-1)) = 0;

 	Array<double,2> returned_array(3*ny,nx);
 	returned_array(Range(0,ny-1),Range(0,nx-1)) = u;
 	returned_array(Range(ny,2*ny-1),Range(0,nx-1)) = v;
 	returned_array(Range(2*ny,3*ny-1),Range(0,nx-1)) = p;
 	return returned_array;
}


void save(Array<double,2> final_u, Array<double,2> final_v,Array<double,2> final_p){
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


int main(int argc, char *argv[]){

	// Parameters
	int nx  = 41;                           // Number of nodes in the x-direction 
	int ny  = 41;                           // Number of nodes in the y-direction 
	float Lx  = 2;                           // Length in the x-direction
	float Ly  = 2;                           // Length in the y-direction
	double dx  = Lx/(nx-1);                  // Grid spacing in the x-direction 
	double dy  = Ly/(ny-1);                  // Grid spacing in the y-direction

	int nt  = 10000;                         // Number of time steps
	int nit = 50;                            // Number of artificial time steps
	double dt  = 0.001*dx;                   // time-step size

	float vis = 0.1;                         // Viscosity
	float rho = 1.0;                         // Density

	int new_ny=ny;

	Array<double,2> u(new_ny,nx);
	Array<double,2> v(new_ny,nx);
	Array<double,2> p(new_ny,nx);
	Array<double,2> b(new_ny,nx);

	u(Range(0,new_ny-1),Range(0,nx-1)) = 0;
	v(Range(0,new_ny-1),Range(0,nx-1)) = 0;
	p(Range(0,new_ny-1),Range(0,nx-1)) = 0;
	b(Range(0,new_ny-1),Range(0,nx-1)) = 0;

	auto start_time = high_resolution_clock::now();


	for(int n=0;n<nt;n++){
		if(n%100 == 0)
			cout<<"Iteration : "<<n<<"/"<<nt<<endl;
		Array<double,2> returned_array = cavity_flow(u, v, p, b, vis, rho, nx, new_ny, nit, dt, dx, dy);
		u = returned_array(Range(0,new_ny-1),Range(0,nx-1));
		v = returned_array(Range(new_ny,2*new_ny-1),Range(0,nx-1));
		p = returned_array(Range(2*new_ny,3*new_ny-1),Range(0,nx-1));
	}
		
    save(u,v,p);
   	cout << "Data Saved" << endl;

   	auto stop_time  = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop_time - start_time)/1e6;
	cout<<"Time Taken to run: "<<duration.count()<<" seconds"<<endl;

	return 0;
}

