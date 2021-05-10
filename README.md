# PHY690Code
Code for course PHY690W High Performance Computing
In physics, the Navier – Stokes equations, named after Claude-Louis Navier and George Gabriel Stokes, describe the motion of fluid substances. These equations arise from applying Newton's second law to fluid motion, together with the assumption that the fluid stress is the sum of a diffusing viscous term (proportional to the gradient of velocity) and a pressure term - hence describing viscous flow. The Pressure – Poisson equation (Poisson equation for pressure) is a derived equation to relate the pressure with momentum equation. It has been derived using the continuity equation as constrain for momentum equation. Adding the partial derivative of x – momentum w.r.t. x and the partial derivative of y – momentum w.r.t. y and then applying the continuity equation yields the Pressure – Poisson equation.

## Usage
Set the initial parameters.

## Parameters
nx:      Number of nodes in the x-direction </br>
ny:     Number of nodes in the y-direction </br>
nt:      Number of time steps </br>
nit:     Number of artificial time steps </br>
vis:     Viscosity </br>
rho:     Density </br>
Lx:      Length in the x-direction </br>
Ly:      Length in the y-direction </br>
dx:      Grid spacing in the x-direction </br>
dy:      Grid spacing in the y-direction </br>
dt:      Time-step size </br>
x:       Node x-ordinates </br>
y:       Node y-ordinates </br>
u:       Nodal velocity x-component </br>
v:       Nodal velocoty y-component </br>
p:       Nodal pressure </br>
un:      Time marched velocity x-direction </br>
vn:      Time marched velocity y-direction </br>
pn:      Temporary pressure for calculations </br>
b:       Nodal source term value from pressure </br>

The images below are for the 2-D case. 

### U Velocity 
<p align="center">
  <img width="720" src="imgs/U_VEL_CPP.png" title="U Velocity Cavity Flow">
</p>

### V Velocity
<p align="center">
  <img width="720" src="imgs/V_VEL_CPP.png" title="U Velocity Cavity Flow">
</p>

### Pressure Field 
<p align="center">
  <img width="720" src="imgs/P_VEL_CPP.png" title="U Velocity Cavity Flow">
</p>

