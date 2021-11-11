#!/usr/bin/env python
# coding: utf-8

# # Solving incompressible heat flow in a cavity

# Let us consider the Stokes equations for the velocity $\mathbf{u}$ and the pressure $p$ of an incompressible fluid
# 
# \begin{align*} 
#     \nabla \cdot \mathbf{u} &= 0,  \\
#      \partial_t \mathbf{u} + \left( \mathbf{u}\cdot\nabla \right)\mathbf{u} + \nabla p - 2\mu \nabla \cdot \mathbf{D}(\mathbf{u}) &= 0,
# \end{align*}
# 
# where  $\mathbf{D}(\mathbf{u}) = \mathrm{sym}(\mathbf{u}) = \frac{1}{2}\left(\nabla \mathbf{u} +  \left( \nabla \mathbf{u} \right)^{\mathrm{T}} \right)$ is the Newtonian fluid's rate of strain tensor and $\mu$ is the viscosity. 
# 
# Moreover, consider a convection-diffusion equation that governs the evolution of the temperature field
# \begin{align*} 
#     \partial_t T +\mathbf{u}\cdot\nabla T - \kappa \Delta T &= 0,
# \end{align*}
# where $\kappa$ is the constant thermal condictivity. 
# 
# The space domain is $\Omega = [0,L] \times [0,1]$, $L > 0$, and the space variable is $\mathbf{x}= \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$. Time domain is $[0,T]$, with $T>0$.
# 
# Boundary conditions for the velocity are 
# \begin{align*} 
#     \mathbf{u}(t,x_0,1) &= \begin{pmatrix} 1 \\ 0 \end{pmatrix}, &&  0 \leq x_0 \leq L \\
#     \mathbf{u}(t,x_0,x_1) &= \begin{pmatrix} 0 \\ 0 \end{pmatrix} && \text{elsewhere on } \partial \Omega,
# \end{align*}
# and for the temperature field we set
# \begin{align*} 
#     T(t,0,x_1) &= T_h, && 0 \leq x_1 \leq 1, \\
#     T(t,L,x_1) &= T_c, && 0 \leq x_1 \leq 1, \\
# \end{align*}
# for all $t \in [0,T]$.
# Initial conditions are
# \begin{align*} 
#     \mathbf{u}(0,\mathbf{x}) &= \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \\
#     T(0,\mathbf{x}) &= T_i.
# \end{align*}
# 

# ## Python packages

# Import the Python packages for use in this notebook.

# We need the finite element method library FEniCS.


import fenics
import dolfin
print(dolfin.__version__)
import ufl
print(ufl.__version__)
import pymor
print(pymor.__version__)
import numpy as np
import matplotlib.pyplot as plt

outputdir = './fig/'

# ## Benchmark parameters

#dynamic_viscosity = 10.
Re = fenics.Constant(500)
#mu = fenics.Constant(dynamic_viscosity)
prandtl_number = 0.71
#Pr = fenics.Constant(prandtl_number)
kappa = fenics.Constant(1./0.71)

# Furthermore the benchmark problem involves hot and cold walls with constant temperatures $T_h$ and $T_c$, respectively, and the initial temperature $T_i$ of the domain.

hot_wall_temperature = 0.5
T_h = fenics.Constant(hot_wall_temperature)
cold_wall_temperature = -0.5
T_c = fenics.Constant(cold_wall_temperature)
initial_temperature = 0.0
T_i = fenics.Constant(initial_temperature)


# ## Structured mesh definition
# Now, define a structured mesh on the rectangle with bottom left corner $(0,0)$ and top right corner $(\bar{x},\bar{y}) = (L,1)$ and ratio $(\bar{x} - 0)/(\bar{y} - 0) = L$.

L = 1.2

if 0:
    N = 8
    mesh = fenics.UnitSquareMesh(N, N)
    ratio = 1.0
elif 1:
    ny = 10
    nx = int(ny*L)#1.0/ratio)
    mesh = fenics.RectangleMesh(fenics.Point(0, 0), 
                                fenics.Point(L, 1.0), nx, ny, "right/left")
else: 
    #unstructured mesh, mshr module needed
    domain = mshr.Rectangle(fenics.Point(0, 0), fenics.Point(L,1.0))
    mesh = mshr.generate_mesh(domain, 8)


fig = fenics.plot(mesh)
plt.title("Mesh")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig(outputdir+'mesh.png')
del fig

hy = 1./ny

#set compiler
from dolfin.cpp.parameter import parameters, Parameters
from dolfin.parameter import ffc_default_parameters

if not parameters.has_parameter_set("form_compiler"):
    parameters.add(ffc_default_parameters())


# ## Mixed finite element function space, test functions, and solution functions

P1 = fenics.FiniteElement('P', mesh.ufl_cell(), 1)
P2 = fenics.VectorElement('P', mesh.ufl_cell(), 2)
mixed_element = fenics.MixedElement([P1, P2,P1])

W = fenics.FunctionSpace(mesh, mixed_element)

# Test functions
psi_p, psi_u, psi_T = fenics.TestFunctions(W)

# Solution functions
w = fenics.Function(W)
p, u, T = fenics.split(w)

# Solution at previous time step
w_n = fenics.Function(W)
p_n, u_n, T_n = fenics.split(w_n)


# ## Time discretization

def initial_condition(constant = True):
    if (constant):
        w_n = fenics.interpolate(fenics.Expression(("0.", "0.", "0.","T_i" ), 
                                 T_i = initial_temperature,element = mixed_element),W)
    else:
        w_n = fenics.interpolate(fenics.Expression(("0.", "0.", "0.",
                                                    "T_h + x[0]*(T_c - T_h)" ),
                                                   T_h = hot_wall_temperature,
                                                   T_c = cold_wall_temperature,
                                                   T_i = initial_temperature,
                                                   element = mixed_element),W)
    #end    
     
    #endif
    return w_n


w_n = initial_condition()
p_n, u_n, T_n = fenics.split(w_n)



fig = fenics.plot(T_n)
plt.colorbar(fig)
plt.title("$T^0$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig(outputdir+'initial_temperature.png')
del fig
plt.clf()  
plt.close() 

# Choose a time step size $\Delta t$and set the time derivative terms.

timestep_size = 0.001
dt = fenics.Constant(timestep_size)
dtinv = fenics.Constant(1./timestep_size)
u_t = (u - u_n)*dtinv
T_t = (T - T_n)*dtinv


# ## Nonlinear variational form

inner, dot, grad, div, sym =     fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
mass = -psi_p*div(u)

momentum = (dot(psi_u, u_t) #mass
            + dot(psi_u, dot(grad(u), u)) # nonlinear term
            - div(psi_u)*p #divergence 
            + 2.*(1./Re)*inner(sym(grad(psi_u)), sym(grad(u)))) #stiffness

energy = psi_T*T_t + dot(grad(psi_T), kappa*grad(T) - T*u)

F = (mass + momentum + energy )*fenics.dx

#MASS = fenics.assemble( (u*psi_u + psi_T*T ) * dtinv * fenics.dx)
#RHS = fenics.assemble((u_n * psi_u + psi_T*T) * dtinv * fenics.dx)

# stationary part of the variational form
F_stationary = (-psi_p*div(u)+
                dot(psi_u, dot(grad(u), u)) - div(psi_u)*p 
                + 2.*(1./Re)*inner(sym(grad(psi_u)), sym(grad(u))) 
                + dot(grad(psi_T), kappa*grad(T) - T*u))*fenics.dx


# ## Boundary conditions

hot_wall = "near(x[0],  0.)" #x=0 
cold_wall = "near(x[0],"+str(L)+")" #x= \bar{x}
top_wall = "near(x[1],  1.)" #y=1
bottom_wall = "near(x[1],  0.)" #y=0
walls = hot_wall + " | " + cold_wall + " | " + bottom_wall
pressure_point = "near(x[0],  0.) & (x[1]<= "+str(hy)+")" #& near(x[1],  0.)


# Define the boundary conditions on the appropriate subspaces.
# Numbering starts at 0 -> W.sub(0) = W_p

W_p = W.sub(0)
# set velocity BC
boundary_conditions = [fenics.DirichletBC(W_p, 0. , pressure_point)]

W_u = W.sub(1)
# set velocity BC
boundary_conditions.extend([fenics.DirichletBC(W_u, (0., 0.), walls), 
                       fenics.DirichletBC(W_u, (1., 0.), top_wall)])

W_T = W.sub(2)
# set temperature BC
boundary_conditions.extend([fenics.DirichletBC(W_T, hot_wall_temperature, hot_wall), 
                             fenics.DirichletBC(W_T, cold_wall_temperature, cold_wall)])


# ## Solve nonlinear variational problem


def plot_w(w, split = False, outdir = './', nt = ''):
    
    if (type(nt)!=type('')):
        nt=str(nt)
    
    if split:
        p, u, T = fenics.split(w.leaf_node())
    else:
        p, u, T  = w.split()   
    
    fig = fenics.plot(u)
    plt.title("Velocity vector field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)  
    plt.savefig(outdir+'velocity'+nt+'.png')
    del fig
    plt.clf()  
    plt.close() 
    
    fig = fenics.plot(p)
    plt.title("Pressure field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)    
    plt.savefig(outdir+'pressure'+nt+'.png')
    del fig
    plt.clf()  
    plt.close() 

    fig = fenics.plot(T)
    plt.title("Temperature field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)    
    plt.savefig(outdir+'temperature'+nt+'.png')
    del fig
    plt.clf()  
    plt.close() 
    
    return




def solve_fom_fenics(reynolds=1., nt = 1, plot = True, outdir = './', constant = False):
    
    # set Reynolds
    Re.assign(reynolds)
    # set initial solutions
    w_n = initial_condition(constant = constant)
    #p_n, u_n, T_n = fenics.split(w_n)
    # build output
    solution_vec = np.zeros((w.leaf_node().vector()[:].size,nt))
    
    t = 0
    for n in range(nt):

        # Update current time
        t += dt

        #solve
        fenics.solve(F == 0, w, boundary_conditions)

        #append solution
        solution_vec[:,n] = w.leaf_node().vector()[:].copy()
        
        # Update previous solution
        w_n.assign(w)
        if plot:
            plot_w(w, nt = n, outdir=outdir)
        #endif
    #end
    
    return solution_vec


solve_fom_fenics(reynolds=1., nt = 3, constant = True, outdir=outputdir)

# ### pyMOR wrapping
from pymor.bindings.fenics import (FenicsVectorSpace, 
                                   FenicsMatrixOperator, 
                                   FenicsOperator, 
                                   FenicsVisualizer)
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

space = FenicsVectorSpace(W)

from pymor.tools.formatsrc import print_source

from pymor.basic import *

op = FenicsOperator(F_stationary, space, space, w, boundary_conditions,
                    parameter_setter=lambda mu: Re.assign(mu['Re'].item()),
                    parameters={'Re':1},
                    solver_options={'inverse': 
                                    {'type': 'newton', 'rtol': 1e-6, 'maxiter':500,
                                    'error_measure': 'residual'}})



rhs = VectorOperator(op.range.zeros())
#rhs = VectorOperator(space.make_array([F]))
#mass = FenicsMatrixOperator(MASS, W, W, name='mass')


# set initial solutions
w_n = initial_condition()
fom_init = VectorOperator(space.make_array([ w_n.leaf_node().vector()[:] ]))

plot_w(w_n, nt=0, outdir = outputdir)

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper

nt = 3
ie_stepper = ImplicitEulerTimeStepper(nt=nt)

# define the instationary model   
fom = InstationaryModel(timestep_size*nt, 
                        fom_init, 
                        op, 
                        rhs,
                        #mass=mass,
                        #num_values=100,
                        time_stepper=ie_stepper,
                        visualizer=FenicsVisualizer(space) )


def solve_fom_pymor(Re=1., rtol=1e-6, return_residuals=True):
    mu = fom.parameters.parse(Re)
    U = fom.solve(mu)
    
    return U


# In[43]:


U_py = solve_fom_pymor()

quit()

plot_w(U_py, )


# In[ ]:




