#!/usr/bin/env python
# coding: utf-8

# # Solving incompressible heat flow in a cavity

# Let us consider the Stokes equations for the velocity $\mathbf{u}$ and the pressure $p$ of an incompressible fluid
# 
# \begin{align*} 
#     \nabla \cdot \mathbf{u} &= 0,  \\
#      \left( \mathbf{u}\cdot\nabla \right)\mathbf{u} + \nabla p - 2\mu \nabla \cdot \mathbf{D}(\mathbf{u}) &= 0,
# \end{align*}
# 
# where  $\mathbf{D}(\mathbf{u}) = \mathrm{sym}(\mathbf{u}) = \frac{1}{2}\left(\nabla \mathbf{u} +  \left( \nabla \mathbf{u} \right)^{\mathrm{T}} \right)$ is the Newtonian fluid's rate of strain tensor and $\mu$ is the viscosity. 
# 
# Moreover, consider a convection-diffusion equation that governs the evolution of the temperature field
# \begin{align*} 
#     \mathbf{u}\cdot\nabla T - \kappa \Delta T &= 0,
# \end{align*}
# where $\kappa$ is the constant thermal condictivity. 
# 
# The domain is $\Omega = [0,L] \times [0,1]$, with $L > 0$, and the space variable is $\mathbf{x}= \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$.
# 
# Boundary conditions for the velocity are 
# 
# \begin{align*} 
#     \mathbf{u}(x_0,1) &= \begin{pmatrix} 1 \\ 0 \end{pmatrix}, &&  0 \leq x_0 \leq L \\
#     \mathbf{u}(x_0,x_1) &= \begin{pmatrix} 0 \\ 0 \end{pmatrix} && \text{elsewhere on } \partial \Omega,
# \end{align*}
# 
# and for the temperature field we set
# 
# \begin{align*} 
#     T(0,x_1) &= T_h, && 0 \leq x_1 \leq 1, \\
#     T(L,x_1) &= T_c, && 0 \leq x_1 \leq 1. \\
# \end{align*}
# 

# ## Python packages

# Import the Python packages for use in this notebook.

# We need the finite element method library FEniCS.

# In[1]:


import fenics


# In[2]:


import dolfin
dolfin.__version__


# In[3]:


import ufl
ufl.__version__


# In[4]:


import pymor
pymor.__version__


# FEniCS has convenient plotting features that don't require us to import `matplotlib`; but using `matplotlib` directly will allow us to annotate the plots.

# In[5]:


import numpy as np
import matplotlib.pyplot as plt


# ## Benchmark parameters

# Set constant Reynolds number and conductivity coefficient Kappa. For each we define a `fenics.Constant` for use in the variational form so that FEniCS can more efficiently compile the finite element code.

# In[6]:


#dynamic_viscosity = 10.
Re = fenics.Constant(500)
#mu = fenics.Constant(dynamic_viscosity)
prandtl_number = 0.71
#Pr = fenics.Constant(prandtl_number)
kappa = fenics.Constant(1./0.71)


# Furthermore the benchmark problem involves hot and cold walls with constant temperatures $T_h$ and $T_c$, respectively.

# In[7]:


hot_wall_temperature = 0.5
T_h = fenics.Constant(hot_wall_temperature)
cold_wall_temperature = -0.5
T_c = fenics.Constant(cold_wall_temperature)


# ## Structured mesh definition
# Now, define a structured mesh on the rectangle with bottom left corner $(0,0)$ and top right corner $(\bar{x},\bar{y}) = (L,1)$ and ratio $(\bar{x} - 0)/(\bar{y} - 0) = L$.

# In[8]:


L = 1.2


# In[9]:


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


# In[10]:


fenics.plot(mesh)
plt.title("Mesh")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()


# In[11]:


hy = 1./ny
hy


# ## Mixed finite element function space, test functions, and solution functions

# Make the mixed finite element. We choose pressure and velocity subspaces analagous to the Taylor-Hood (i.e. P2P1) mixed element [1], but we extend this further with a P1 element for the temperature.

# In[12]:


P1 = fenics.FiniteElement('P', mesh.ufl_cell(), 1)
P2 = fenics.VectorElement('P', mesh.ufl_cell(), 2)

#P2 = fenics.VectorFunctionSpace(mesh, 'P', 2)
#P1 = fenics.FunctionSpace(mesh, 'P', 1)

mixed_element = fenics.MixedElement([P1, P2,P1])


# |Note|
# |----|
# |`fenics.FiniteElement` requires the `mesh.ufl_cell()` argument to determine some aspects of the domain (e.g. that the spatial domain is two-dimensional).|

# Make the mixed finite element function space $\mathbf{W}$, which enumerates the mixed element basis functions for pressure, velocity and temperature on each cell of the mesh.

# In[13]:


W = fenics.FunctionSpace(mesh, mixed_element)


# Make the test functions $\psi_p,\boldsymbol{\psi}_u, \psi_T \in \mathbf{W}$.

# In[14]:


# TestFunctions
psi_p, psi_u, psi_T = fenics.TestFunctions(W)


# In[15]:


type(psi_p), type(psi_u), type(psi_T)


# Make the Heat-Navier-Stokes system solution function $\mathbf{w} \in \mathbf{W}$ and obtain references to its components $p$, $\mathbf{u}, T$.

# In[16]:


#solution functions
w = fenics.Function(W)
p, u, T = fenics.split(w)

#test functions
w_n = fenics.Function(W)
p_n, u_n, T_n = fenics.split(w_n)


# ## Nonlinear variational form

# Multiply equations by suitable test funcions $\psi_p, \boldsymbol{\psi}_u, \psi_T$ and integrate over the domain. We obtain
# 
# \begin{align*} 
#     ({\psi}_p, \nabla \cdot \mathbf{u}) &= 0  \\
#     \left( \boldsymbol{\psi}_u,  ( \mathbf{u}\cdot\nabla) \mathbf{u} \right) -(p, \nabla \cdot \boldsymbol{\psi}_u) + \mu (\nabla \mathbf{u}, \nabla \boldsymbol{\psi}_u)  &=0,
# \end{align*}
# where
# \begin{align*}
# b( \mathbf{u},\mathbf{v},\boldsymbol{\psi}_u) &= \left( ( \mathbf{u}\cdot\nabla) \mathbf{v}, \boldsymbol{\psi}_u \right),
# \end{align*}
# and 
# \begin{align*} 
#     - (\mathbf{u}\cdot\nabla \psi_T, T) + \kappa(\nabla T, \nabla \psi_T) &= 0.
# \end{align*}
# 
# We can write the nonlinear system of equations as
# 
# \begin{equation*}
# \mathbf{F}(\mathbf{w}) = \mathbf{0}.
# \end{equation*}
# 

# In[17]:


inner, dot, grad, div, sym =     fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
mass = -psi_p*div(u)

momentum = dot(psi_u, dot(grad(u), u)) - div(psi_u)*p     + 2.*(1./Re)*inner(sym(grad(psi_u)), sym(grad(u)))

energy = dot(grad(psi_T), kappa*grad(T) - T*u)

F = (mass + momentum + energy )*fenics.dx


# ## Boundary conditions

# We need boundary conditions before we can define a nonlinear variational *problem* (i.e. in this case a boundary value problem).
# For the velocity, we consider the moving top lid 
# \begin{align*}
#     \mathbf{u}(1,x_1) &= 
#     \begin{pmatrix} 1 \\ 0 \end{pmatrix}, 0 \leq x_1 \leq L.
# \end{align*}
# We physically consider *no slip* velocity boundary conditions for the orther boundaries. These manifest as homogeneous Dirichlet boundary conditions. For the temperature boundary conditions, we consider a constant hot temperature on the left wall, a constant cold temperature on the right wall, and adiabatic (i.e. zero heat transfer) conditions on the top and bottom walls. Because the problem's geometry is simple, we can identify the boundaries with the following piece-wise function.
# 
# \begin{align*}
#     T(\mathbf{x}) &= 
#     \begin{cases}
#         T_h , && x_0 = 0, \\
#         T_c , && x_0 = L.
#     \end{cases}
# \end{align*}

# In[18]:


hot_wall = "near(x[0],  0.)" #x=0 

cold_wall = "near(x[0],"+str(L)+")" #x= \bar{x}

top_wall = "near(x[1],  1.)" #y=1

bottom_wall = "near(x[1],  0.)" #y=0

walls = hot_wall + " | " + cold_wall + " | " + bottom_wall


# In[19]:


pressure_point = "near(x[0],  0.) & (x[1]<= "+str(hy)+")" #& near(x[1],  0.)


# Define the boundary conditions on the appropriate subspaces.

# In[20]:


W_p = W.sub(0)
# set velocity BC
boundary_conditions = [fenics.DirichletBC(W_p, 0. , pressure_point)]


# In[21]:


# numbering starts at 0 -> W.sub(0) = W_p
W_u = W.sub(1)
# set velocity BC
boundary_conditions.extend([fenics.DirichletBC(W_u, (0., 0.), walls), 
                       fenics.DirichletBC(W_u, (1., 0.), top_wall)])


# In[22]:


# set temperature BC

W_T = W.sub(2)
# set velocity BC
boundary_conditions.extend([fenics.DirichletBC(W_T, hot_wall_temperature, hot_wall), 
                             fenics.DirichletBC(W_T, cold_wall_temperature, cold_wall)])


# In[23]:


boundary_conditions


# ## Nonlinear variational problem

# Now we have everything we need to solve the variational problem.

# In[25]:


#set compiler
from dolfin.cpp.parameter import parameters, Parameters
from dolfin.parameter import ffc_default_parameters

if not parameters.has_parameter_set("form_compiler"):
    parameters.add(ffc_default_parameters())


# In[26]:


# solve the system for velocity and pressure
fenics.solve(F == 0, w, boundary_conditions)


# In[27]:


dir(F)
F.coefficients()


# |Note|
# |----|
# |`solver.solve` will modify the solution `w`, which means that `u` and `p` will also be modified.|

# Now plot the velocity field, and pressure and temperature.

# In[28]:


def plot_w(w, split = False):
    
    if split:
        p, u, T = fenics.split(w.leaf_node())
    else:
        p, u, T  = w.split()   
    
    fig = fenics.plot(u)
    plt.title("Velocity vector field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)  
    plt.show()

    
    fig = fenics.plot(p)
    plt.title("Pressure field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)    
    plt.show()
    
    fig = fenics.plot(T)
    plt.title("Temperature field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)    
    plt.show()
    
    return


# In[29]:


def solve_fom_fenics(mu=1.):
    Re.assign(mu)
    fenics.solve(F == 0, w, boundary_conditions)
    return w


# In[30]:


w = solve_fom_fenics(mu=5.)


# In[31]:


plot_w(w)


# In[32]:


w_l = fenics.Function(W)
w = solve_fom_fenics(mu=5.)
w_l.leaf_node().vector()[:] = w.leaf_node().vector()[:]
plot_w(w_l)


# In[33]:


w_h = fenics.Function(W)
w = solve_fom_fenics(mu=500.)
w_h.leaf_node().vector()[:] = w.leaf_node().vector()[:]
plot_w(w_h)


# In[34]:


diff = fenics.Function(W)
diff.leaf_node().vector()[:] = w_h.leaf_node().vector()[:] - w_l.leaf_node().vector()[:]
plot_w(diff)


# # ROM with Pymor
# 
# ## Target parameters
#  
# $L, Re, T_c, T_h, \kappa$

# In[35]:


# ### pyMOR wrapping
from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator


# In[36]:


space = FenicsVectorSpace(W)


# In[37]:


from pymor.tools.formatsrc import print_source


# In[38]:


from pymor.basic import *


# In[39]:


op = FenicsOperator(F, space, space, w, boundary_conditions,
                    parameter_setter=lambda mu: Re.assign(mu['Re'].item()),
                    parameters={'Re':1},
                    #parameters={'T_i': 1,'T_c': 1,'T_h':1,'kappa':1,'Re':1},
                    solver_options={'inverse': 
                                    {'type': 'newton', 'rtol': 1e-6, 'maxiter':500,
                                    'error_measure': 'residual'}})


# In[40]:


rhs = VectorOperator(op.range.zeros())


# In[41]:


fom = StationaryModel(op, rhs,
                     visualizer=FenicsVisualizer(space))


# In[42]:


def solve_fom_pymor(Re=1., rtol=1e-6, return_residuals=True):
    mu = fom.parameters.parse([Re])
    UU, data = newton(fom.operator, fom.rhs.as_vector(), 
                     mu=mu, rtol=rtol, return_residuals=return_residuals, 
                     error_measure='residual')
    # fom solution
    U_fe = fenics.Function(W)
    U_fe.leaf_node().vector()[:] = UU.to_numpy().squeeze()#.reshape((UU.to_numpy().size,))
    
    return U_fe


# (n,1) -> (n,)

# In[43]:


reynolds = 500.
U_py = solve_fom_pymor(Re=reynolds) # fom pymor
w = solve_fom_fenics(mu=reynolds) #fom fenics


# In[44]:


# fom solution
plot_w(U_py)


# In[45]:


#plot error
plt.semilogy(np.absolute( w.leaf_node().vector()[:]-U_py.leaf_node().vector()[:] ),'*')


# In[46]:


diff = fenics.Function(W)
diff.leaf_node().vector()[:] = U_py.leaf_node().vector()[:] - w.leaf_node().vector()[:]
plot_w(diff)


# In[47]:


parameter_space = fom.parameters.space((1, 500.))

# ### ROM generation (POD/DEIM)
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import StationaryRBReductor


# In[48]:


# collect snapshots

U = fom.solution_space.empty()
residuals = fom.solution_space.empty()

for mu_val in parameter_space.sample_randomly(10):#mu_list:
    #UU = fom.solve(mu)
    UU, data = newton(fom.operator, fom.rhs.as_vector(), 
                      mu=mu_val, #fom.parameters.parse([mu_val]), 
                      rtol=1e-6, return_residuals=True, 
                      maxiter=500,
                      error_measure='residual')
    U.append(UU)
    residuals.append(data['residuals'])
    # U_fe = fenics.Function(W)
    # U_fe.leaf_node().vector()[:] = UU.to_numpy().reshape((UU.to_numpy().size,))
    # plot_w(U_fe)
    


# In[49]:


U.dim, len(U)


# In[50]:


if 0: #assertion error
    dofs, cb, _ = ei_greedy(residuals, rtol=1e-7)
    ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, 
                                      interpolation_dofs=dofs, triangular=True) 
    
    rb, svals = pod(U, rtol=1e-7)
    fom_ei = fom.with_(operator=ei_op)
    reductor = StationaryRBReductor(fom_ei, rb)
    rom = reductor.reduce()
    # the reductor currently removes all solver_options so we need to add them again
    rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))
else:
    rb = gram_schmidt(U) # pod(U, rtol=1e-7)
    reductor = StationaryRBReductor(fom, rb)
    
    rom = reductor.reduce()
    # the reductor currently removes all solver_options so we need to add them again
    rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))


# In[51]:


rom.operator.solver_options


# In[52]:


# ensure that FFC is not called during runtime measurements
rom.solve(1)


# ## ROM validation

# In[53]:


import time


# In[54]:


errs = []
speedups = []
for mu in parameter_space.sample_randomly(3):
    tic = time.perf_counter()
    U = fom.solve(mu)
    t_fom = time.perf_counter() - tic

    tic = time.perf_counter()
    u_red = rom.solve(mu)
    t_rom = time.perf_counter() - tic

    U_red = reductor.reconstruct(u_red)
    errs.append(((U - U_red).norm() / U.norm())[0])
    speedups.append(t_fom / t_rom)
print(f'Maximum relative ROM error: {max(errs)}')
print(f'Median of ROM speedup: {np.median(speedups)}')


# In[55]:


def solve_rom_pymor(Re=1., rtol=1e-6, return_residuals=True):
    mu = rom.parameters.parse([Re])
    u_red = rom.solve(mu) 
    U_red = reductor.reconstruct(u_red)
    # fom solution
    U_fe = fenics.Function(W)
    U_fe.leaf_node().vector()[:] = U_red.to_numpy().squeeze()#.reshape((UU.to_numpy().size,))
    
    return U_fe


# In[56]:


reynolds = 50.
U_py = solve_rom_pymor(Re=reynolds)
w = solve_fom_fenics(mu=reynolds)


# In[57]:


# visualize reconstructed solution
plot_w(U_py)


# In[58]:


# difference with fenics solution
diff = fenics.Function(W)
diff.leaf_node().vector()[:] = U_py.leaf_node().vector()[:] - w.leaf_node().vector()[:]
plot_w(diff)


# In[ ]:




