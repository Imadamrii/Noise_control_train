import demo_control_polycopie2023
# Python packages
import matplotlib.pyplot as plt
import numpy
import os
import scipy

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions
import compute_alpha
c_0=340
beta=numpy.linspace(0,1,20)

# ----------------------------------------------------------------------
# -- Fell free to modify the function call in this cell.
# ----------------------------------------------------------------------
# -- set parameters of the geometry
N = 64  # number of points along x-axis
M = 2 * N  # number of points along y-axis
level = 3 # level of the fractal
spacestep = 1.0 / N  # mesh size

# Material = [phi, gamma_p, sigma, rho_0, alpha_h, c_0]
material = [0.70, 7.0/5.0, 140000.0,1.2,1.02,340.0]

# -- set parameters of the partial differential equation
#kx = -1.0
#ky = -1.0
#wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
#wavenumber = 10.0

# ----------------------------------------------------------------------
# -- Do not modify this cell, these are the values that you will be assessed against.
# ----------------------------------------------------------------------
# --- set coefficients of the partial differential equation
beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

# -- set right hand sides of the partial differential equation
f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

# -- set geometry of domain
domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

# ----------------------------------------------------------------------
# -- Fell free to modify the function call in this cell.
# ----------------------------------------------------------------------
# -- define boundary conditions
# planar wave defined on top

omega = 2*numpy.pi*100
wavenumber = omega/material[-1]

# planar wave defined on top
c_0 = material[-1]

def g(x,omega):
        return (2*numpy.sin(omega*x/c_0) 
                + numpy.sin((37.5*omega-1250)*x/c_0))*numpy.exp(-((x-0.5)**2)/2)


f_dir[:, :] = 0.0
for j in range(N):
    f_dir[:, j] = g(j/N, omega)
# spherical wave defined on top
#f_dir[:, :] = 0.0
#f_dir[0, int(N/2)] = 10.0

# -- initialize
alpha_rob[:, :] = - wavenumber * 1j


# -- this is the function you have written during your project
import compute_alpha
Alpha = compute_alpha.compute_alpha(omega, material)[0]
print('Voici alpha : ', Alpha)


# -- set parameters for optimization
S = 0  # surface of the fractal
for i in range(0, M):
    for j in range(0, N):
        if domain_omega[i, j] == _env.NODE_ROBIN:
            S += 1
V_0 = 1  # initial volume of the domain

mu = 5# initial gradient step
mu1 = 10 ** (-5)  # parameter of the volume functional

# ----------------------------------------------------------------------
# -- Do not modify this cell, these are the values that you will be assessed against.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# -- Fell free to modify the function call in this cell.
# ----------------------------------------------------------------------
# -- compute optimization
energy = numpy.zeros((100+1, 1), dtype=numpy.float64)



def betaenergy(V_obj):
    # -- define material density matrix
        # -- set parameters for optimization
    c_0=340
    mu = 5# initial gradient step

    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    print(V_obj)
    alpha_rob = Alpha * chi
    chi, energy, u, grad = demo_control_polycopie2023.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi, V_obj)
    
    return energy[-1][0]

# print(betaenergy(0.4))
# beta_values= [betaenergy(v) for v in beta]
# plt.plot(beta, beta_values)
# plt.show()


def cost_function(V_0):
    cost = V_0 + betaenergy(V_0)
    return cost 

# Set the bounds for beta to be between 0 and 1
bounds = [(0, 1)]

# Callback function to store the intermediate values
beta_values = []
cost_values = []

def store_intermediate_values(beta):
    beta_values.append(beta)
    cost_values.append(cost_function(beta))

# Set the bounds for beta to be between 0 and 1
bounds = [(0, 1)]

# Perform the minimization with the specified bounds
options = {
    'maxiter': 10,  # Maximum number of iterations
    'disp': True,    # Display convergence messages
}

beta_opt = scipy.optimize.minimize(
    cost_function, 
    x0=[0.2], 
    bounds=[(0, 1)], 
    callback=store_intermediate_values, 
    tol=1e-2,       # Increased tolerance
    options=options
)

# Plotting
plt.plot(beta_values, cost_values, marker='o')
plt.xlabel('Beta')
plt.ylabel('Cost')
plt.title('Optimization of Cost Function')
plt.grid(True)
plt.show()



