# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot as plt
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions
import compute_alpha

def compute_gradient_descent(chi, grad, domain, mu):
	(M, N) = numpy.shape(domain)
     
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			a = preprocessing.BelongsInteriorDomain(domain[i + 1, j])
			b = preprocessing.BelongsInteriorDomain(domain[i - 1, j])
			c = preprocessing.BelongsInteriorDomain(domain[i, j + 1])
			d = preprocessing.BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:

				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
			
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]
	return chi

def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = numpy.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = preprocessing.set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = preprocessing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi

        
def integral(chi):
    integral = 0.0
    for i in range(M):
        for j in range(N):
            integral += chi[i,j]*spacestep
    return integral

def projection_finale(chi, V_obj): 
    table = []
    for i in range (M):
         for j in range (N):
              table.append((chi[i,j],(i,j)))

    table= sorted(table)
    chi1=numpy.zeros(M,N)
    for index in range (int(V_obj*N)):

    
    return chi 

def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = numpy.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = preprocessing.set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = preprocessing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi

        
def integral(chi):
    integral = 0.0
    for i in range(M):
        for j in range(N):
            integral += chi[i,j]*spacestep
    return integral

def projection_finale(chi, V_obj): 
    table = []
    for i in range (M):
         for j in range (N):
              table.append((chi[i,j],(i,j)))

    table= sorted(table)
    chi1=numpy.zeros(M,N)
    for index in range (int(V_obj*N)):

    
    return chi 

def compute_objective_function(domain_omega, u, spacestep):

    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """

    energy = 0.0
    M, N = numpy.shape(domain_omega)

    for i in range(M):
        for j in range(N):
                energy += (numpy.abs(u[i, j]) ** 2) * (spacestep ** 2)

    return energy

def optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 10
    epsilon_0 = 10 ** -5
   

    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        # print('1. computing solution of Helmholtz problem, i.e., u')
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        # print('2. computing solution of adjoint problem, i.e., p')
        p = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*numpy.conjugate(u),numpy.zeros((M,N)), f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        # print('3. computing objective function, i.e., energy')
        energy[k] = compute_objective_function(domain_omega, u, spacestep)

        # print('4. computing parametric gradient')
        grad = numpy.zeros((M,N))
        for i in range(M):
            for j in range(N):
                # if processing.is_on_robin_boundary([domain_omega[i,j]]):
                grad[i,j] += - numpy.real(Alpha*u[i,j]*p[i,j])
        print(numpy.linalg.norm(grad))

        #solution helmotz problem 
        

        ene = energy[k]

        while ene >= energy[k] and mu > epsilon_0:
            
            l = 0
            
            # print('    a. computing gradient descent')
            chi = compute_gradient_descent(chi,grad, domain_omega, mu)
            
            # print('    b. computing projected gradient')
            chi = compute_projected(chi, domain_omega, V_obj)
            print(numpy.linalg.norm(chi-chi0))
            # print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = Alpha*chi # Mettre à jour le coefficient alpha_rob pour le nouveau chi_k+1
            u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

            # print('    d. computing objective function, i.e., energy (E)')
            ene = compute_objective_function(domain_omega, u, spacestep)
            
            if ene < energy[k]:
                # The step is increased if the energy decreased
                mu += 0.01
            else:
                # The step is decreased is the energy increased
                mu = mu/ 2
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')
    alpha_rob = Alpha*chi
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi = projection_max(chi)
    return chi, energy, u, grad


def compute_objective_function(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """

    energy = 0.0
    M, N = numpy.shape(domain_omega)

    for i in range(M):
        for j in range(N):
                energy += (numpy.real(u[i, j]) ** 2 + numpy.imag(u[i,j])**2) * (spacestep ** 2)

    return energy


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
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
    
    def g(x, omega):
        return numpy.exp(-((x-0.5)**2)/2)/(numpy.sqrt(2*numpy.pi))
    
    f_dir[:, :] = 0.0
    for j in range(N):
        f_dir[0, j] = g(j/N, omega)
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    material = [0, 0, 0, 0, 0, 0]
    # -- this is the function you have written during your project
    import compute_alpha
    Alpha = compute_alpha.compute_alpha(omega)
    alpha_rob = Alpha[0] * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5 # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    # chi, energy, u, grad = your_optimization_procedure(...)
    chi, energy, u, grad = optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi, V_obj)
    # --- en of optimization
    
    chin = chi.copy()
    un = u.copy()
    
    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')