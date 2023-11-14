import processing
import preprocessing
import compute_alpha
import matplotlib.pyplot as plt
import demo_control_polycopie2023
import numpy
import _env

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    # Material = [phi, gamma_p, sigma, rho_0, alpha_h, c_0]
    material = [0.529,7.0 / 5.0,  151429.0, 1.2, 1.37, 340.0]
    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0
    omega = 50
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
    
    def gaussienne(x,sigma,mu):
     return numpy.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

    def A1(omega):
        mu1=numpy.log10(70*2*numpy.pi)
        sigma1=numpy.log10(100*2*numpy.pi)-numpy.log10(70*2*numpy.pi)
        return gaussienne(numpy.log10(omega),sigma1,mu1)

    def A2(omega):
        mu2=numpy.log10(1200*2*numpy.pi)
        sigma2=numpy.log10(1600*2*numpy.pi)-numpy.log10(1000*2*numpy.pi)
        return 0.5*gaussienne(numpy.log10(omega),sigma2,mu2)
    
    def g(x, omega):
        return A2(omega)*numpy.sinc(omega*x/material[-1])*omega/material[-1]+ A1(omega)*numpy.sinc(omega*(x-0.5)/material[-1])*omega/material[-1]
    
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

    # -- this is the function you have written during your project
    import compute_alpha
    Alpha = compute_alpha.compute_alpha(omega, material)
    alpha_rob = Alpha[0] * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
                
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional
    for i in range(M):
        for j in range(N):
            if processing.is_on_robin_boundary(domain_omega[i,j]):
                chi[i,j] = 1
            
    omega = numpy.linspace(100, 4000, 100)
    energie = []
    for elem in omega:
        for j in range(N):
            f_dir[0, j] = g(j/N, elem)
        Alpha = compute_alpha.compute_alpha(elem, material)
        alpha_rob = Alpha[0] * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, elem, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energie.append(demo_control_polycopie2023.compute_objective_function(domain_omega, u, spacestep))

    plt.plot(omega, energie, 'x')
    plt.plot(omega, energie)
    plt.title('Graphe de $J(\chi)$ en fonction de $\omega$')
    plt.xlabel('Fréquence $\omega$')
    plt.ylabel('Énergie $J(\chi)$')
    plt.show()