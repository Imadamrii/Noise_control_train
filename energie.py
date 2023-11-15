import processing
import preprocessing
import compute_alpha
import matplotlib.pyplot as plt
import demo_control_polycopie2023
import numpy
import _env

def run_etude_frequentielle_chi_0(): 
    N = 64  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    # Material = [phi, gamma_p, sigma, rho_0, alpha_h, c_0]
    material = [0.529,7.0 / 5.0,  151429.0, 1.2, 1.37, 340.0]
    c_0 = material[-1]
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    omega = numpy.linspace(2*numpy.pi*20, 2*numpy.pi*1000, 100)
    energie = []
    
    def g(x,omega):
        return (numpy.sin(omega*x/c_0) + numpy.sin(10*omega*x/c_0))*numpy.exp(-((x-0.5)**2)/2)
    """L'enrgie pour  partout 0  """
    for elem in omega:
        chi = preprocessing._set_chi(M, N, x, y)
        chi = preprocessing.set2zero(chi, domain_omega)

        f_dir[:, :] = 0.0
        for j in range(N):
            f_dir[:, j] = g(j/N, elem)  
        
        Alpha = compute_alpha.compute_alpha(elem, material)
        alpha_rob = Alpha[0] * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, elem/c_0, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energie.append(demo_control_polycopie2023.compute_objective_function(domain_omega, u, spacestep))

    plt.plot(omega, energie, marker = 'x', color = 'darkblue')
    plt.plot(omega, energie, color = 'darkblue')
    plt.title('Graphe de $J(\chi)$ en fonction de $\omega$; chi = 0 sur neuman ')
    plt.xlabel('Fréquence $\omega$')
    plt.ylabel('Énergie $J(\chi)$')
    # plt.show()
    filename = 'fig_energy_chi_0_partout.jpg'
    plt.savefig(filename)
    plt.close()


def run_etude_frequentielle_chi_1(): 
    N = 64  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    # Material = [phi, gamma_p, sigma, rho_0, alpha_h, c_0]
    material = [0.529,7.0 / 5.0,  151429.0, 1.2, 1.37, 340.0]
    c_0 = material[-1]
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    omega = numpy.linspace(2*numpy.pi*20, 2*numpy.pi*1000, 100)
    energie = []
    
    def g(x,omega):
        return (numpy.sin(omega*x/c_0) + numpy.sin(10*omega*x/c_0))*numpy.exp(-((x-0.5)**2)/2)
    for elem in omega:
        chi = preprocessing._set_chi(M, N, x, y)
        chi = preprocessing.set2zero(chi, domain_omega)
        for i in range(M):
            for j in range(N):
                if processing.is_on_robin_boundary(domain_omega[i,j]):
                    chi[i,j] = 1
        f_dir[:, :] = 0.0
        for j in range(N):
            f_dir[:, j] = g(j/N, elem)  
        Alpha = compute_alpha.compute_alpha(elem, material)
        alpha_rob = Alpha[0] * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, elem/c_0, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energie.append(demo_control_polycopie2023.compute_objective_function(domain_omega, u, spacestep))
    
    
    plt.plot(omega, energie, marker = 'x', color = 'darkblue')
    plt.plot(omega, energie, color = 'darkblue')
    plt.title('Graphe de $J(\chi)$ en fonction de $\omega$; chi = 1 sur neuman ')
    plt.xlabel('Fréquence $\omega$')
    plt.ylabel('Énergie $J(\chi)$')
    # plt.show()
    filename = 'fig_energy_chi_1_partout.jpg'
    plt.savefig(filename)
    plt.close()




def run_frequentielle_chi_optimise_vobj_donnée(): 
    N = 64  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    # Material = [phi, gamma_p, sigma, rho_0, alpha_h, c_0]
    material = [0.529,7.0 / 5.0,  151429.0, 1.2, 1.37, 340.0]
    c_0 = material[-1]

    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    omega = numpy.linspace(2*numpy.pi*20, 2*numpy.pi*1000, 2)
    energie = []
    
    def g(x,omega):
        return (numpy.sin(omega*x/c_0) + numpy.sin(10*omega*x/c_0))*numpy.exp(-((x-0.5)**2)/2)
    
    for elem in omega:

        chi = preprocessing._set_chi(M, N, x, y)
        chi = preprocessing.set2zero(chi, domain_omega)
        # -- set parameters for optimization
        S = 0  # surface of the fractal
        for i in range(0, M):
            for j in range(0, N):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    S += 1
        V_0 = 1  # initial volume of the domain
        V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
        mu = 1# initial gradient step
        mu1 = 10 ** (-5)  # parameter of the volume functional

        f_dir[:, :] = 0.0
        for j in range(N):
            f_dir[:, j] = g(j/N, elem)  
        Alpha = compute_alpha.compute_alpha(elem, material)[0]
        alpha_rob = Alpha * chi
        # u = processing.solve_helmholtz(domain_omega, spacestep, elem/c_0, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        # -- compute optimization
        energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
        # chi, energy, u, grad = your_optimization_procedure(...)
        chi, energy, u, grad = demo_control_polycopie2023.optimization_procedure(domain_omega, spacestep, elem/c_0, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                            Alpha, mu, chi, V_obj)
        # --- en of optimization


        energie.append(demo_control_polycopie2023.compute_objective_function(domain_omega, u, spacestep))
    
    
    plt.plot(omega, energie, marker = 'x', color = 'darkblue')
    plt.plot(omega, energie, color = 'darkblue')
    plt.title('Graphe de $J(\chi)$ en fonction de $\omega$; optimisé pour v_obj fixé  ')
    plt.xlabel('Fréquence $\omega$')
    plt.ylabel('Énergie $J(\chi)$')
    # plt.show()
    filename = 'fig_energy_chi_optimise.jpg'
    plt.savefig(filename)
    plt.close()




if __name__ == '__main__':
    # run_etude_frequentielle_chi_0()
    # run_etude_frequentielle_chi_1()
    run_frequentielle_chi_optimise_vobj_donnée()


