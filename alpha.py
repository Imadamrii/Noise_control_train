from math import *
from cmath import *
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

def alpha(omega,Constantes,g):
    phi,sigma,alpha_h=Constantes
    #constantes caractéristiques de l'air
    ro_0=1.2 #kg.m^(-3)
    c_0=340 #m.s^(-1)
    gamma_p=1.4

    #dimension du domaine Omega
    L=10
    l=10
    delta_x=0.5

    eta_0= 1
    xi_0=1/c_0**2
    eta_1= phi/alpha_h
    xi_1=phi*gamma_p/(c_0**2)
    a= sigma*(phi**2)*gamma_p / ((c_0**2)*ro_0*alpha_h)

    #définition de g(omega,y)
    # d=2
    # def g(omega,y):
    #    return sin(omega*abs(y+d)/c_0)
    
    #décomposition de g(omega, y) en série de Fourrier et définition de g_k
    #on utilise la fonction quad du module scipy.integrate pour intégrer
    #cette fonction renvoie un tuple dont seule la première valeur nous intéresse
    def g_indice(k, omega):
        integrale_partie_reelle = integrate.quad(lambda y: (g(omega, y) * rect(1, -k*y)).real, -l, l)[0]
        integrale_partie_imag=integrate.quad(lambda y: (g(omega, y) * rect(1, -k*y)).imag, -l, l)[0]
        return (integrale_partie_reelle + integrale_partie_imag)/(2*l)
    
    #définition des fonctions intermédiaires
    def lambda_1(k, omega):
        terme_reel= (1/sqrt(2)) *sqrt(k**2 - (xi_1*omega**2)/eta_1 + sqrt((k**2 - (xi_1*omega**2)/eta_1)**2 + (a*omega/eta_1)**2))
        terme_imaginaire= - (1/sqrt(2))* sqrt(-k**2 + (xi_1*omega**2)/eta_1 + sqrt((k**2 - (xi_1*omega**2)/eta_1)**2 + (a*omega/eta_1)**2))
        return complex(terme_reel, terme_imaginaire)
    
    def Lambda_0(k, omega) :
        if k**2 >= xi_0 * omega**2 / eta_0 :
            return sqrt(k**2 + xi_0 * omega**2 / eta_0)
        else :
            return complex (0, sqrt(- k**2 + xi_0 * omega**2 / eta_0))

    def f(x, k, omega):
        lambda_0=Lambda_0(k, omega)
        return (lambda_0*eta_0 - x)*exp(-lambda_0*L) + (lambda_0*eta_0 + x)*exp(lambda_0*L)

    def Xi(k, alpha, omega):
        lambda_0=Lambda_0(k, omega)
        return g_indice(k, omega) * ( (lambda_0*eta_0 - lambda_1(k, omega)*eta_1) / f(lambda_1(k, omega)*eta_1, k, omega) - (lambda_0*eta_0 - alpha) / f(alpha, k, omega))

    def Gamma(k, alpha, omega):
        lambda_0=Lambda_0(k, omega)
        return g_indice(k, omega) * ( (lambda_0*eta_0 + lambda_1(k, omega)*eta_1) / f(lambda_1(k, omega)*eta_1, k, omega) - (lambda_0*eta_0 + alpha) / f(alpha, k, omega))
    
    #définition des e_k
    def e_k(k, alpha, omega):
        if k**2 >= xi_0 * omega**2 / eta_0 :
            lambda_0=Lambda_0(k, omega)
            xi=Xi(k,alpha, omega)
            gamma=Gamma(k,alpha, omega)
        
            terme_1 =(1+k**2)*((1/(2*lambda_0)) * ((abs(xi)**2)*(1-exp(-2*lambda_0*L)) + (abs(gamma)**2)*(exp(2*lambda_0*L)-1)) + 2*L*(xi*gamma.conjugate()).real)
            terme_2 = lambda_0/2 * ((abs(xi)**2)*(1-exp(-2*lambda_0*L)) + (abs(gamma)**2)*(exp(2*lambda_0*L)-1))
            terme_3 = -2 * (lambda_0**2) * L * (xi*gamma.conjugate()).real
            return terme_1+terme_2+terme_3
    
        else:
            lambda_0=Lambda_0(k, omega)
            xi=Xi(k,alpha, omega)
            gamma=Gamma(k,alpha, omega) 
            Re_terme_1= (1 + k**2)*(L*(abs(xi)**2 +abs(gamma)**2))
            Im_terme_1= (1/lambda_0 * (xi*gamma.conjugate()*(1- exp(-2*lambda_0*L))).imag)*(1 + k**2)
            terme_1=complex(Re_terme_1, Im_terme_1)
            terme_2= L*abs(lambda_0)**2 * (abs(xi)**2 +abs(gamma)**2)
            terme_3= complex(0, lambda_0* (xi*gamma.conjugate()*(1- exp(-2*lambda_0*L))).imag)
            return terme_1+terme_2+terme_3

    def e(alpha, omega, delta_x):
        n_max= int(L/delta_x)
        return sum(e_k(n*pi/l, alpha, omega) for n in range(-n_max, n_max))
    
    def fonction_a_minimiser(xy):
        x, y = xy
        alpha= x + y*1j
        return abs(e(alpha, omega, delta_x))

    res = opt.minimize(fonction_a_minimiser, [0,0])
    return res.x

## définition de g(omega,y)
d=2
def g(omega,y):
    return sin(omega*abs(y+d)/340)

def module(alpha):
    return(sqrt(alpha[0]**2+alpha[1]**2))

alphas = []
matériaux = []
prix = []

f=100
omega=2*pi*f
ISOREL = (0.70,142300,1.15)
matériaux.append('ISOREL')
alphas.append(module(alpha(omega,ISOREL,g)))
prix.append(4.50)
ITFH = (0.94,9067,1)
matériaux.append('ITFH')
alphas.append(module(alpha(omega,ITFH,g)))
prix.append(0)
B5 = (0.20,2124000,1.22)
matériaux.append('B5')
alphas.append(module(alpha(omega,B5,g)))
prix.append(0)
Chènevotte = (0.91,3500,2.5)
matériaux.append('Chènevotte')
alphas.append(module(alpha(omega,Chènevotte,g)))
prix.append(17.16*4/18)
matériaux.append('Laine biofib trio')
alphas.append(module(alpha(omega,(0.98,12800, 1.00),g)))
prix.append(29.51)
matériaux.append('Fibres de bois Protect L Dry')
alphas.append(module(alpha(omega,(0.8,135000, 1.00),g)))
prix.append(35.70)
matériaux.append('Fibres de bois Universal')
alphas.append(module(alpha(omega,(0.82,400000, 1.00),g)))
prix.append(11.35)
matériaux.append('Laines JetFibNature')
alphas.append(module(alpha(omega,(0.9,1000, 1.00),g)))
prix.append(28.80*5/8.5)
matériaux.append('Laines Cellaouate')
alphas.append(module(alpha(omega,(0.98,1000, 1.00),g)))
prix.append(23)
matériaux.append('Bétons Chanvribat')
alphas.append(module(alpha(omega,(0.7,2000, 1.15),g)))
prix.append(4)
matériaux.append('Bétons Isocanna')
alphas.append(module(alpha(omega,(0.5,15000, 4.4),g)))
prix.append(31.21*4/20)
plt.scatter(alphas,prix)
plt.title('Module de alpha pour f ='+ str(f) +'Hz')
plt.xlabel('Matériaux utilisés')
plt.ylabel('Module de alpha')
plt.xticks(rotation=45)
plt.show()

