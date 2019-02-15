import numpy as np

# Increasing diagonal Gaussian specific
p = 10 # dimension
m = 1/p
L = 1
x_star = 0
L_twiddles = 0

def EM_error(K, h, x):
    ''' Given number of steps K and step size h, produces upper bound on TV between Pi_h and distribution of nth iterate. '''
    global p, m, L, x_star
    kappa = 2 * m * L / (m + L)
    assert h < 2/(m+L)
    return (4 * np.pi * kappa * (1 - kappa * h)**(K/2))**-0.5 * (1 - kappa * h)**(K/2) * (np.linalg.norm(x - x_star) + (2 * p / kappa)**0.5)

def Pi_gamma_error(h):
    global p, m, L, L_twiddles, x_star
    kappa = 2 * m * L / (m + L)
    assert h < 1/(m+L)

    E_1 =  2 * p / kappa * (2 * L**2 + 4 / kappa * (p * L_twiddles**2 / 3 + h * L**4  / 4) + h**2 * L**4 / 6)
    E_2 = L**4 * (4 / (3 * kappa) + h)

    return (4 * np.pi)**-0.5 * ( h**2 * E_1 + 2 * p * h**2 * E_2 * (2 * p / kappa + p / m))**0.5 + 2**(-1.5) * L * (2 * p * h**3 * L**2 / (3 * kappa) + p * h**2)**0.5


p = 10
h = 1e-5
K = 1e7
x = 0
print(EM_error(K, h, x) + Pi_gamma_error(h))
