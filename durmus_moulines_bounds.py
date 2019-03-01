import numpy as np
import matplotlib.pyplot as plt

# Increasing diagonal Gaussian specific



def EM_error(K, h, x):
    ''' Given number of steps K and step size h, produces upper bound on TV between Pi_h and distribution of nth iterate. '''
    global p, m, L, x_star
    kappa = 2 * m * L / (m + L)
    assert h < 2/(m+L)
    return (4 * np.pi * kappa * (1 -  (1 - kappa * h)**(K/2)))**-0.5 * (1 - kappa * h)**(K/2) * (np.linalg.norm(x - x_star) + (2 * p / kappa)**0.5)

def Pi_h_error(h):
    ''' Given number of steps K and step size h, produces upper bound on TV between Pi_h and true distribution. '''
    global p, m, L, L_twiddles, x_star
    kappa = 2 * m * L / (m + L)
    assert h < 1/(m+L)

    E_1 =  2 * p / kappa * (2 * L**2 + 4 / kappa * (p * L_twiddles**2 / 3 + h * L**4  / 4) + h**2 * L**4 / 6)
    E_2 = L**4 * (4 / (3 * kappa) + h)

    term_1 = (4 * np.pi)**-0.5 * ( h**2 * E_1 + (2 * p * h**2 * E_2)/(h*m))
    term_2 = (4 * np.pi)**-0.5 * np.ceil(np.log(1/h) / np.log(2)) * (h**2 * E_1 + h**2 * E_2 * (2/kappa * p + p/m))**0.5
    term_3 = 2**(-3/2) * L * ((2 * p * h**3 * L**2)/(3*kappa) + p*h**2)**0.5
    return term_1 + term_2 + term_3

L = 1
x_star = 0
L_twiddles = 0

p = 2
Ks = 10**4
x = 0   
m=1
hs = [0.01, 0.1, 0.4, 0.49]
Pi_h = np.array([Pi_h_error(h) for h in hs])
EM = np.array([EM_error(Ks, h, x) for h in hs])
print(EM + Pi_h)

plt.plot(hs, EM + Pi_h)
plt.show()
p = 1
#h = 0.01
#x = 0
m=1
#K = 1e4
#hs = [1e-2, 1e-1]
# Pi_h = [Pi_h_error(h) for h in hs]
# EM = [EM_error(K, h, x) for h in hs]
# print(EM)
