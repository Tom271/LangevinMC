import numpy as np

# Increasing diagonal Gaussian specific
p = 2 # dimension
m = 1/p
M = 1

def corrolary1(eps):
    ''' Given epsilon, returns number of steps K and step size h, to run ULA s.t. TV <= epsilon '''
    global p, m, M
    T = 0.5 * (4 * np.log(1/eps) + p * np.log(M / m))/m
    alpha = (1 + M * p * T * eps**(-2))/2
    h = eps**2 * (2*alpha - 1) / (M**2 * T * p * alpha)
    K = np.ceil(T/h)
    return K, h

def theorem2(K, h):
    ''' Given number of steps K and step size h, produces upper bound on TV. '''
    global p, m, M
    alpha = min(1/(h*M), K)
    assert alpha >= 1
    assert K >= alpha
    T = K*h
    return 0.5 * np.exp(0.25 * p * np.log(M/m) - 0.5 * T * m) + ( (p * M**2 * T * h * alpha) / (4 * (2*alpha - 1)) )**0.5

for eps in (0.02, 0.05, 0.1, 0.2, 0.5):
    print(corrolary1(eps))
