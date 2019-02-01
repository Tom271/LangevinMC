import numpy as np
from numpy.linalg import norm
from numpy.random import normal, uniform
import matplotlib.pyplot as plt
from time import clock

class Potential:
    """ Represents a potential function. """
    def __init__(self, potential="gaussian", dimension=1):
        self.dim = dimension
        self.name = potential
        # To add a potential, add it to the dictionary below and implement it/its gradient.
        self.function, self.gradient, self.gradient2, self.vector_lap_grad = {
            "gaussian":         (self.gaussian,         self.gaussian_grad,         None, None),
            "double_well":      (self.double_well,      self.double_well_grad,      self.double_well_grad2, self.double_well_vector_lap_grad),
            "Ginzburg_Landau":  (self.Ginzburg_Landau,  self.Ginzburg_Landau_grad,  None, None)
        }[potential]

        # Quantities to store in the Potential class, to avoid needless re-computation.
        self.inv_sigma = 1. / np.arange(1, dimension+1, dtype=float) # for Gaussian

    def gaussian(self, x):
        return 0.5 * np.dot(x, np.multiply(self.inv_sigma, x))

    def gaussian_grad(self, x):
        return np.multiply(self.inv_sigma, x)

    def double_well(self, x):
        normx = norm(x)
        return 0.25 * normx**4 - 0.5 * normx**2

    def double_well_grad(self, x):
        return (norm(x)**2 - 1) * x

    def double_well_grad2(self, x):
        mx = np.matrix(x)
        return (norm(x)**2 - 1) * np.identity(self.dim) + 2 * np.transpose(mx) * mx

    def double_well_vector_lap_grad(self, x):
        return 6*x

    def Ginzburg_Landau(self, x, tau=2.0, lamb=0.5, alpha=0.1):
        d_ = round(self.dim ** (1./3))
        x = np.reshape(x, (d_,d_,d_))
        nabla_tilde = sum( norm(np.roll(x, -1, axis=a) - x)**2 for a in [0,1,2] )
        return 0.5 * (1. - tau) * norm(x)**2 + \
               0.5 * tau * alpha * nabla_tilde + \
               0.25 * tau * lamb * np.sum(np.power(x, 4))

    def Ginzburg_Landau_grad(self, x, tau=2.0, lamb=0.5, alpha=0.1):
        d_ = round(self.dim ** (1./3))
        x = np.reshape(x, (d_,d_,d_))
        temp = sum( np.roll(x, sgn, axis=a) for sgn in [-1,1] for a in [0,1,2] )
        return ((1. - tau) * x + \
                tau * lamb * np.power(x, 3) + \
                tau * alpha * (6*x - temp)).flatten()

class Evaluator:
    """ Evaluates a set of Langevin algorithms on given potentials. """
    def __init__(self, potential="gaussian", dimension=1, N=10**2, burn_in=10**2, N_sim=5, x0=[0], step=0.01, timer=None):
        self.dim = dimension
        # To add an algorithm, add it to the dictionary below and implement it as a class method.
        self.algorithms = {
            "ULA":     self.ULA,
            # "tULA":    self.tULA,
            # "tULAc":   self.tULAc,
            # "MALA":    self.MALA,
            # "RWM":     self.RWM,
            # "tMALA":   self.tMALA,
            # "tMALAc":  self.tMALAc,
            # "tHOLA":   self.tHOLA,
            # "LM":      self.LM,
            # "tLM":     self.tLM,
            # "tLMc":    self.tLMc
        }
        self.N = N
        self.burn_in = burn_in
        self.N_sim = N_sim
        self.x0 = x0
        self.step = step
        self.timer = timer
        if timer:
            self.N = 10**10 # ~~practically infinity
            self.start_time = clock()
        self.potential = Potential(potential, dimension)
        # invoked by self.potential.funtion(parameters), self.potential.gradient(parameters)
    def ULA(self):
        x = np.array(self.x0)
        sample = np.zeros((self.dim, self.N + self.burn_in), dtype=float)
        sqrtstep = np.sqrt(2*self.step)

        for i in range(self.burn_in + self.N):
            if self.timer and clock() - self.start_time > self.timer:
                break
            sample[:,i]=x
            x += -self.step * self.potential.gradient(x) + sqrtstep * normal(size=self.dim)

        return sample #i+1 = no. of iterations

    def sampler(self, algorithm="ULA"):
        if self.timer:
            self.start_time = clock()
        return self.algorithms[algorithm]()

potential = 'gaussian'
d = 2
N = 10**3
burn_in = 0
N_sim = 1
x0 = np.array([10] + [10]*(d-1), dtype=float)
step = 0.1

# TIMER MODE: number of seconds which we allow the algorithms to run
# To run normally without a timer, omit the last parameter
timer = 2.5

e = Evaluator(potential, dimension=d, N=N, burn_in=burn_in, N_sim=N_sim, x0=x0, step=step)

data = e.sampler("ULA")

fig = plt.figure()
plt.title("Trace" )
plt.plot(data[0],data[1])

from matplotlib import cm

from scipy.stats import multivariate_normal as MVN

# Our 2-dimensional distribution will be over variables X and Y
N = 600
X = np.linspace(min(data[0]), max(data[0]), N)
Y = np.linspace(min(data[1]), max(data[1]) N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0])
Sigma = np.diag(np.arange(1, 3, dtype=float))

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

F = MVN(mu, Sigma)
Z = F.pdf(pos)

ax = fig.gca()
ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.hot, alpha=0.5)
cset = plt.contour(X, Y, Z, cmap=cm.hot, alpha=0.5)

plt.show()
