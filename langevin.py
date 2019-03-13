import numpy as np
from numpy.linalg import norm
from numpy.random import normal, uniform
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.stats import entropy # KL div
from scipy.stats import wasserstein_distance
from scipy.stats import norm as norm_dist
from KDEpy import FFTKDE

import pickle
import itertools as it
from time import process_time
EPS = 1e-12

def sliced_wasserstein_distance(p, q, bin_coors, dim, iters=20):
    '''
        Utility function for the 1-Sliced Wasserstein distance.
        Assumes that p, q are represented as histograms.
    '''
    if dim == 1: # Use explicit formula
        return wasserstein_distance(bin_coors.flatten(), bin_coors.flatten(), p, q)

    dist = 0
    for _ in range(iters):
        # Sample (almost uniformly) at random from the (dim-1) sphere
        proj_vec = normal(size=dim)
        proj_vec = proj_vec / norm(proj_vec)

        bins, ps, qs = [], [], []
        # Compute projections
        for idx in it.product(*[range(d) for d in p.shape]):
            bins.append(np.dot( proj_vec, bin_coors[idx] ))
            ps.append( p[idx] )
            qs.append( q[idx] )

        dist += wasserstein_distance(bins, bins, ps, qs)
    return dist/iters


def sliced_wasserstein_no_histogram(p, q, iters=20):
    '''
        Utility function for the 1-Sliced Wasserstein distance.
        Tries to fight the course of dimensionality by only considering the
        "bins" around the sampled values.

        If the sampled values are entirely wrong, this method will not work >> use first moment test first.

        p = sampled values
        q = density function at the sampled values
    '''
    if any(np.isnan(x) for x in p.flatten()):
        return float('inf')
    if len(set(tuple(x) for x in p)) / len(p) <= 0.1:
        return float('inf')

    dim = p.shape[1]
    if dim == 1:
        return wasserstein_distance(p.flatten(), p.flatten(), np.ones(p.shape[0]) + EPS, q + EPS)

    dist = 0
    for _ in range(iters):
        proj_vec = normal(size=dim)
        proj_vec = proj_vec / norm(proj_vec) # sample randomly from dim-1 sphere
        bins = [np.dot( proj_vec, pt ) for pt in p]
        dist += wasserstein_distance(bins, bins, np.ones(p.shape[0]) + EPS, q + EPS)
    return dist/iters



class Potential:
    """ Represents a potential function. """
    def __init__(self, potential="gaussian", dimension=1, gaussian_sigma=None):
        ''' Parameters:
                potential:      Name of the potential                     (string)
                dimension:      Dimension of the potential                (int)
                gaussian_sigma: *Diagonal* covariance matrix for gaussian (1d array)
        '''
        self.name, self.dim = potential, dimension

        # When adding a new type of the potential, one needs to:
        #   - implement the potential function, gradient, possibly second gradient... in this class
        #                  (these are used by some of the algorithms)
        #   - link the implemented functions in the dictionary below
        self.function, self.gradient, self.gradient2, self.vector_lap_grad = {
            "gaussian":         (self.gaussian,         self.gaussian_grad,         self.gaussian_grad2,    self.gaussian_vector_lap_grad),
            "double_well":      (self.double_well,      self.double_well_grad,      self.double_well_grad2, self.double_well_vector_lap_grad),
            "Ginzburg_Landau":  (self.Ginzburg_Landau,  self.Ginzburg_Landau_grad,  None, None)
        }[potential]

        # Quantities to store in the Potential class, to avoid needless re-computation.
        if type(gaussian_sigma) != type(None): # Inverse covariance matrix for Gaussian
            self.inv_sigma = 1. / gaussian_sigma
        else:
            self.inv_sigma = 1. / np.arange(1, self.dim+1, dtype=float) # default

    def plot_density(self, rng=(-5, 5)):
        ''' Plots the density of the Potential.  '''
        if self.dim == 1:
            arr = np.array([np.exp(-self.function(i)) for i in np.arange(rng[0], rng[1], 0.01) ])
            plt.plot(np.arange(rng[0], rng[1], 0.01), arr / (np.sum(arr) * 0.01))
            plt.show()

    def get_histogram(self, edges):
        '''
            Helper function for construction of histograms.
            Edges: corresponding edges along each dimension, describing the bin.
        '''
        # calculate the centres of the bins (and thus edges)
        edges = np.array([np.array([(edge[i] + edge[i+1])/2 for i in range(len(edge)-1)]) for edge in edges])

        # q, the true distribution
        q = np.zeros(list(len(e) for e in edges))
        bin_coors = np.zeros(list(len(e) for e in edges) + [self.dim])

        # iterate over dim-dimensional indices
        for idx in it.product(*[range(len(e)) for e in edges]):
            # coordinate of the center
            coors = np.array([e[i] for i, e in zip(idx, edges)])
            bin_coors[idx] = coors
            q[idx] = np.exp(- self.function(coors))
        return q, bin_coors

    def get_density(self, pts):
        ''' Applies the density function on given array of points. '''
        return np.array([ np.exp(- self.function(pt)) for pt in pts])

    def gaussian(self, x):
        ''' Gaussian potential function. '''
        return 0.5 * np.dot(x, np.multiply(self.inv_sigma, x))

    def gaussian_grad(self, x):
        ''' Gradient of the Gaussian potential function. '''
        return np.multiply(self.inv_sigma, x)

    def gaussian_grad2(self, x):
        ''' Second gradient of the Gaussian potential function. '''
        return np.matrix(np.diag(self.inv_sigma))

    def gaussian_vector_lap_grad(self, x):
        return np.zeros(self.dim)

    def double_well(self, x):
        ''' Double Well potential function. '''
        normx = norm(x)
        return 0.25 * normx**4 - 0.5 * normx**2

    def double_well_grad(self, x):
        ''' Gradient - Double Well. '''
        return (norm(x)**2 - 1) * x

    def double_well_grad2(self, x):
        ''' Second gradient - Double Well. '''
        mx = np.matrix(x)
        return (norm(x)**2 - 1) * np.identity(self.dim) + 2 * np.transpose(mx) * mx

    def double_well_vector_lap_grad(self, x):
        ''' Laplacian of gradient -- Double Well. '''
        return 6*x

    def Ginzburg_Landau(self, x, tau=2.0, lamb=0.5, alpha=0.1):
        ''' Ginzburg-Landau potential function. '''
        d_ = round(self.dim ** (1./3))
        x = np.reshape(x, (d_,d_,d_))
        nabla_tilde = sum( norm(np.roll(x, -1, axis=a) - x)**2 for a in [0,1,2] )
        return 0.5 * (1. - tau) * norm(x)**2 + \
               0.5 * tau * alpha * nabla_tilde + \
               0.25 * tau * lamb * np.sum(np.power(x, 4))

    def Ginzburg_Landau_grad(self, x, tau=2.0, lamb=0.5, alpha=0.1):
        ''' Gradient - Ginzburg-Landau. '''
        d_ = round(self.dim ** (1./3))
        x = np.reshape(x, (d_,d_,d_))
        temp = sum( np.roll(x, sgn, axis=a) for sgn in [-1,1] for a in [0,1,2] )
        return ((1. - tau) * x + \
                tau * lamb * np.power(x, 3) + \
                tau * alpha * (6*x - temp)).flatten()





class Sampler:
    """ Samples a distribution defined by a given potential, using specific algorithms. """
    def __init__(self, potential="gaussian", dimension=1, x0=np.array([0.0]), step=0.01):
        ''' Parameters:
                potential:      Name of the potential                     (string)
                dimension:      Dimension of the potential                (int)
                x0:             Starting point                            (array of given dimension)
                step:           Step size                                 (float)
        '''
        self.dim, self.potential = dimension, Potential(potential, dimension)

        # When adding an algorithm, one needs to:
        #   - implement the algorithm in this class
        #   - link the implemented function in the dictionary below
        self.algorithms = {
            "ULA":     self.ULA,
            "tULA":    self.tULA,
            "tULAc":   self.tULAc,
            "MALA":    self.MALA,
            "RWM":     self.RWM,
            "tMALA":   self.tMALA,
            "tMALAc":  self.tMALAc,
            "tHOLA":   self.tHOLA,
            "LM":      self.LM,
            "tLM":     self.tLM,
            "tLMc":    self.tLMc
        }
        self.x0 = x0
        self.step = step

    def ULA(self):
        ''' Unadjusted Langevin Algoritm '''
        x = np.array(self.x0)
        sqrtstep = np.sqrt(2*self.step)
        while 1:
            yield x
            x = x - self.step * self.potential.gradient(x) + sqrtstep * normal(size=self.dim)

    def tULA(self, taming=(lambda g, step: g/(1. + step*norm(g)))):
        ''' Tamed Unadjusted Langevin Algorithm. '''
        x = np.array(self.x0)
        sqrtstep = np.sqrt(2*self.step)
        while 1:
            yield x
            x = x - self.step * taming(self.potential.gradient(x), self.step) + sqrtstep * normal(size=self.dim)

    def tULAc(self):
        ''' Coordinate-wise Tamed Unadjusted Langevin Algorithm. '''
        return self.tULA(lambda g, step: np.divide(g, 1. + step*np.absolute(g)))

    def MALA(self):
        ''' Metropolis Adjusted Langevin Algorithm. '''
        x = np.array(self.x0)
        while 1:
            yield x
            U_x, grad_U_x = self.potential.function(x), self.potential.gradient(x)
            y = x - self.step * grad_U_x + np.sqrt(2 * self.step) * normal(size=self.dim)
            U_y, grad_U_y = self.potential.function(y), self.potential.gradient(y)
            logratio = -U_y + U_x + 1./(4*self.step) * (norm(y - x + self.step*grad_U_x)**2 \
                       -norm(x - y + self.step*grad_U_y)**2)
            if np.log(uniform(size=1)) <= logratio:
                x = y

    def RWM(self):
        ''' Random Walk Metropolis algorithm. '''
        x = np.array(self.x0)
        while 1:
            yield x
            y = x + np.sqrt(2*self.step) * normal(size=self.dim)
            logratio = self.potential.function(x) - self.potential.function(y)
            if np.log(uniform(size = 1)) <= logratio:
                x = y

    def tMALA(self, taming=(lambda g, step: g/(1. + step*norm(g)))):
        ''' Tamed Metropolis Adjusted Langevin Algorithm. '''
        x = np.array(self.x0)
        while 1:
            yield x
            U_x, grad_U_x = self.potential.function(x), self.potential.gradient(x)
            tamed_gUx = taming(grad_U_x, self.step)
            y = x - self.step * tamed_gUx + np.sqrt(2*self.step) * normal(size=self.dim)
            U_y, grad_U_y = self.potential.function(y), self.potential.gradient(y)
            tamed_gUy = taming(grad_U_y, self.step)

            logratio = -U_y + U_x + 1./(4*self.step) * \
                        (norm(y - x + self.step*tamed_gUx)**2 - norm(x - y + self.step*tamed_gUy)**2)
            if np.log(uniform(size = 1)) <= logratio:
                x = y

    def tMALAc(self):
        ''' Coordinate-wise Tamed Metropolis Adjusted Langevin Algorithm. '''
        return self.tMALA(lambda g, step: np.divide(g, 1. + step * np.absolute(g)))

    def tHOLA(self):
        ''' Tamed Higher Order Langevin Algorithm. '''
        x = np.array(self.x0)
        while 1:
            yield x
            norm_x = norm(x)
            grad_U = self.potential.gradient(x)
            norm_grad_U = norm(grad_U)
            grad_U_gamma = grad_U / (1 + (self.step * norm_grad_U)**1.5)**(2./3)
            grad2_U = self.potential.gradient2(x)
            norm_grad2_U = norm(grad2_U)
            grad2_U_gamma = grad2_U / (1 + self.step * norm_grad2_U)

            laplacian_grad_U = self.potential.vector_lap_grad(x)
            laplacian_grad_U_gamma = laplacian_grad_U / (1 + self.step**0.5 * norm_x * norm(laplacian_grad_U))

            grad2_U_grad_U_gamma = np.matmul(grad2_U, grad_U).A1 / (1 + self.step * norm_x * norm_grad2_U * norm_grad_U)

            x = x - self.step * grad_U_gamma + 0.5 * self.step**2 * (grad2_U_grad_U_gamma - laplacian_grad_U_gamma) + \
                  np.sqrt(2*self.step) * normal(size=self.dim) - np.sqrt(2) * np.matmul(grad2_U_gamma, normal(size=self.dim)).A1 * np.sqrt(self.step**3/3)

    def LM(self):
        ''' Leimkuhler-Matthews algorithm. '''
        x = np.array(self.x0)
        sqrtstep = np.sqrt(0.5 * self.step)
        gaussian = normal(size=self.dim)
        while 1:
            yield x
            gaussian_plus1 = normal(size=self.dim)
            x = x - self.step * self.potential.gradient(x) + sqrtstep * (gaussian + gaussian_plus1)
            gaussian = gaussian_plus1

    def tLM(self, taming=(lambda g, step: g/(1. + step * norm(g)))):
        ''' Tamed Leimkuhler-Matthews algorithm. '''
        x = np.array(self.x0)
        sqrtstep = np.sqrt(0.5 * self.step)
        gaussian = normal(size=self.dim)
        while 1:
            yield x
            gaussian_plus1 = normal(size=self.dim)
            x = x - self.step * taming(self.potential.gradient(x), self.step) + sqrtstep * (gaussian + gaussian_plus1)
            gaussian = gaussian_plus1

    def tLMc(self):
        ''' Coordinate-wise Tamed Leimkuhler-Matthews algorithm. '''
        return self.tLM(lambda g, step: np.divide(g, 1. + step * np.absolute(g)))

    def get_samples(self, algorithm="ULA", burn_in=0, n_chains=1, n_samples=1e2, measuring_points=None, timer=None):
        ''' Returns n_samples from a given algorithm. '''

        all_samples = []
        for _ in range(n_chains):
            samples = []
            algo = self.algorithms[algorithm]()
            if timer:
                start_time = process_time()
                while 1:
                    if process_time() - start_time < timer:
                        samples.append(next(algo))
                    else: break
            else:
                samples.extend( [next(algo) for _ in range(n_samples)] )

            if type(measuring_points) == type(None):
                all_samples.extend(samples[burn_in:]) # return all samples
            else:
                for sample in np.array(samples[burn_in:])[measuring_points]:
                    all_samples.append(sample)

        return np.array(all_samples)






class Evaluator:
    ''' Analyses a set of sampling algoritms based on given parameters. '''
    def __init__(self, potential="gauFssian", dimension=1, x0=np.array([0.0]), step=0.01, N=10, burn_in=10**2, N_sim=3, N_chains=1, measuring_points=None, timer=None):
        self.potential = potential
        self.dim = dimension
        self.x0 = x0
        self.step = step
        self.N = N
        self.burn_in = burn_in
        self.N_sim = N_sim
        self.N_chains = N_chains
        self.timer = timer
        self.sampler = Sampler(potential=potential, dimension=dimension, x0=x0, step=step)
        self.measuring_points = measuring_points

    def analysis(self, algorithms=["tULA", "RWM"], measure="histogram", bins=10, repeat=1, experiment_mode=False):
        if not experiment_mode:
            # Print information about the analysis
            print('\n####### Initializing analysis #########\n' + '#'*39)
            print(' ALGORITHMS: {:s}'.format(str(algorithms)))
            print(' MEASURE: {:s}'.format(measure))
            print(' PARAMETERS:')
            for p in [('Potential', self.potential), ('Dimension', self.dim), ('x0', self.x0), ('Step', self.step), ('Number of iterations', self.N), \
                      ('Burn-in period', self.burn_in), ('Number of simulations', self.N_sim), ('Number of chains', self.N_chains), \
                      ('Measuring points', self.measuring_points), ('Time allocation', self.timer)]:
                print('  ' + '{:>22}:   {:s}'.format(*map(str,p)))
            print('#'*39 + '\n')

        # Collect the measurements.
        # For N_sim simulations, we store the measurement we are interested in (first moment, second moment, all samples...)
        measurements = {}
        for algo in algorithms:
            measurements[algo] = []

            for s in range(self.N_sim):
                samples = self.sampler.get_samples(algorithm=algo, burn_in=self.burn_in, n_chains=self.N_chains, n_samples=self.N, measuring_points=self.measuring_points, timer=self.timer)
                samples = np.unique(samples, axis=0)# remove duplicates (for rejection-based algorithms)

                if measure == "first_moment":
                    measurement = np.sum(samples, axis=0)/len(samples)

                elif measure == "second_moment":
                    measurement = np.sum(samples**2, axis=0)/len(samples)

                elif measure == "trace":
                    measurement = samples

                elif measure == "histogram":
                    measurement = np.histogram(samples, bins=bins, range=(-5, 5), density=True)

                elif measure in ["FFTKDE_KL", "FFTKDE_TV", "FFTKDE_SW"]:
                    measurement = samples

                elif measure in ["KL_divergence", "total_variation", "sliced_wasserstein"]:
                    try: # some algorithms blow up
                        measurement = np.histogramdd(samples, bins=bins)
                    except:
                        measurement = None, None

                elif measure == "sliced_wasserstein_no_histogram":
                    measurement = samples

                measurements[algo].append(measurement)
                print('   Algorithm: {:>5}, simulation {:d}, collected {:d} samples.'.format(algo, s, len(samples)))
        print()


        # Plot the results
        if measure in ["first_moment", "second_moment"]:
            data = [[m[0] for m in measurements[algo]] for algo in algorithms]
            # data = [[norm(m) for m in measurements[algo]] for algo in algorithms]

            if not experiment_mode:
                plt.boxplot(data, labels=algorithms)
            else:
                self.experiment_data["results"] = data

        elif measure == "trace":
            if not experiment_mode:
                for algo in algorithms:
                    plt.plot([p[0] for p in measurements[algo][0] if norm(p)<1e6], [p[1] for p in measurements[algo][0] if norm(p)<1e6], '-', linewidth=1, alpha=0.8)
                plt.legend(algorithms)

        elif measure == "histogram":
            if not experiment_mode:
                for algo in algorithms:
                    hist, bins = measurements[algo][0]
                    width = 0.85 * (bins[1] - bins[0])
                    center = (bins[:-1] + bins[1:])/2
                    plt.bar(center, hist, align='center', width=width, alpha=0.6)
                self.sampler.potential.plot_density()
                plt.legend(['true density'] + algorithms)

        elif measure in ["FFTKDE_KL", "FFTKDE_TV", "FFTKDE_SW"]:
            data = []
            for algo in algorithms:
                scores = []
                for s in range(self.N_sim):
                    weights = np.arange(len(measurements[algo][s])) + 1
                    # Don't know what this does ^
                    estimator = FFTKDE(kernel = 'gaussian')
                    x, ys = estimator.fit(measurements[algo][s], weights=weights).evaluate(30) # 30 is arbitrary
                    true_ys = self.sampler.potential.get_density(x)

                    if measure == "FFTKDE_KL":
                        scores.append( entropy(ys/np.sum(ys), true_ys/np.sum(true_ys) ))
                    if measure == "FFTKDE_TV":
                        scores.append( sum(abs( ys/np.sum(ys) - true_ys/np.sum(true_ys) ))/2 )
                    if measure == "FFTKDE_SW":
                        # print(ys, true_ys, x)
                        scores.append( sliced_wasserstein_distance( ys/np.sum(ys), true_ys/np.sum(true_ys), x, self.dim))
                data.append(scores)

            if not experiment_mode:
                plt.boxplot(data, labels=algorithms)
            else:
                self.experiment_data["results"] = data

        elif measure in ["KL_divergence", "total_variation", "sliced_wasserstein"]:
            data = []
            for algo in algorithms:
                scores = []
                for p, edges in measurements[algo]:
                    if type(p) == type(None):
                        continue
                    # true distribution histogram
                    q, bin_coors = self.sampler.potential.get_histogram(edges)

                    if measure == "KL_divergence":
                        ps, qs = p.flatten(), q.flatten()
                        scores.append( entropy(ps/sum(ps), qs/sum(qs) ))
                    elif measure == "total_variation":
                        ps, qs = p.flatten(), q.flatten()
                        scores.append( sum(abs( ps/sum(ps) - qs/sum(qs) ))/2 )
                    elif measure == "sliced_wasserstein":
                        scores.append( sliced_wasserstein_distance( p/np.sum(p), q/np.sum(q), bin_coors, self.dim ))
                data.append(scores)

            if not experiment_mode:
                plt.boxplot(data, labels=algorithms)
            else:
                self.experiment_data["results"] = data

        elif measure == "sliced_wasserstein_no_histogram":
            data = []
            for algo in algorithms:
                scores = []
                for p in measurements[algo]:
                    scores.append(sliced_wasserstein_no_histogram(p, self.sampler.potential.get_density(p) ))
                data.append(scores)
            if not experiment_mode:
                plt.boxplot(data, labels=algorithms)
            else:
                self.experiment_data["results"] = data

        if not experiment_mode:
            # Label and show
            plt.title('Measure: {:s}, '.format(measure) + '\nPotential: {:s}'.format(self.potential))
            plt.show()


    def run_experiment(self, file_path, algorithm, measure, bins=None):
        self.experiment_data = { "algorithm": algorithm,
                                  "measure": measure,
                                     "bins": bins,
                                "potential": self.potential,
                                "dimension": self.dim,
                                       "x0": self.x0,
                                     "step": self.step,
                                        "N": self.N,
                                  "burn_in": self.burn_in,
                                    "N_sim": self.N_sim,
                                    "timer": self.timer,
                                 "N_chains": self.N_chains }
        print('\n####### Running experiment #########\n' + '#'*39)
        print(' ALGORITHM: {:s}'.format(algorithm))
        print(' MEASURE: {:s}\n'.format(measure))

        self.analysis(algorithms=[algorithm], measure=measure, bins=bins, experiment_mode=True)
        pickle.dump( self.experiment_data, open( file_path, "wb" ) )
        self.experiment_data = {}

        print('\n####### Experiment finished #########\n' + '#'*39)
        print('Saved at: {:s}\n\n'.format(file_path))





####################################
# Using evaluator
####################################
d = 3 # dimension
e = Evaluator(potential="double_well", dimension=d, x0=np.array([0]+[0]*(d-1)), burn_in=2, N=1000, N_sim=2, step=0.05, N_chains=2, measuring_points=None, timer=None)

# Example of an analysis - produces a plot, doesn't store anything
e.analysis(algorithms=["ULA", "tULA", "RWM"], measure='total_variation', bins=40)

# Example of an experiment - only a single algorithm - does not produce a plot, stores the given path.
exp_name = 'Experiments/my_little_experiment'
e.run_experiment(file_path=exp_name, algorithm='ULA', measure='total_variation', bins=10)

# How to read an experiment in the future:
my_little_experiment = pickle.load(open( exp_name, 'rb' ))
for k, v in my_little_experiment.items():
   print(k, ':', v)
