# ### TULA Code from https://github.com/nbrosse/TULA/blob/master/code_TULA.py
# ##### Tamed Unadjusted Langevin Algorithm from Brosse et al. 2018

import numpy as np


class potential:
    """ Implements the potentials U and the algorithms described in the article.
    """

    # Parameters for Ginzburg-Landau model
    tau=2.0
    lamb=0.5
    alpha=0.1
    # threshold for stopping the trajectory of ULA
    threshold=10**5
    def __init__(self, typ, d):
        """ Initialize the object.
        :param typ: potential U
        :param d: dimension
        """

        self.type = typ
        self.d = d
        self.acc_mala = 0 # acceptance probability for MALA
        self.acc_rwm = 0 # acceptance probability for RWM
        self.acc_tmala = 0 # acceptance probability for TMALA
        self.acc_tmalac = 0 # acceptance probability for TMALAc

    def potential(self, X):
        """ Definition of the potential
        :param X: point X of Rd where the potential is evaluated
        :return: U(X)
        """
        if self.type == "ill-cond-gaussian":
            d = self.d
            v = np.array(np.arange(1,d+1), dtype = float)
            InvSigma = 1. / v
            return 0.5*np.dot(X, np.multiply(InvSigma, X))

        elif self.type == "double-well":
            return (1./4)*np.linalg.norm(X)**4 - (1./2)*np.linalg.norm(X)**2

        elif self.type == "Ginzburg-Landau":
            tau = self.tau
            alpha = self.alpha
            lamb = self.lamb
            dim = int(np.rint((self.d)**(1./3))) #rint -> round to nearest integer
            X = np.reshape(X, (dim,dim,dim)) #Make X into 3d array R^(d x d x d)
            temp = np.linalg.norm(np.roll(X, -1, axis=0) - X)**2                    + np.linalg.norm(np.roll(X, -1, axis=1) - X)**2                    + np.linalg.norm(np.roll(X, -1, axis=2) - X)**2
            return 0.5*(1-tau)*np.linalg.norm(X)**2 + 0.5*tau*alpha*temp                     + (1./4)*tau*lamb*np.sum(np.power(X,4))
        else:
            print("Error potential not defined")

    def gradpotential(self, X):
        """ Definition of the gradient of the potential
        :param X: point X of Rd where the gradient is evaluated
        :return: gradU(X)
        """

        if self.type == "ill-cond-gaussian":
            d = self.d
            v = np.array(np.arange(1,d+1), dtype = float)
            InvSigma = 1. / v
            return np.multiply(InvSigma, X)
        elif self.type == "double-well":
            return (np.linalg.norm(X)**2 - 1)*X
        elif self.type == "Ginzburg-Landau":
            tau = self.tau
            alpha = self.alpha
            lamb = self.lamb
            dim = int(np.rint((self.d)**(1./3)))
            X = np.reshape(X, (dim,dim,dim))
            temp = np.roll(X, 1, axis=0)+np.roll(X, -1, axis=0)                     +np.roll(X, 1, axis=1)+np.roll(X, -1, axis=1)                     +np.roll(X, 1, axis=2)+np.roll(X, -1, axis=2)
            gradU = (1.0-tau)*X + tau*lamb*np.power(X,3) + tau*alpha*(6*X - temp)
            return gradU.flatten()
        else:
            print("Error gradpotential not defined")
        def grad2potential(self,X):
            """ Definition of the gradient of the potential
            :param X: point X of Rd where the gradient is evaluated
            :return: gradU(X)
            """
        if self.type == "double-well":
            return ((np.linalg.norm(X)**2 - 1) * np.identity(self.d) + 2 * np.transpose(np.matrix(X)) * np.matrix(X))

    def vectorLaplaciangradpotential(self,X):
        if self.type == "double-well":
            return(6*X)


    def tHOLA(self, x0, step, N=10**6):
        """Tamed Higher order Langevin Algorithm (Sabanis & Zhang)
        """
        d = self.d
        X = x0
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        burnin = 10**4 # burn in period
        for k in np.arange(burnin):
            if np.linalg.norm(X, np.inf)>self.threshold:
                m1[:] = np.nan
                m2[:] = np.nan
                return (m1,m2)
            gradU = self.gradpotential(X)
            gradUgamma = gradU / ((1 + step ** (3 / 2) * np.linalg.norm(gradU) ** (3/2)) ** (2/3))
            grad2U = self.grad2potential(X)
            grad2Ugamma = grad2U/(1 + step * np.linalg.norm(grad2U))
            laplacianGradU = self.vectorLaplaciangradpotential(X)
            laplacianGradUgamma = laplacianGradU/(1 + step ** (1/2) * np.linalg.norm(X) * np.linalg.norm(laplacianGradU))
            gradUgrad2Ugamma = (gradU * grad2U) / (1 + step * np.linalg.norm(X) * np.linalg.norm(grad2U) * np.linalg.norm(gradU))
            X += - step*gradUgamma + (step**2)/2 * (gradUgrad2Ugamma - laplacianGradUgamma) + np.sqrt(2*step)*np.random.normal(size=d) - np.sqrt(2) * grad2Ugamma * np.sqrt(step ** 3 / 3) * np.random.normal(size=d)
        for k in np.arange(N):
            if np.linalg.norm(X, np.inf)>self.threshold:
                m1[:] = np.nan
                m2[:] = np.nan
                return (m1,m2)
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            gradU = self.gradpotential(X)
            gradUgamma = gradU / ((1 + step ** (3 / 2) * np.linalg.norm(gradU) ** (3/2)) ** (2/3))
            grad2U = self.grad2potential(X)
            grad2Ugamma = grad2U/(1 + step * np.linalg.norm(grad2U))
            laplacianGradU = self.vectorLaplaciangradpotential(X)
            laplacianGradUgamma = laplacianGradU/(1 + step ** (1/2) * np.linalg.norm(X) * np.linalg.norm(laplacianGradU))
            gradUgrad2Ugamma = (gradU * grad2U) / (1 + step * np.linalg.norm(X) * np.linalg.norm(grad2U) * np.linalg.norm(gradU))
            X += - step*gradUgamma + (step**2)/2 * (gradUgrad2Ugamma - laplacianGradUgamma) + np.sqrt(2*step)*np.random.normal(size=d) - np.sqrt(2) * grad2Ugamma * np.sqrt(step ** 3 / 3) * np.random.normal(size=d)
        return (m1,m2)

    def ULA(self, x0, step, N=10**6):
        """ Algorithm ULA
        :param x0: starting point
        :param step: step size of the algorithm ?gamma in paper?
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments m1,m2
        """
        d = self.d
        X = x0
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        burnin = 10**4 # burn in period
        for k in np.arange(burnin):
            #Check divergence? Max row sum > threshold
            if np.linalg.norm(X, np.inf)>self.threshold:
                m1[:] = np.nan
                m2[:] = np.nan
                return (m1,m2)

            gradU = self.gradpotential(X)
            #E-M discretization of Langevin SDE, take a step to X_k+1
            X = X - step*gradU + np.sqrt(2*step)*np.random.normal(size=d)
        #Same again but after burn-in
        for k in np.arange(N):
            if np.linalg.norm(X, np.inf)>self.threshold:
                m1[:] = np.nan
                m2[:] = np.nan
                return (m1,m2)
            #Calculate mean
            m1 += X / float(N)
            #Calculate E[X^2]
            m2 += np.power(X, 2) / float(N)
            gradU = self.gradpotential(X)
            X = X - step*gradU + np.sqrt(2*step)*np.random.normal(size=d)
        return (m1,m2)

    def TULA(self, x0, step, N=10**6):
        """ Algorithm TULA
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments
        """
        #TULA same as ULA but use tamed grad instead of grad

        d = self.d
        X = x0
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        burnin = 10**4
        for k in np.arange(burnin):
            gradU = self.gradpotential(X)
            gradUTamed = gradU / (1.0 + step*np.linalg.norm(gradU))
            X = X - step*gradUTamed + np.sqrt(2*step)*np.random.normal(size=d)
        for k in np.arange(N):
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            gradU = self.gradpotential(X)
            gradUTamed = gradU / (1.0 + step*np.linalg.norm(gradU))
            X = X - step*gradUTamed + np.sqrt(2*step)*np.random.normal(size=d)
        return (m1,m2)

    def TULAc(self, x0, step, N=10**6):
        """ Algorithm TULAc
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments
        """

        d = self.d
        X = x0
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        burnin = 10**4
        for k in np.arange(burnin):
            gradU = self.gradpotential(X)
            #Here is the difference:
            gradUTamed = np.divide(gradU, 1.0 + step*np.absolute(gradU))
            #Does not require norm calculation, only abs value. np.divide is element div
            X = X - step*gradUTamed + np.sqrt(2*step)*np.random.normal(size=d)
        for k in np.arange(N):
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            gradU = self.gradpotential(X)
            gradUTamed = np.divide(gradU, 1.0 + step*np.absolute(gradU))
            X = X - step*gradUTamed + np.sqrt(2*step)*np.random.normal(size=d)
        return (m1,m2)


    def TMALA(self, x0, step, N=10**6):
        """ Algorithm TMALA
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments
        """

        acc=0 # to store the empirical acceptance probability
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        X = x0
        burnin = 10**4
        for k in np.arange(burnin):
            U_X = self.potential(X)
            #MALA but tame gradU as in TULA
            grad_U_X = self.gradpotential(X)
            Tgrad_U_X = grad_U_X / (1. + step*np.linalg.norm(grad_U_X))
            #Y is proposal
            Y = X - step * Tgrad_U_X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            grad_U_Y = self.gradpotential(Y)
            Tgrad_U_Y = grad_U_Y / (1. + step*np.linalg.norm(grad_U_Y))
            logratio = - U_Y + U_X + (1./(4*step))*(np.linalg.norm(Y-X+step*Tgrad_U_X)**2                            - np.linalg.norm(X-Y+step*Tgrad_U_Y)**2)
            #Metropolis rejection
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y

        #Same again after burn-in
        for k in np.arange(N):
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            U_X = self.potential(X)
            grad_U_X = self.gradpotential(X)
            Tgrad_U_X = grad_U_X / (1. + step*np.linalg.norm(grad_U_X))
            Y = X - step * Tgrad_U_X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            grad_U_Y = self.gradpotential(Y)
            Tgrad_U_Y = grad_U_Y / (1. + step*np.linalg.norm(grad_U_Y))
            logratio = - U_Y + U_X + (1./(4*step))*(np.linalg.norm(Y-X+step*Tgrad_U_X)**2                            - np.linalg.norm(X-Y+step*Tgrad_U_Y)**2)
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
                acc+=1
        self.acc_tmala = float(acc)/N # empirical acceptance probability
        return (m1,m2)


    def TMALAc(self, x0, step, N=10**6):
        """ Algorithm TMALAc
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments
        """

        acc=0
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        X = x0
        burnin=10**4
        for k in np.arange(burnin):
            U_X = self.potential(X)
            grad_U_X = self.gradpotential(X)
            Tgrad_U_X = np.divide(grad_U_X, 1. + step*np.absolute(grad_U_X))
            Y = X - step * Tgrad_U_X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            grad_U_Y = self.gradpotential(Y)
            Tgrad_U_Y = np.divide(grad_U_Y, 1. + step*np.absolute(grad_U_Y))
            logratio = - U_Y + U_X + (1./(4*step))*(np.linalg.norm(Y-X+step*Tgrad_U_X)**2                            - np.linalg.norm(X-Y+step*Tgrad_U_Y)**2)
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
        for k in np.arange(N):
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            U_X = self.potential(X)
            grad_U_X = self.gradpotential(X)
            Tgrad_U_X = np.divide(grad_U_X, 1. + step*np.absolute(grad_U_X))
            Y = X - step * Tgrad_U_X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            grad_U_Y = self.gradpotential(Y)
            Tgrad_U_Y = np.divide(grad_U_Y, 1. + step*np.absolute(grad_U_Y))
            logratio = - U_Y + U_X + (1./(4*step))*(np.linalg.norm(Y-X+step*Tgrad_U_X)**2                            - np.linalg.norm(X-Y+step*Tgrad_U_Y)**2)
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
                acc+=1
        self.acc_tmalac = float(acc)/N
        return (m1,m2)


    def MALA(self, x0, step, N=10**6):
        """ Algorithm TMALA
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments
        """

        acc=0
        d = self.d
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        X = x0
        burnin = 10**4
        for k in np.arange(burnin):
            U_X = self.potential(X)
            grad_U_X = self.gradpotential(X)
            Y = X - step * grad_U_X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            grad_U_Y = self.gradpotential(Y)
            logratio = - U_Y + U_X + (1./(4*step))*(np.linalg.norm(Y-X+step*grad_U_X)**2                            - np.linalg.norm(X-Y+step*grad_U_Y)**2)
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
        for k in np.arange(N):
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            U_X = self.potential(X)
            grad_U_X = self.gradpotential(X)
            Y = X - step * grad_U_X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            grad_U_Y = self.gradpotential(Y)
            logratio = - U_Y + U_X + (1./(4*step))*(np.linalg.norm(Y-X+step*grad_U_X)**2                            - np.linalg.norm(X-Y+step*grad_U_Y)**2)
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
                acc+=1
        self.acc_mala = float(acc)/N
        return (m1,m2)

    def RWM(self, x0, step, N=10**6):
        """ Algorithm RWM
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :return: empirical averages of 1st and 2nd moments
        """

        acc=0
        d = self.d
        m1 = np.zeros(d)
        m2 = np.zeros(d)
        X = x0
        burnin = 10**4
        for k in np.arange(burnin):
            U_X = self.potential(X)
            Y = X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            logratio = - U_Y + U_X
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
        for k in np.arange(N):
            m1 += X / float(N)
            m2 += np.power(X, 2) / float(N)
            U_X = self.potential(X)
            Y = X + np.sqrt(2*step)*np.random.normal(size=self.d)
            U_Y = self.potential(Y)
            logratio = - U_Y + U_X
            if np.log(np.random.uniform(size=1))<=logratio:
                X = Y
                acc+=1
        self.acc_rwm = float(acc)/N
        return (m1,m2)

    #Actually running the method
    def sampler(self, x0, step, N=10**6, choice="ULA"):
        """ General call to a sampler ULA, TULA, TULAc, MALA, RWM, TMALA or TMALAc
        :param x0: starting point
        :param step: step size of the algorithm
        :param N: number of iterations (after burn in period)
        :param choice: choice of the algorithm
        :return: empirical averages of 1st and 2nd moments
        """

        if choice=="ULA":
            return self.ULA(x0, step, N=N)
        elif choice=="TULA":
            return self.TULA(x0, step, N=N)
        elif choice=="TULAc":
            return self.TULAc(x0, step, N=N)
        elif choice=="MALA":
            return self.MALA(x0, step, N=N)
        elif choice=="RWM":
            return self.RWM(x0, step, N=N)
        elif choice=="TMALA":
            return self.TMALA(x0, step, N=N)
        elif choice=="TMALAc":
            return self.TMALAc(x0, step, N=N)
        else:
            print("error sampler not defined")

    def analysis(self, x0, NSimu=100, N=10**6,                      stepTab = np.array([10**(-3), 10**(-2), 10**(-1), 1.0]),                      choice="ULA"):
        """ Analysis for one given algorithm, different step sizes, and NSimu
        independent simulations

        Warning: this function may need high computational power

        :param x0: starting point
        :param NSimu: number of independent simulations
        :param N: number of iterations (after burn in period)
        :param stepTab: table of different step sizes for the algorithm
        :param choice: choice of the algorithm

        :return: algorithm, stepTab, initial condition,
        average acceptance probability for MALA, RWM, TMALA and TMALAc,
        empirical averages of 1st and 2nd moments for the first and  last coordinate,
        the different step sizes and the different independent simulations (NSimu)
        """

        d = self.d
        nStep = stepTab.size

        # Acceptance probabilities
        accMala = np.zeros(nStep)
        accRwm = np.zeros(nStep)
        accTmala = np.zeros(nStep)
        accTmalac = np.zeros(nStep)

        # To store the empirical evrages of 1st and 2nd moments for the
        # first and last coordinate
        moment_1 = np.zeros((nStep, NSimu, 2))
        moment_2 = np.zeros((nStep, NSimu, 2))

        for i in np.arange(nStep):
            m1tab = np.zeros((NSimu, d))
            m2tab = np.zeros((NSimu, d))
            for k in np.arange(NSimu):
                m1,m2 = self.sampler(x0, step=stepTab[i], N=N, choice=choice)
                m1tab[k,:] = m1
                m2tab[k,:] = m2
                accMala[i] += float(self.acc_mala)/NSimu
                accRwm[i] += float(self.acc_rwm)/NSimu
                accTmala[i] += float(self.acc_tmala)/NSimu
                accTmalac[i] += float(self.acc_tmalac)/NSimu
            moment_1[i, :] = m1tab[:,[0,-1]]
            moment_2[i, :] = m2tab[:,[0,-1]]

        if choice=="MALA":
            return (choice, stepTab, x0, accMala, moment_1, moment_2)
        elif choice=="RWM":
            return (choice, stepTab, x0, accRwm, moment_1, moment_2)
        elif choice=="TMALA":
            return (choice, stepTab, x0, accTmala, moment_1, moment_2)
        elif choice=="TMALAc":
            return (choice, stepTab, x0, accTmalac, moment_1, moment_2)
        else:
            return (choice, stepTab, x0, moment_1, moment_2)


pot=potential("double-well",2)

pot.analysis(np.array([0,0]),NSimu=2,N=10**4,choice="TULA")
