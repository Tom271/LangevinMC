import pickle
import numpy as np
import matplotlib.pyplot as plt

Ns = list(map(int, np.logspace(0.33, 6, 15)))
dalalyan = [0.71174503721166149, 0.71010807269829468, 0.70288001492738117, 0.67765855577197875, 0.61335255602197492, 0.47724641965989473, 0.28814923813543181, 0.20994713436079471, 0.3055842130261629, 0.4870173812986866, 0.77634629376277431, 1.2375307721670203, 1.9726840629103033, 3.1445494040917645, 5.0125470711708555]
step = 0.01

results = []
for N in Ns:
    e = pickle.load(open( 'N_' + str(N) + '_step_' + str(step), 'rb' ))
    results.append(e["results"][0])

plt.plot(range(1, len(Ns)+1), dalalyan)
plt.legend(['Dalalyan, Theorem 2'])
plt.boxplot(results, labels=Ns)
plt.xlabel('Number of iterations')
plt.ylabel('Total variation')
plt.title('Gaussian, dim=2, sigma=diag(1,2), step=0.01   [ULA]')
plt.show()
