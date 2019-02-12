import pickle
import numpy as np
import matplotlib.pyplot as plt

Ns = list(map(int, np.logspace(0.33, 6, 15)))
dalalyan = [1.1459606749835214, 1.1447288042104871, 1.1378411329651315, 1.1116599299911056, 1.0405653733443379, 0.87432667993205815, 0.58434170180566702, 0.33440854241007578, 0.3765149327563308, 0.59647220743535923, 0.95082614170982593, 1.5156594664008312, 2.416034688925317, 3.8512707554988657, 6.139091318025435]
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
plt.title('Gaussian, dim=3, sigma=diag(1,2,3), step=0.01   [ULA]')
plt.show()
