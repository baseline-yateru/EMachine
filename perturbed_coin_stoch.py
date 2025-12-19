import numpy as np
from emachine import EMachine as EM
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import fmin


em =lambda p: EM(np.array([[[1-p, 0],
               [0.3,0]],
               [[0, p],
                [0, 0.7]]])**(1/2))
ITER = 1000
ENS = 50
n = np.arange(1,ITER+1)

#print(np.sum(em.emission_distribution(0).T, axis = 2))
def fit(p, x):
    return np.sum((x + np.log10(n[int(ITER/2)]) - p[0])**2)

colors = plt.cm.viridis(np.linspace(0, 1, 20))
plt.gca().set_prop_cycle(color=colors)
plt.style.use('seaborn-v0_8')
intc_ls = []
intc_err_ls = []
for k, p in enumerate(np.linspace(0.01,0.99,20)):
    print(f"Calculating P = {p: .2f}, time:{dt.datetime.now()}")
    e = em(p)
    f = np.array([1,0])
    e_dist = np.array([np.dot(f @ e.emission_distribution(i), np.array([0, 1])) for i in range(ITER)])
    cum_avg = np.array([(np.cumsum([e.propagator(ITER, f)])/n - e_dist) for _ in range(ENS)])
    var_avg = np.array([np.mean(cum_avg[:,i]**2) for i in range(cum_avg.shape[1])])
    intc = 10**np.mean(np.log10(var_avg[int(ITER/2):]*n[int(ITER/2):]))
    intc_err = np.log(10) * intc * np.std(np.log10(var_avg[int(ITER/2):]*n[int(ITER/2):]))
    intc_ls.append(intc)
    intc_err_ls.append(intc_err)
    #plt.loglog(n[int(ITER/2):], intc/n[int(ITER/2):], "--", color=colors[k], label=f'Fit P={p: .2f}')
    #plt.loglog(n[int(ITER/2):], var_avg[int(ITER/2):], color=colors[k], label=f'P={p: .2f}')
np.savetxt('intc_pc_ls.csv', [intc_ls, intc_err_ls], delimiter=',')

plt.legend()
plt.xlabel('Iterations')
plt.ylabel(r'$\sigma^2/n$')
plt.show()
