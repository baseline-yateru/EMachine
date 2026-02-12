import numpy as np
import matplotlib.pyplot as plt
from emachine import EMachine as em
from emachine import pprint

def A_ge(p,e):
    T_e = np.array([[[p-e, 0],
                    [0, 0]],
                    [[0, 1-p],
                    [1, 0]]])
    T_g = np.array([[[0, p],
                    [0, 0]],
                    [[1-p-e, 0],
                    [1, 0]]])

    T_eg = np.zeros((2,4,4))
    T_eg[:,0:2,0:2] = T_e
    T_eg[:,2:4,2:4] = T_g
    T_eg[0,0,2] += e
    T_eg[1,2,0] += e
    return T_eg**(1/2)

A_even = np.array([[[0.7,0],
                   [0,0]],
                   [[0,0.3],
                    [1,0]]])**(1/2)

A_golden = np.array([[[0, 0.7],
                    [0, 0]],
                    [[0.3, 0],
                    [1, 0]]])**(1/2)

# Fixed first argument (p), vary second argument (e)
P_fixed = 0.7
E = np.linspace(0.00001, 0.1, 1000)

em_goldn = em(A_golden)
em_even = em(A_even)
ham_goldn = em_goldn.parent_hamiltonian(5, 6)
ham_even = em_even.parent_hamiltonian(5, 6)

eig_goldn = np.real(np.linalg.eigvals(ham_goldn))
eig_even = np.real(np.linalg.eigvals(ham_even))

# Store eigenvalues for each e
eigenvalues_dict = {e_val: [] for e_val in E}

for e in E:
    em_ge = em(A_ge(P_fixed, e))
    ph = em_ge.parent_hamiltonian(5, 6)
    eigs = np.real(np.linalg.eigvals(ph))
    eigenvalues_dict[e] = eigs

# Plot eigenvalues vs e
plt.figure(figsize=(9, 6))
for i in range(len(eigenvalues_dict[E[0]])):
    eig_values = [eigenvalues_dict[e][i] for e in E]
    plt.scatter(E, eig_values, alpha=0.7, s=1, color="#1f77b4")
for i, eg in enumerate(eig_goldn):
    if i == 0:
        plt.axhline(y=eg, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Golden Eigenvalues')
    else:
        plt.axhline(y=eg, color='red', linestyle='--', alpha=0.5, linewidth=1)
for i, ee in enumerate(eig_even):
    if i == 0:
        plt.axhline(y=ee, color='green', linestyle='-.', alpha=0.5, linewidth=1, label='Even Eigenvalues')
    else:
        plt.axhline(y=ee, color='green', linestyle='-.', alpha=0.5, linewidth=1)
plt.legend(fontsize=17)

plt.xlabel('Parameter e', fontsize=17)
plt.ylabel('Eigenvalues', fontsize=17)
plt.title(f'Parent Hamiltonian Eigenvalues vs e (p={P_fixed})', fontsize=17)
plt.tick_params(labelsize=17)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('golden_even_ham_5_6.png', dpi=300, bbox_inches='tight')
plt.show()
