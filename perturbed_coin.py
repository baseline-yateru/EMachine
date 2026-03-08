import numpy as np
import emachine_v2 as EM
import matplotlib.pyplot as plt
import pandas as pd

pc = EM.perturbed_coin

P_vals = np.linspace(0.01, 0.99, 300)
Q_vals = np.linspace(0.01, 0.99, 300)

# Create meshgrid for 3D plot
P_grid, Q_grid = np.meshgrid(P_vals, Q_vals)

# Compute quantum statistical memory for each (P, Q) pair
qsm_grid = np.zeros_like(P_grid)
for i in range(len(P_vals)):
    for j in range(len(Q_vals)):
        qsm_grid[j, i] = pc(P_vals[i], Q_vals[j]).quantum_statistical_memory()

sm_grid = np.zeros_like(P_grid)
for i in range(len(P_vals)):
    for j in range(len(Q_vals)):
        sm_grid[j, i] = pc(P_vals[i], Q_vals[j]).statistical_memory()

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(P_grid, Q_grid, qsm_grid, cmap='viridis', alpha=0.5)

surf2 = ax.plot_surface(P_grid, Q_grid, sm_grid, cmap='plasma', alpha=0.5)

ax.set_xlabel('P')
ax.set_ylabel('Q')
ax.set_zlabel('Quantum Statistical Memory / Statistical Memory')
ax.set_title('Quantum Statistical Memory vs Statistical Memory of Perturbed Coin vs P and Q')

ax.legend(['Quantum Statistical Memory', 'Statistical Memory'])
# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=5)

plt.show()