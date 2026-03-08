from emachine_v2 import aklt, pprint
import numpy as np
from matplotlib import pyplot as plt
from Centraliser import TensorProductGroup as TPG
from scipy.linalg import expm, null_space

mps = aklt()
bound = np.array([[1,0],[0,1]])
ph_2_3 = mps.parent_hamiltonian(2,3)
ph_2_2 = mps.parent_hamiltonian(2,2)
anlge = np.pi/3

S1 = np.array([[np.cos(anlge), -np.sin(anlge), 0], [np.sin(anlge), np.cos(anlge), 0], [0, 0, 1]])  # Z rotation
S2 = np.array([[np.cos(anlge), 0, np.sin(anlge)], [0, 1, 0], [-np.sin(anlge), 0, np.cos(anlge)]])  # Y rotation
S3 = np.array([[1, 0, 0], [0, np.cos(anlge), -np.sin(anlge)], [0, np.sin(anlge), np.cos(anlge)]])  # X rotation

Sx = (1/np.sqrt(2)) * np.array([
    [0,1,0],
    [1,0,1],
    [0,1,0]
], dtype=complex)

Sy = (1/np.sqrt(2)) * np.array([
    [0,-1j,0],
    [1j,0,-1j],
    [0,1j,0]
], dtype=complex)

Sz = np.array([
    [1,0,0],
    [0,0,0],
    [0,0,-1]
], dtype=complex)

Jx = np.kron(Sx, np.eye(3)) + np.kron(np.eye(3), Sx)
Jy = np.kron(Sy, np.eye(3)) + np.kron(np.eye(3), Sy)
Jz = np.kron(Sz, np.eye(3)) + np.kron(np.eye(3), Sz)

state = mps.to_ground_space(3, bound)

omps_1 = mps.apply_symmetry(S1)
omps_2 = mps.apply_symmetry(S2)
omps_3 = mps.apply_symmetry(S3)

pprint(Jx @ ph_2_2 - ph_2_2 @ Jx)
pprint(Jy @ ph_2_2 - ph_2_2 @ Jy)
pprint(Jz @ ph_2_2 - ph_2_2 @ Jz)