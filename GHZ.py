from emachine_v2 import ghz, pprint
from emachine_v2 import MatrixProductState as MPS
import numpy as np
from matplotlib import pyplot as plt
mps = ghz()

ph = mps.parent_hamiltonian(3,3)

ph_art = np.eye(8)
ph_art[1:7, 1:7] = np.random.rand(6, 6)
ph_art[0,0] = 0
ph_art[7,7] = 0

U, = mps.sun_symmetry(ph, full_output=False, return_coeff=False)
umps = mps.apply_generator(U, theta=np.pi/3)

print(umps.to_ground_space(3, np.eye(2)))
