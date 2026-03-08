import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
import pandas as pd
from emachine_v2 import EMachine as EM, pprint
from emachine_v2 import MatrixProductState as MPS
from emachine_v2 import even_process as ep
from einops import rearrange
from scipy.linalg import expm

em = ep(0.7)
pep = ep(0.7).A.astype(np.complex128)
carg = (1+1j)/np.sqrt(2)
pep[0] *= carg
pep[1] *= np.conj(carg)
mps = MPS(pep)

em_ph = em.parent_hamiltonian(3,3)
mps_ph = mps.parent_hamiltonian(3,3)

print(em.A.reshape(2,-1))