import emachine_v2 as em
import numpy as np
PITY = 3

g = em.gacha(p=0.1, pity=PITY)
initial_state = np.zeros((PITY,))
initial_state[0] = 1
print(g.interaction_rank(5))
