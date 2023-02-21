import numpy as np

def step(x):
    return np.array(x>0, dtype=np.int)