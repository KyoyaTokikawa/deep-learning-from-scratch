import numpy as np
def leakyrelu(x):
    y=np.where(x>0, x, 0.001*x)
    return y