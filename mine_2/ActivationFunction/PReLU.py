import numpy as np
def prelu(x, p):
    y=np.where(x > 0, x, p*x)
    return y