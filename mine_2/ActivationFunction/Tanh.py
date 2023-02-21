import numpy as np
def tanh(x):
    y=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y

def tanh_derivative(x):
    y= 4 / (np.exp(x) + np.exp(-x))**2