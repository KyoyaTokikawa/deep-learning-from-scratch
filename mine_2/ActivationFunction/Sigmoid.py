import numpy as np
class sigmoid(object):
    def __init__(self, x):
        self.x = x

    def forward(self):
        output = 1.0 / (1.0 + np.exp(-self.x))
        self.output = output
        return output

    def backword(self, grad):
        grad_x = grad*(self.output*(1.0-self.output))
        return grad_x
