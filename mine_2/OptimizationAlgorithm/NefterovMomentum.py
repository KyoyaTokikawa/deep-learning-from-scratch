import numpy as np
class nefterovmomentumSGD:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for key, val in params.items():
                self.v[key] = np.zeros.like(val)
        for key in params.key():
            self.v[key] = self.momentum * self.v[key] - self.lr * (grads[key] + self.momentum * self.v[key])
            params[key] += self.v[key]
        return params
