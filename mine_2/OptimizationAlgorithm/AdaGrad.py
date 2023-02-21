import numpy as np
class adagrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for key, param in params:
                self.h[key] = np.zeros.like(param)
        for key in params.key():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
        return params