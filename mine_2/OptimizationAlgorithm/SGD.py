class sgd(object):
    def __init__(self, lr):
        self.lr = lr
    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]
        return params
