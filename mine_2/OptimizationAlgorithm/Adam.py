import numpy as np
def Adam_update(self, pred, y):
    grad = y*self.calc_dloss(y*pred)*self.features
    self.m = self.beta1*self.m + (1 - self.beta1)*grad
    self.v = self.beta2*self.v + (1 - self.beta2)*grad**2
    mhat = self.m/(1-self.beta1**self.t)
    vhat = self.v/(1-self.beta2**self.t)
    self.alpha *= np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
    self.weight -= self.alpha * mhat/(np.sqrt(vhat) + self.epsilon)