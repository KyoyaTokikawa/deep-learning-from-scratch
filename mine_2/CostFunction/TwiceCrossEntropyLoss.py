def binary_cross_entropy(y, t):
    loss = -1 * (t*np.log(y) + (1-t)*np.log(1-y))