def cross_entropy_error(y, t):
    delta = 1e-7
    loss = -np.sum(t * np.log(y + delta))
    return loss