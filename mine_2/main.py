import numpy as np
x = np.array([[1, 2, 3], [6, 5, 2], [4, 6, 3]])
l_index = np.argsort(x)[::-1]
print(l_index)