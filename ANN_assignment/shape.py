import numpy as np
x = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
m = x.shape[1]
w1 = np.random.rand(2, 2)
w2 = ([[2, 2], [3, 3]])
z1 = np.dot(w1, x)


def linear(x):
    return np.sum(2, x)


print(linear(2))
