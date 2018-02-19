import numpy as np

def costFunction(X, Y, theta):
    m = X.shape[0]
    hypothesis = np.matmul(X, theta)
    err = np.square(hypothesis - Y)
    cost = 1 / (2 * m) * err.sum()

    return (cost)
