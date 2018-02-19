import numpy as np

def normalization(X):

    mu = X.mean(axis=0)
    std = X.std(axis=0)

    X_nor = (X - mu) / std
    return X_nor, mu, std

def gradientDescent(X, Y, Theta, learningRate, numIter):
    m = X.shape[0]
    for i in range(numIter):
        err = np.matmul(X, Theta) - Y

        Theta = Theta - learningRate / m *\
            learningRate * np.matmul(X.transpose(), err)

    return (Theta)
