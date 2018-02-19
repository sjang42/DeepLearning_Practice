import numpy as np
from tools import sigmoid
from costFunction import costFunction


def gradientDescent(X, Y, Theta, learninRate, numIter):

    m = X.shape[0]
    for i in range(numIter):
        
        H = sigmoid(np.matmul(X, np.transpose(Theta)))
        Theta = Theta - learninRate / m * np.matmul(np.transpose(H - Y), X)

        cost = costFunction(X, Y, Theta)
        if (i % 100 == 0):
            print(i, ":", cost)
    return (Theta)
