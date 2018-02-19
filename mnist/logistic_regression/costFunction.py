import numpy as np
from tools import sigmoid


import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


def costFunction(X, Y, Theta):
    m = X.shape[0]

    H = sigmoid(np.matmul(X, np.transpose(Theta)))
    # print(H.shape)
    J = -1 / m * np.sum(Y * np.log(H + 0.001) + (1 - Y) * np.log(1 - H + 0.001), axis=0)
    J = np.reshape(J, (Theta.shape[0], 1))
    return J
