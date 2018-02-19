# main.py
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from tools import load_mnist, predict
from costFunction import costFunction
from train import gradientDescent
import numpy as np


# load data
# set X, Y, Theta
X, Y = load_mnist(1000, '../data/')
Theta = np.zeros((Y.shape[1], X.shape[1] + 1))
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

print(Theta.shape)
# set hyper parameters
learningRate = 0.001
numIter = 3000

# set cost function
cost = costFunction(X, Y, Theta)
print("before : ", cost)

# set gradient descent
Theta = gradientDescent(X, Y, Theta, learningRate, numIter)

cost = costFunction(X, Y, Theta)
print("after : ", cost)

# test
X_test, Y_test = load_mnist(1000, '../data/', is_test=True)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

pred = predict(X, Theta)
print(pred)

real = np.argmax(Y_test, axis=1)
print(real)

















