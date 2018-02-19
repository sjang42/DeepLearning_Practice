#!/Users/sjang/anaconda3/bin/python
import numpy as np
from tools import normalization, gradientDescent
from costFunction import costFunction

# loading data and set X, Y, Theta
data = np.genfromtxt('ds_bostonHousing_train.tsv', delimiter='\t')

X = data[:, 1:-1]
Y = np.reshape(data[:, -1], (X.shape[0], 1))
Theta = np.zeros((X.shape[1] + 1, 1))

# # Mean normalization
X_nor, mu, std = normalization(X)

# append 1 column to X
X_nor = np.concatenate((np.ones((X_nor.shape[0],1)), X), axis=1)
# set hyper
learningRate = 0.001
numIter = 10000

# train with gradientDescent
cost = costFunction(X_nor, Y, Theta)
print('cost before training on training set :', cost)

Theta = gradientDescent(X_nor, Y, Theta, learningRate, numIter)
cost = costFunction(X_nor, Y, Theta)
print('cost after training on training set :', cost)

# test
data_test = np.genfromtxt('ds_bostonHousing_test.tsv', delimiter='\t')
X_test = data_test[:, 1:-1]
X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis=1)
Y_test = np.reshape(data_test[:, -1], (X_test.shape[0], 1))

cost = costFunction(X_test, Y_test, Theta)
print('cost after training on test set :', cost)

# kaggle test
data_test = np.genfromtxt('kaggle_test.csv', delimiter=',')
ID = np.reshape(data_test[:, 0], (data_test.shape[0], 1))
X_test = data_test[:, 1:]
X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis=1)

Y = np.matmul(X_test, Theta)
output = np.append(ID, Y, axis=1)
np.savetxt('submit.csv', output, delimiter=',', fmt='%.8f')
