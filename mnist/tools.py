import numpy as np
import struct

# load mnist data

def load_mnist(num_image, path_to_data, is_test=False):
    if (is_test):
        f_image = open(path_to_data + '/t10k-images.idx3-ubyte','rb')
        f_label = open(path_to_data + '/t10k-labels.idx1-ubyte','rb')
    else:
        f_image = open(path_to_data + '/train-images.idx3-ubyte','rb')
        f_label = open(path_to_data + '/train-labels.idx1-ubyte','rb')

    X = np.zeros((10000, 28*28))
    Y = np.zeros((10000, 10))

    img = np.zeros((28, 28))

    s = f_image.read(16)    #read first 16byte
    l = f_label.read(8)     #read first  8byte

    for i in range(num_image):
        label = np.zeros((1, 10))
        s = f_image.read(28*28)
        l = f_label.read(1)

        img = np.reshape(struct.unpack(len(s) * 'B', s), (1, 28*28))
        l = struct.unpack('B', l)
        label[0][l] = 1

        X[i] = img
        Y[i] = label

    f_image.close()
    f_label.close()
    return X, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(X, Theta):
    H = sigmoid(np.matmul(X, np.transpose(Theta)))
    pred = np.argmax(H, axis=1)

    return pred
