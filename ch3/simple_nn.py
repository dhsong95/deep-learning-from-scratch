"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 3장
Description: Simple Neural Network
"""

import numpy as np

def step(x):
    '''
    Step Function.

    Args:
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    '''
    Sigmoid Function.

    Args:
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    return 1 / (1 + np.exp(-x))

def relu(x):
    '''
    Relu Function.

    Args:
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    return max(x, 0)

def identity(x):
    '''
    Identity Function.

    Args:
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    return x

def softmax(x):
    '''
    Softmax Function.

    Args:
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    exp_x = np.exp(x - np.max(x))
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


def init_network():
    '''
    Initialize Network with weights(and bias).

    Args:
    Return:
        network (dict): initialized weight dictionary
    '''
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    '''
    Feed Forward Network with input x.

    Args:
        network (dict): Initialized weights
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity(a3)

    return y


if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
