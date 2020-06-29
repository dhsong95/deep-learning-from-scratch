"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 3장
Description: Activation Fuction: Step Function
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
    return 1. / (1. + np.exp(-x))

def relu(x):
    '''
    Relu Function.

    Args:
        x (numpy array): input array
    Return:
        y (numpy array): output array
    '''
    return np.max(0, x)

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
    return exp_x / np.sum(exp_x)