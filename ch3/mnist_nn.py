"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 3장
Description: MNIST with Neural Network. Pretrained Weights
"""

import sys
import os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from activations import sigmoid, softmax

import pickle

import numpy as np

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return (x_train, y_train), (x_test, y_test)

def init_network():
    with open(file='sample_weight.pkl', mode='rb') as f:
        network = pickle.load(f)
    return network

def predict(network , x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    (_, _), (x, t) = get_data()
    network = init_network()   
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print('Accuracy: {:.4f}'.format(accuracy_cnt / len(x)))