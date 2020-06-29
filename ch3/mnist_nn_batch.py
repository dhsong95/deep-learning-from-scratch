"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 3장
Description: MNIST with Neural Network. Pretrained Weights. Batch Feed Forwarding.
"""
import os
import sys
sys.path.append(os.pardir)

import pickle
import numpy as np

from dataset.mnist import load_mnist
from activations import sigmoid, softmax

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return (x_train, y_train), (x_test, y_test)

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
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
    (_, _), (x, y) = get_data()
    network = init_network()
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        y_hat = predict(network, x[i:i + batch_size])
        p = np.argmax(y_hat, axis=-1)
        accuracy_cnt += np.sum(np.equal(p, y[i:i + batch_size]))
    print('Accuracy: {:.4f}'.format(accuracy_cnt / len(x)))      