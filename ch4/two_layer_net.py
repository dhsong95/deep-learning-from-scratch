"""
Author: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 4장
Description: Simple Two Layer Network. 
"""

import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy(r, p):
    batch_size = r.shape[0]
    return -np.sum(r * np.log(p)) / batch_size

def numerical_gardient(f, x):
    grad = np.zeros_like(x)
    h = 1e-4

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        temp = x[idx]

        x[idx] = temp + h
        fh1 = f(x)

        x[idx] = temp - h
        fh2 = f(x)

        grad[idx] = (fh1 - fh2) / (2 * h)

        x[idx] = temp 

        it.iternext()
    return grad

class TwoLayerNet:
    def __init__(self, input_size=784, hidden_size=100, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)  
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y_hat = softmax(a2)

        return y_hat

    def loss(self, x, y):
        y_hat = self.predict(x)
        loss = cross_entropy(y, y_hat)

        return loss

    def accuracy(self, x, y):
        y_hat = self.predict(x)
        return np.mean(np.equal(np.argmax(y_hat, axis=-1), np.argmax(y, axis=-1)))

    def numerical_gardient(self, x, y):
        loss_W = lambda W: self.loss(x, y)

        grads = dict()
        grads['W1'] = numerical_gardient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gardient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gardient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gardient(loss_W, self.params['b2'])

        return grads
    
