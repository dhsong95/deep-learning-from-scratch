"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 5장
Description: Layer with forward and backward
"""

import numpy as np

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y
        
    def backward(self, dout):
        dx = dout
        dx = self.y * dout
        dy = self.x * dout

        return dx, dy


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1. / (1. + np.exp(x))
        self.out = out
        return self.out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx

class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
    
    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        return dx

class SoftmaxWithLossLayer:
    def __init__(self):
        self.y = None
        self.y_hat = None

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1).reshape(-1, 1))
        return exp_x / np.sum(exp_x, axis=-1).reshape(-1, 1)

    def _cross_entropy(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-4))

    def forward(self, x, y):
        self.y = y
        y_hat = self._softmax(x)
        self.y_hat = y_hat
        loss = self._cross_entropy(y, y_hat)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) / batch_size
        return dx