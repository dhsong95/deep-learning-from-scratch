"""
Author: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 4장
Description: Simple Neural Net with Calculating Gradients
"""

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy(real, prediction, onehot=True):
    batch_size = prediction.shape[0]
    if onehot:
        loss = -np.sum(real * np.log(prediction)) / batch_size
    else:
        loss = -np.sum(np.log(prediction[np.arange(batch_size), real.ravel()])) / batch_size
    
    return loss

def numerical_gradient(f, x):
    grad = np.zeros_like(x)
    h = 1e-4
    for idx in range(x.shape[0]):
        for jdx in range(x.shape[1]):
            temp = x[idx][jdx]
            x[idx][jdx] = temp + h
            fh1 = f(x)

            x[idx][jdx] = temp - h
            fh2 = f(x)

            grad[idx][jdx] = (fh1 - fh2) / (2 * h)
            x[idx][jdx] = temp
    return grad


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, y):
        z = self.predict(x)
        y_hat = softmax(z)
        loss = cross_entropy(y, y_hat)
        return loss


if __name__ == '__main__':
    net = SimpleNet()
    print(net.W)

    x = np.array([[0.6, 0.9]])
    y_hat = net.predict(x)
    print(y_hat)

    y = np.array([[0, 0, 1]])
    loss = net.loss(x, y)
    print(loss)

    f = lambda w: net.loss(x, y)
    grad = numerical_gradient(f, net.W)
    print(grad)