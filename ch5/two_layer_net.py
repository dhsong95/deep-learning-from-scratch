
from collections import OrderedDict

import numpy as np

from layer_naive import AffineLayer, ReLULayer, SoftmaxWithLossLayer


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(W=self.params['W1'], b=self.params['b1'])
        self.layers['ReLU'] = ReLULayer()
        self.layers['Affine2'] = AffineLayer(W=self.params['W2'], b=self.params['b2'])
        self.last_layer = SoftmaxWithLossLayer()

    def _numerical_gradient(self, f, x):
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

            grad[idx] = (fh2 - fh1) / (2 * h)

            x[idx] = temp

            it.iternext()

        return grad


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def accuracy(self, x, y):
        y_hat = self.predict(x)
        return np.mean(np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)))

    def loss(self, x, y):
        y_hat = self.predict(x)
        loss = self.last_layer.forward(y_hat, y)
        return loss

    def numerical_gradient(self, x, y):
        loss_f = lambda W: self.loss(x, y)
        grads = dict()

        grads['W1'] = self._numerical_gradient(loss_f, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_f, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_f, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_f, self.params['b2'])

        return grads

    def gradient(self, x, y):
        self.loss(x, y)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads