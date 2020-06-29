"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 5장
Description: Train Neural Network
"""

import sys
import os
sys.path.append(os.pardir)

import numpy as np

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNetwork

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

network = TwoLayerNetwork(input_size=28*28, hidden_size=50, output_size=10)
batch_size = 100
n_iter = 10000
train_size = x_train.shape[0]
learning_rate = 0.1
iter_per_epoch = n_iter // batch_size

train_loss_list = list()
train_acc_list = list()
test_acc_list = list()

for i in range(n_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)
    for key in grads.keys():
        network.params[key] -= (learning_rate * grads[key])
    
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)