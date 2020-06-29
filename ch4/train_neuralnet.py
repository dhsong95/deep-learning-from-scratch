"""
Author: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 4장
Description: Train Two Layer Network. 
"""

import sys
import os
sys.path.append(os.pardir)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

iters_num = 10000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

train_loss_list = list()
for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grad = network.numerical_gardient(x_batch, y_batch)

    loss = network.loss(x_batch, y_batch)
    print('[{:8d}] Before Loss: {:.4f}'.format(i + 1, loss))

    for key in ['W1', 'W2', 'b1', 'b2']:
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, y_batch)
    print('[{:8d}] After Loss: {:.4f}'.format(i + 1, loss))
    train_loss_list.append(loss)

plt.figure(figsize=(16, 9))
plt.plot(train_loss_list)
plt.title('Train Loss')
plt.show()