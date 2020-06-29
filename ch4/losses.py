"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 4장
Description: Loss Fuction
"""

import numpy as np

def mse(real, prediction):
    '''
    Mean Squared Error.

    Args:
        real (numpy array): real array
        prediction (numpy array): prediction array
    Return:
        loss (numpy array): Sum of Squared Error
    '''
    loss = 0.5 * np.sum(np.square(real - prediction))
    return loss

def cross_entropy(real, prediction, onehot=True):
    if real.ndim == 1:
        real = real.reshape(1, -1)
        prediction = prediction.reshape(1, -1)
    
    delta = 1e-7
    batch_size = real.shape[0]
    if onehot:
        loss = -np.sum(real * np.log(prediction + delta)) / batch_size
    else:
        real = real.reshape(-1)
        loss = -np.sum(np.log(prediction[:, real])) / batch_size
    return loss


if __name__ == '__main__':
    real = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    prediction = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy(real, prediction, onehot=True))
    print(mse(real, prediction))
    print()
    prediction = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(cross_entropy(real, prediction, onehot=True))
    print(mse(real, prediction))
    