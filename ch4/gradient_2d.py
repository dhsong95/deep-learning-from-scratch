"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 4장
Description: Gradient Method in 2 Variables
"""

import numpy as np

def numerical_gradient(f, x):
    '''
    Numerical Calculation of Gradient

    Args:
        f (function): function. Conceptually Loss Function.
        x (numpy array): Input. Conceptually Weights.
    Return:
        grad (numpy array): Output.
    '''
    grad = np.zeros_like(x)

    h = 1e-4
    for idx in range(x.size):
        temp = x[idx]

        x[idx] = temp + h
        fh1 = f(x)

        x[idx] = temp - h
        fh2 = f(x)
        
        grad[idx] = (fh1 - fh2) / (2 * h)
        x[idx] = temp
    
    return grad

def function_2(x):
    '''
    Function

    Args:
        x (numpy array): Input
    Return:
        y (numpy array): Output
    '''
    return np.sum(np.square(x))

if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))