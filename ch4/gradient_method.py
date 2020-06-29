"""
Author: DHSong
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

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    '''
    Numerical Calculation of Gradient

    Args:
        f (function): function. Conceptually Loss Function.
        init_x (numpy array): Input. Conceptually Weights.
        lr (float): Learning Rate
        step_num (int): number of steps
    Return:
        x (numpy array): Newly updated x. Gradient Descent makes x, which results the smaller f output.
    '''
    x = init_x
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= (lr * grad)

    return x        

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
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))

    print(gradient_descent(function_2, init_x, lr=10., step_num=100))
    print(gradient_descent(function_2, init_x, lr=1e-10, step_num=100))