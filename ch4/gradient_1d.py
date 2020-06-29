"""
Authod: DHSong
Last Modified: 2020.06.29
Textbook: 밑바닥부터 시작하는 딥러닝 1 - 4장
Description: Gradient Method in 1 Variable
"""

def numerical_diff(f, x):
    '''
    Numerical Calculatino of Derivation

    Args:
        f (function): function. Conceptually Loss Function.
        x (float): Input. Conceptually Weights.
    Return:
        derivation (float): Output.
    '''
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def fuction_1(x):
    '''
    Function

    Args:
        x (float): Input
    Return:
        y (float): Output
    '''
    return 0.01 * x ** 2 + 0.1 * x

if __name__ == '__main__':
    print(numerical_diff(fuction_1, 5))
    print(numerical_diff(fuction_1, 10))