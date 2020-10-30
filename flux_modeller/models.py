"""
This file contains the models that are used to model the flux-linkage curves
"""

import numpy as np


def _sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * x + b))


def subtracted_sigmoids(x, a, b):
    return _sigmoid(x, a, -b) - _sigmoid(x, a, b)


def d_sigmoid(x, C, a):
    return C * _sigmoid(x, a, b=0) * (1 - _sigmoid(x, a, b=0))


def d_sigmoid_mod(x, C, a):
    return C / (1 + np.exp(-a * x * x)) * (1 - 1 / (1 + np.exp(-a * x * x)))


def d_sigmoid_mod_pow_4(x, C, a):
    return C / (1 + np.exp(-a * x * x * x * x)) * (1 - 1 / (1 + np.exp(-a * x * x)))


def tanh_model(x, a, b):
    return np.tanh(a * x + b) - np.tanh(a * x - b)


def d_tanh(x, C, a):
    return C * (1 - (np.tanh(a * x) * np.tanh(a * x)))


def d_tanh_power(x, a, b):  # Explore this at a later date using brute-force (b must be integer!)
    return 1 - (np.tanh(a * x ** b) * np.tanh(a * x ** b))
