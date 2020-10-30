"""
Helper functions used for performing postprocessing tasks
"""
import numpy as np
import pandas as pd
from collections import OrderedDict

from typing import Any


def get_predictions_from_linear_model(X: np.ndarray,
                                      reg: Any,
                                      kernel_space: np.ndarray) -> np.ndarray:
    """Calculate the predicted flux curves.

    Parameters
    ----------
    X : array(n, d)[float]
        The input training data consisting of `n` samples and `d` dimensions.
    reg : LinearRegression
        The fitted LinearRegression model that predicts the kernel weights from `X`
    kernel_space : array(p, k)[float]
        The kernel space

    Returns
    -------
    array(n, p)[float]
        The predicted flux curves.

    """
    kernel_weights_hat = reg.predict(X)
    y_hat = kernel_space.dot(kernel_weights_hat.T)
    return np.array(y_hat).T
