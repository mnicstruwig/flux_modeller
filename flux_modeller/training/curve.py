"""
Defines all the helper functions for performing the curve-fit of the flux-linkage curves.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def _gauss_rbf(x, mag=1, center=0, shape=0.003):
    """Sample from a Gaussian radial basis function kernel.

    Parameters
    ----------
    x : float
        The position at which to sample from the kernel.
    mag : float
        The peak magnitude of the kernel.
    center : float
        The center of the peak of the kernel.
    shape : float
        The shape parameter of the kernel.

    Returns
    -------
    float
        The value of the kernel at position `x`

    """
    return mag*np.exp(-(x-center)**2/(2*shape**2))


def _build_unweighted_kernel_space(k, xs, mag, shape):
    """Build an unweighted kernel space.

    Parameters
    ----------
    k : int
        The number of kernels to distribute across the space.
        Consider this a hyperparameter.
    xs : array[float]
        The points at which to sample each kernel.
    mag : float
        The magnitude of the kernel.
    shape: float
        The shape parameter that controls the shape of the RBF kernel.
        Consider this a hyperparameter.

    Returns
    -------
    array(len(xs), k)
        The kernel values at each point of `xs`.

    """
    try:
        # Add two, since we remove the first and last.
        peak_locations = np.linspace(min(xs), max(xs), k + 2)
    except TypeError:
        peak_locations = np.linspace(min(xs), max(xs), k.value + 2)
    peak_locations = peak_locations[1:-1]  # Trim off the end kernels

    kernels = []
    for p in peak_locations:
        kernels.append([_gauss_rbf(x, mag=mag, center=p, shape=shape) for x in xs])
    kernels = np.array(kernels)
    return kernels.T


def _get_kernel_weights(kernel_space: np.ndarray,
                        y: np.ndarray) -> np.ndarray:
    """Determine the best kernel weights to reproduce curve `y`.

    This is calculated using Least Squares / Linear Regression.

    Parameters
    ----------
    kernels: array(n, k)
        The unweighted kernel space.
    y : array[float]
        The target curve to approximate.

    Returns
    -------
    array(k, )[float]
        The weights of each kernel that best recreates `y`.

    """
    kernel_weights = []
    for y_s in y:
        reg = LinearRegression().fit(kernel_space, y_s)
        kernel_weights.append(reg.coef_)

    return np.array(kernel_weights)


def fit_linear_model(X, y, xs, n_kernels, kernel_magnitude, kernel_shape):
    """Train the kernel weights estimator model

    Parameters
    ----------
    X : array(n, d)
        The input training data consisting of `n` samples and `d` dimensions.
    y : array(n, p)
        The target curves, consisting of `n` curves, each consisting of `p` points.
    xs : array(p, )
        The `p` points that the kernel space should be sampled at.
    n_kernels : int
        The number of kernels to use. Consider this a hyperparameter.
    kernel_magnitude : float
        The magnitude of each kernel.
    kernel_shape: float
        The shape parameter of the RBF kernel. Consider this a hyperparameter.

    Returns
    -------
    reg : LinearRegression
        The LinearRegression model that maps from `X` to the kernel weights.
    kernel_space : array(len(xs), k)
        The kernel values at each point of `xs`.
    kernel_weights : array(k, )[float]
        The weights of each kernel that best recreates `y`.

    """
    kernel_space = _build_unweighted_kernel_space(
        k=n_kernels,
        xs=xs,
        mag=kernel_magnitude,
        shape=kernel_shape
    )
    kernel_weights = _get_kernel_weights(kernel_space, y)

    reg = LinearRegression()
    reg = reg.fit(X, kernel_weights)

    return reg, kernel_space, kernel_weights
