"""
Helper functions for postprocessing predictions for the parameter mapping from the device design parameters to the
flux-curve model parameters
"""
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_squared_log_error, mean_absolute_error, \
    r2_score


def inverse_normalize(input_, fitted_scaler):
    """Run the `inverse_transform` function from fitted_scaler on `input_`."""
    return fitted_scaler.inverse_transform(input_[0])


def calculate_r2_score(input_):
    """Calculate the R2 score for the two vectors in `input_`"""
    return r2_score(input_[0], input_[1])


def calculate_mse(input_):
    """Calculate the mean-squared-error for the two vectors in `input_`"""
    return mean_squared_error(input_[0], input_[1])


def calculate_msle(input_):
    """Calculate the mean-squared-log-error for two vectors in `input_`"""
    return mean_squared_log_error(input_[0], input_[1])


def calculate_mae(input_):
    """Calculate the mean-absolute-error for two vectors in `input_`"""
    return mean_absolute_error(input_[0], input_[1])


def calculate_explained_variance_score(input_):
    """Calculate the mean-squared-error for the two vectors in `input_`"""
    return explained_variance_score(input_[0], input_[1])


def ravel_arr(input_):
    """Flattens an array to (n, 1) dimensions."""
    return np.ravel(input_[0])


def compute_func_score_against_curve(input_, compute_func, scorer):
    """
    Uses `scorer` to score the quality of a curve fit with model `func` for a curve in a dataframe at `input_[0]`
    and any curve model parameters at `input_[1]`
    """
    df = input_[0]  # Original xdata
    xdata = df['displacement(m)']
    ydata = df['flux_linkage']

    popt = input_[1]  # Parameters for func

    y_pred_curve = compute_func(xdata, *popt)  # Make a curve

    return scorer(ydata, y_pred_curve)  # Score vs. original curve
