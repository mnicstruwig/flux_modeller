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


def insert_predicted_flux_curve(input_, X_col, curve_model):
    """
    Insert the predicted flux-linkage curve into a dataframe at `input_[0]` using parameters at `input_[1]`.

    Parameters
    ----------
    input_ : list
        input_[0] contains the dataframe with input to `curve_model` at column `X_col`
        input_[1] contains the predicted parameters for `curve_model`.
    X_col : str
        Column in the dataframe containing the input / x-values for `curve_model`.
    curve_model : func
        Function that predicts the flux linkage curve. Receives input as the first parameter, and model parameters
        as all subsequent models.

    Returns
    -------
    dataframe
        Updated dataframe with the predicted curve inserted under column with name `flux_linkage_pred`.
    """
    df = input_[0]
    pred_popt = input_[1]
    X = df[X_col].values  # "input" values for the `curve_model`

    predicted_curve = curve_model(X, *pred_popt)

    df['flux_linkage_pred'] = predicted_curve
    return df


def apply_func_to_dataframe_curve(input_, curve_col, func):
    """
    Apply a function `func` to a curve, denoted by column-name `curve_col` in a dataframe at `input_[0]`.

    Parameters
    ----------
    input_
    curve_col
    func

    Returns
    -------

    """
    df = input_[0]
    y = df[curve_col].values

    return func(y)


def inverse_transform_dataframe_curve(input_, curve_col):
    """
    Inverse-transform / unscale a curve in column `curve_vol` in dataframe at `input_[0]` using scaler at `input_[1]
    and insert this unscaled curve into the dataframe.

    Parameters
    ----------
    input_ : list
        input_[0] contains the dataframe that must contain a column `curve_col`.
        input_[1] contains the scaler that will be used to inverse-transform the curve. The `inverse_transform` method
        is called on this object.

    curve_col : Column in the dataframe containing the curve to be scaled.

    Returns
    -------
    dataframe
        Updated dataframe with the inverse-transformed curve insert under the column `curve_vol`_original

    """
    df = input_[0]
    scaler = input_[1]
    scaled_curve = df[curve_col].values.reshape(1, -1)
    unscaled_curve = scaler.inverse_transform(scaled_curve)[0]  # The entire array is in the first element

    df[curve_col + '_unscaled'] = unscaled_curve

    return df


def compute_score_dataframe_curves(input_, y_col, yhat_col, scorer):
    """
    Compute a score using `scorer` on curves given by columns `y_col` and `yhat_col` in dataframe at `input_[0]`

    Parameters
    ----------
    input_ : list
        input_[0] contains the dataframe containing the curves at columns `y_col` and `yhat_col`.
    y_col : str
        Column in the dataframe containing the ground-truth curve.
    yhat_col : str
        Column in the dataframe containing the predicted curve.
    scorer : func
        Function that computes a metric given the two curves.

    Returns
    -------
    float
        The score as computed by `scorer`.

    """
    df = input_[0]
    y = df[y_col]
    yhat = df[yhat_col]

    return scorer(y, yhat)


# TODO: Better docstring
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
