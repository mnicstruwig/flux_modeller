"""
Defines all the helper functions for performing the curve-fit of the flux-linkage curves.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from scipy.optimize import curve_fit


def fit_training_sample(func, training_sample_dict, use_bounds=True):
    """
    Fit a single training example to `func` using the `training_sample` dict and adds the
    fitted function parameters and performance metrics to the dictionary once computed.

    Parameters
    ----------
    func : function
        The function to fit to.
    training_sample_dict : dict
        Dictionary that contains the training sample data
    use_bounds: {True, False}
        Specifies whether to enforce positive and negative bounds of the parameters in `func`.

    Returns
    -------
    dict
        Dictionary containing the fitted parameters, as well as performance metrics describing the quality of the fit.
    """
    xdata = training_sample_dict['dataframe']['displacement(m)']
    ydata = training_sample_dict['dataframe']['flux_linkage']

    # TODO: Implement way of specifying bounds
    if use_bounds:
        lower_bounds = (0, 0)  # (C, a, b)
        upper_bounds = (np.inf, np.inf)
        bounds = (lower_bounds, upper_bounds)
        popt, pcov = curve_fit(func, xdata, ydata, maxfev=10000, method='trf', bounds=bounds, xtol=1e-25, gtol=1e-25,
                               loss='huber')
    else:
        popt, pcov = curve_fit(func, xdata, ydata, maxfev=10000, method='trf', xtol=1e-25, gtol=1e-25)

    y_fit = func(xdata, *popt)

    training_sample_dict['popt'] = popt
    training_sample_dict['r2_score_curve_fit'] = r2_score(ydata, y_fit)
    training_sample_dict['mae_score_curve_fit'] = mean_absolute_error(ydata, y_fit)
    training_sample_dict['mse_score_curve_fit'] = mean_squared_error(ydata, y_fit)

    return training_sample_dict


def fit_all_training_samples(func, training_samples_list, **kwargs):
    """
    Fit all training samples in `training_samples_list` to `func`. Adds fitted function parameters and performance
    metrics to each sample, and returns the fitted list.

    Parameters
    ----------
    func : function
        The function to fit to.
    training_samples_list: list
        List of dicts that contains the training sample data
    **kwargs
        Keyword arguments for `fit_training_sample`.

    Returns
    -------
    list
        List containing dicts with the added fitted parameters and performance metrics.
    """

    curve_fitted_samples = []
    use_bounds = kwargs['use_bounds']

    for sample_dict in tqdm(training_samples_list):
        trained_sample_dict = fit_training_sample(func, sample_dict, use_bounds=use_bounds)

        curve_fitted_samples.append(trained_sample_dict)

    return curve_fitted_samples
