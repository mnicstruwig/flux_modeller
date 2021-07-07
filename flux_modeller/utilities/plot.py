"""
Contains helper methods for making comparative plots
"""

import matplotlib.pyplot as plt
from plotnine import *


def plot_curve_fit_comparison(func, popt, xdata, ydata):
    """
    Generate datapoints using `xdata` for a function `func` with parameters `popt` and plot this alongside `ydata`.

    Parameters
    ----------
    func : function
        Function with parameters `*popt` that takes `xdata` as input and returns corresponding y values.
    popt : array_like
        Parameters for `func`
    xdata : array_like
        The actual x-values, corresponding to values in `ydata`
    ydata : array_like
        The actual y-values, corresponding to values in `xdata`

    Returns
    -------
    array_like
        The output of `func`, i.e. the predicted y-values that are estimated using `xdata`.
    """
    y_fit = func(xdata, *popt)
    plt.plot(xdata, ydata, "r", label="original")
    plt.plot(xdata, y_fit, "b", label="curve fit")
    plt.legend()
    return y_fit


def fast_comparison_plot_from_dataframe(df, y_columns):
    df = df.melt(value_Vars=y_columns)

    p = ggplot(aes(x=""))


def fast_curve_fit_comparison_plot(func, dict_):
    """
    A helper that allows for a fast comparison plot of a fitted curve for a known curve_fitted dictionary, using
    hard-coded values

    Parameters
    ----------
    func : function
        Function with parameters `*popt` that takes `xdata` as input and returns corresponding y values.
    dict_ : dict
        A single dictionary containing the parameters of `func` and an extracted dataframe containing the
         flux-linkage curve and magnet displacement
    """
    plot_curve_fit_comparison(
        func=func,
        popt=dict_["popt"],
        xdata=dict_["dataframe"]["displacement(m)"],
        ydata=dict_["dataframe"]["flux_linkage"],
    )
