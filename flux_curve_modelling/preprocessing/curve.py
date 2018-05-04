"""
This file contains the preprocessing methods for the raw flux-linkage data that is exported from ANSYS Maxwell.
"""

import peakutils
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from flux_curve_modelling.utilities.raw_csv_utils import get_parameters_dict


def _get_peak_index(df, col, flipped=False):
    """
    Return the index of the peak of the waveform

    Parameters
    ----------
    df : dataframe
        The dataframe containing all the raw Maxwell data
    col : str
        The name of the column in `df` whose peak index must be found.
    flipped : {False, True}
        Specifies whether the sample has a negative peak.

    Returns
    -------
    int
        The index of the peak
    """
    sample = df[col]

    if flipped:
        sample = -1 * sample

    indexes = peakutils.indexes(sample, thres=0.2, min_dist=15)
    return indexes[0]


def _shift_index_value_to_zero(df, col, index):
    """
    Shift the value in `col` of `df` so that the value at `index` becomes zero and return the column.

    Parameters
    ----------
    df : dataframe
        The dataframe containing all the raw Maxwell data
    col : str
        The name of the column in `df` whose peak must be shifted to zero
    index : int
        The index of the peak

    Returns
    -------
    series
        The shifted column
    """
    amount_to_shift = df.loc[index, col]
    return df[col] - amount_to_shift


def _smooth_signal(ydata, window_length):
    """
    Smooth a signal by convolving it with a normalized vector of length `window_length`

    Parameters
    ----------
    ydata : array_like
       The discrete data representing the signal that must be smoothed.
    window_length : int
        The length of the normalized window that must be convolved with `ydata`

    Returns
    -------
    ndarray
        The smoothed signal.
    """
    convolution_box = np.ones(window_length) / window_length

    return np.convolve(ydata, convolution_box, mode='same')


def smooth_column(df, col, **_smooth_signal_kwargs):
    """
    Smooth a series from `col` in a `df` and return the smoothed column.

    Parameters
    ----------
    df : dataframe
        The dataframe containing the raw Maxwell data.
    col : str
        The name of the column in `df` that must be smoothed.
    **_smooth_signal_kwargs
        Keyword arguments to be passed to `_smooth_signal`.
    Returns
    -------
    series
        The smoothed column
    """
    window_length = _smooth_signal_kwargs['window_length']

    df[col] = _smooth_signal(df[col], window_length)

    return df[col]


def create_training_sample(df, col, shift_peak_to_zero=True, smooth_filter=False, minmaxscale=False):
    """
    Extracts the flux-linkage waveform at `col` in `df`, and returns a dict containing this waveform, the
    corresponding magnet displacement and the design parameters that produced the waveform.

    Parameters
    ----------
    df : dataframe
        The dataframe containing the raw Maxwell data
    col : str
        The column name in `df` that contains the flux-linkage waveform to be extracted.
    shift_peak_to_zero : {True, False}
        Whether to shift the peak in df[col] to zero.
    smooth_filter : {False, True}
        Whether to smooth the flux-linkage waveform at df[col]
    minmaxscale : {False, True}
        Whether to apply a MinMaxScaler to the curve (i.e. scale curve to min=0 and max=1). If True, will also append
        to the training sample dict under the 'minmaxscaler' key for later inverse-transformation back to original
        curve.

    Returns
    -------
    dict
        Dictionary containing the new extracted dataframe (consisting of the magnet displacement and the flux-linkage),
        and the design parameters that produced the waveform.
    """
    new_df = pd.DataFrame()
    new_df['displacement(m)'] = df['displacement(m)']
    # TODO: Implement a flag and way of flipping the raw curve, rather than hard-coded
    new_df['flux_linkage'] = -1 * df[col]

    # Pass through smoothing filter
    if smooth_filter:
        new_df['flux_linkage'] = smooth_column(df=new_df, col='flux_linkage', window_length=11)

    # Shift peak of curve to zero
    if shift_peak_to_zero:
        index = _get_peak_index(new_df, 'flux_linkage', flipped=False)
        new_df['displacement(m)'] = _shift_index_value_to_zero(new_df, 'displacement(m)', index)

    # MinMaxScaling
    mms = None
    if minmaxscale:
        mms = MinMaxScaler()
        y_temp = new_df['flux_linkage']
        y_temp = y_temp.reshape(-1, 1)
        new_df['flux_linkage'] = mms.fit_transform(y_temp)

    dict_ = get_parameters_dict(col, winding_diameter=0.127)
    dict_['dataframe'] = new_df
    dict_['minmaxscaler'] = mms

    return dict_


def create_all_training_samples(df, waveform_columns, **kwargs):
    """
    Extracts the flux-linkage waveform at `col` in `df`, and returns a dict containing this waveform, the
    corresponding magnet displacement and the design parameters that produced the waveform.

    Parameters
    ----------
    df : dataframe
        The dataframe containing the raw Maxwell data
    waveform_columns : str
        The column name in `df` that contains the flux-linkage waveform to be extracted.
    **kwargs
        Keyword arguments for `create_training_sample`.

    Returns
    -------
    list of dicts
        List tof dicts containing the extracted dataframes (consisting of the magnet displacement and the flux-linkage),
        and the design parameters that produced each respective waveform.
    """
    # TODO: Implement better kwarg handling
    shift_peak_to_zero = kwargs['shift_peak_to_zero']
    smooth_filter = kwargs['smooth_filter']
    minmaxscale = kwargs['minmaxscale']

    training_samples = []
    for col in tqdm(waveform_columns):
        training_samples.append(create_training_sample(df,
                                                       col,
                                                       shift_peak_to_zero=shift_peak_to_zero,
                                                       smooth_filter=smooth_filter,
                                                       minmaxscale=minmaxscale)
                                )

    return training_samples
