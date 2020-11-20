"""
This file contains the preprocessing methods for the raw flux-linkage data that is exported from ANSYS Maxwell.
"""

import peakutils
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from flux_modeller.utilities.raw_csv_utils import get_parameters_dict
from typing import Dict, Any, List, Tuple


def _get_peak_index(df, col):
    """
    Return the index of the peak of the waveform.

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

    indexes = peakutils.indexes(np.abs(sample), thres=0.2, min_dist=15)
    return indexes[0]


def _shift_index_value_to_zero(df, col, index):
    """
    Shift the value in column `col` of `df` so that the value at `index`
    becomes zero and return the column.

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
    pandas series
        The shifted column

    """
    amount_to_shift = df.loc[index, col]
    return df[col] - amount_to_shift


# TODO: Implement catching cases where the `window_length` is longer than the number of samples in `ydata`.
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


def _smooth_flux_linkage(df, window_length):
    df.iloc[:, 1] = _smooth_signal(df.iloc[:, 1].values, window_length=window_length)
    return df


def smooth_column(df, col, **_smooth_signal_kwargs):
    """
    Smooth a series in column `col` in a `df` and return the dataframe.

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
    dataframe
        return the dataframe containing the smoothed_column

    """
    window_length = _smooth_signal_kwargs['window_length']
    df[col] = _smooth_signal(df[col], window_length)
    return df


def _calculate_time_step(df, time_col):
    """
    Calculate the timestep of  series `time_col` in dataframe `df`.

    Parameters
    ----------
    df: dataframe
        Pandas dataframe containing the timesteps
    time_col: str
        Column in `df` that contains the timesteps

    Returns
    -------
    float
        The timestep.
    """

    return np.diff(df[time_col].values)[0]


def create_training_sample(df: pd.DataFrame,
                           col: str,
                           shift_peak_to_zero: bool=True,
                           smooth_filter: bool=False) -> Dict[str, Any]:
    """
    Extract the flux-linkage waveform at `col` in `df`, and return a dict
    containing this waveform, the corresponding magnet displacement and the
    design parameters that produced the waveform.

    Parameters
    ----------
    df : dataframe
        The dataframe containing the raw Maxwell data
    col : str
        The column name in `df` that contains the flux-linkage waveform to be
        extracted.
    shift_peak_to_zero : bool
        Whether to shift the peak in df[col] to zero.
        Default value is True.
    smooth_filter : bool
        Whether to smooth the flux-linkage waveform at df[col]
        Default value is False.

    Returns
    -------
    dict
        Dictionary containing the new extracted dataframe (consisting of the
        magnet displacement and the flux-linkage), and the design parameters
        that produced the waveform.

    """
    new_df = pd.DataFrame()
    new_df['displacement(m)'] = df['displacement(m)']
    new_df['flux_linkage'] = np.abs(df[col]) # type: ignore

    timestep = _calculate_time_step(df, 'time(s)')

    if smooth_filter:
        new_df = smooth_column(df=new_df, col='flux_linkage', window_length=11)

    if shift_peak_to_zero:
        index = _get_peak_index(new_df, 'flux_linkage')
        new_df['displacement(m)'] = _shift_index_value_to_zero(new_df, 'displacement(m)', index)


    dict_: Dict[str, Any] = get_parameters_dict(col, winding_diameter=0.127) # TODO: Handle this hard-coding
    dict_['dataframe'] = new_df
    dict_['timestep'] = timestep
    return dict_


def create_sklearn_training_data(training_samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Create training dataset for passing to Scikit-learn models.

    Parameters
    ----------
    training_samples: List[Dict]
        The list of training samples generated by `create_all_training_samples`.

    Returns
    -------
    X : array(n, d)
        The input data of the scikit-learn model.
    y : array(n, p)
        The target data of the scikit-learn model.


    """
    X = []
    y = []

    for sample in training_samples:
        X.append(np.array([sample['winding_num_z'], sample['winding_num_r']]))
        y.append(sample['dataframe']['flux_linkage'].values)
    return np.array(X), np.array(y)


def create_training_dataset(df, waveform_columns, **kwargs):
    """
    Extracts the flux-linkage waveform at `col` in `df`, and returns a dict
    containing this waveform, the corresponding magnet displacement and the
    design parameters that produced the waveform.

    Parameters
    ----------
    df : dataframe
        The dataframe containing the raw Maxwell data
    waveform_columns : str
        The column name in `df` that contains the flux-linkage waveform to be
        extracted.
    **kwargs
        Keyword arguments passed to `create_training_sample` function.

    Returns
    -------
    list of dicts
        List of dicts containing the extracted dataframes (consisting of the
        magnet displacement and the flux-linkage), and the design parameters
        that produced each respective waveform.

    """
    training_samples = []
    for col in tqdm(waveform_columns):
        training_samples.append(
            create_training_sample(
                df,
                col,
                **kwargs
            )
        )

    return training_samples
