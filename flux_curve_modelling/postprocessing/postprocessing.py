"""
Helper functions used for performing postprocessing tasks
"""
import pandas as pd
from collections import OrderedDict


def _fetch_key_values_to_index(dict_, keys):
    """
    Extracts the values of the keys in `keys` from `dict_` and returns them as a tuple
    """
    return tuple([dict_[key] for key in keys])


def create_score_heatmap_from_dataframe(curve_fitted_samples, score_key):
    """
    Extract the performance metric data from all the fitted samples in `curve_fitted_samples` and return a new
    dataframe containing the score data in a schema that allows for immediate plotting of a heatmap, with axis
    parameters being `winding_num_z` and `winding_num_r`.

    Parameters
    ----------
    curve_fitted_samples : list
        List containing dictionaries with the fitted samples
    score_key : str
        String that specifies which key for the samples in `curve_fitted_samples` must be used as the performance
        metric for the heatmap.

    Returns
    -------
    dataframe
        Dataframe containing the score metrics with axis `winding_num_z` and `winding_num_r`.
    """
    df_err = pd.DataFrame()
    for sample in curve_fitted_samples:
        # TODO: Allow selection of axes to use
        df_dict = {'winding_num_z': sample['winding_num_z'],
                   'winding_num_r': sample['winding_num_r'],
                   'score': sample[score_key]}
        df_err = df_err.append(df_dict, ignore_index=True)

    return df_err


def create_dict_with_keys(dict_list, keys_to_index):
    """
    Converts a list of dictionaries into a single dictionary with specified with `keys_to_index`.
    The specified keys in `keys_to_index` must be in each dict in `dict_list`.

    Parameters
    ----------
    dict_list: list
        A list of dicts
    keys_to_index: list
        A list of strings representing the keys in the dicts in `dict_list` that will be used to index the new dict.

    Returns
    -------
    dict
        A dictionary containing all the elements of `dict_list` indexed using `keys_to_index`
    """
    database = OrderedDict()

    for dict_ in dict_list:
        new_key = _fetch_key_values_to_index(dict_, keys_to_index)
        data = dict_

        database[new_key] = data

    return database

