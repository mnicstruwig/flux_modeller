"""
This file contains all the functions used to engineer features.
"""
import numpy as np


def _calculate_coil_height(winding_num_z, winding_diameter):
    """Calculate coil height."""
    return winding_num_z * winding_diameter


def _calculate_coil_width(winding_num_r, winding_diameter):
    """Calculate coil width."""
    return winding_diameter * winding_num_r


def _calculate_n(winding_num_z, winding_num_r):
    """Calculate the number of windings."""
    return winding_num_z * winding_num_r


def _calculate_coil_density(winding_num_z, winding_num_r, winding_diameter):
    """Calculate the density of the coil."""
    coil_height = _calculate_coil_height(winding_num_z, winding_diameter)
    coil_width = _calculate_coil_width(winding_num_r, winding_diameter)
    N = _calculate_n(winding_num_z, winding_num_r)
    return N / (coil_height * coil_width)


def _calculate_width_to_height_ratio(winding_num_z, winding_num_r):
    """Calculate the width-to-height ratio of the coil."""
    return winding_num_r / winding_num_z


def _calculate_coil_area(winding_num_z, winding_num_r, winding_diameter):
    """Calculate the area of the coil."""
    coil_height = _calculate_coil_height(winding_num_z, winding_diameter)
    coil_width = _calculate_coil_width(winding_num_r, winding_diameter)

    return coil_height * coil_width


def _calculate_coil_diagonal_length(winding_num_z, winding_num_r, winding_diameter):
    """Calculate the hypotenuse of the coil's diagonal."""
    coil_height = _calculate_coil_height(winding_num_z, winding_diameter)
    coil_width = _calculate_coil_width(winding_num_r, winding_diameter)

    return np.sqrt((coil_height * coil_height) + (coil_width * coil_width))


def engineer_features(additional_features, **kwargs):
    """
    Engineers a number of features specified in `additional_features` for a *single* sample
    from values passed with `kwargs`.

    Parameters
    ----------
    additional_features : list
        List of strings specifying which features to engineer
    **kwargs :
        Keyword arguments for the parameters to engineer features from.
        Currently: `winding_num_r`, `winding_num_z` and `winding_diameter`.

    Returns
    -------
    Dict
        A dictionary of engineered features, with keys as in `additional_features`.
    """
    winding_num_r = kwargs['winding_num_r']
    winding_num_z = kwargs['winding_num_z']
    winding_diameter = kwargs['winding_diameter']

    features = dict()

    if 'coil_height' in additional_features:
        features['coil_height'] = _calculate_coil_height(winding_num_z, winding_diameter)
    if 'coil_width' in additional_features:
        features['coil_width'] = _calculate_coil_width(winding_num_r, winding_diameter)
    if 'coil_area' in additional_features:
        features['coil_area'] = _calculate_coil_area(winding_num_z, winding_num_r, winding_diameter)
    if 'N' in additional_features:
        features['N'] = _calculate_n(winding_num_z, winding_num_r)
    if 'coil_density' in additional_features:
        features['coil_density'] = _calculate_coil_density(winding_num_z, winding_num_r, winding_diameter)
    if 'w_to_h_ratio' in additional_features:
        features['w_to_h_ratio'] = _calculate_width_to_height_ratio(winding_num_z, winding_num_r)
    if 'diagonal_length' in additional_features:
        features['diagonal_length'] = _calculate_coil_diagonal_length(winding_num_z, winding_num_r, winding_diameter)

    return features


def get_x_y_from_dict_list(dict_list, additional_features):
    """
    Extract the training input `X` and training target `y` from dicts in `curve_fitted_samples`
    with additional features engineered, specified with `additional_features`.

    Parameters
    ----------
    dict_list : list
        List of dicts containing the design parameters and fitted model parameters of the FEA flux curves
    additional_features : list
        List of strings specifying which additional features must be engineered.

    Returns
    -------
    X : ndarray (n, d)
        Training data.
    y : ndarray (n, )
        Target data.
    """

    X = []
    y = []

    for sample in dict_list:
        x = [
            sample['winding_diameter'],
            sample['winding_num_r'],
            sample['winding_num_z']
        ]

        if additional_features is not None:
            features_dict = engineer_features(additional_features, **sample)

            for key, value in features_dict.items():
                x.append(value)

        X.append(x)
        y.append(sample['popt'])

    return np.array(X), np.array(y)
