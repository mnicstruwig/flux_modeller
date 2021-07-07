"""
Pre-processing helper functions for the processed raw Maxwell data and for preparation of training data.
"""


def normalize(data, scaler, **scaler_kwargs):
    """
    Normalize `data` using `scaler`

    Parameters
    ----------
    data : ndarray
        Data to be normalized
    scaler : cls
        Class used to normalize `data`. The `fit_transform` method is called in the instantiated object.
    **scaler_kwargs
        Keyword arguments passed to `scaler` during instantiation.
    Returns
    -------
    data : ndarray
        Normalized data.
    ss : object
        Instance of `scaler` that was used to transform `data`.
    """

    ss = scaler(**scaler_kwargs)
    data = ss.fit_transform(data)

    return data, ss


def normalize_x_y(X, y, scaler, **scaler_kwargs):
    """
    Normalize `X` and `y` using a `scaler`.
    Parameters
    ----------
    X : ndarray
        Training data
    y : ndarray
        Target data
    scaler : cls
        Class used to normalize `data`. The `fit_transform` method is called in the instantiated object.
    scaler_kwargs :
        Keyword arguments passed to `scaler` during instantiation.

    Returns
    -------
    X : ndarray
        Normalized training data
    ss_X : Instance of `scaler` used to transform `X`
    y : ndarray
        Normalized target data
    ss_y : Instance of `scaler` used to transform `y`
    """

    X, ss_X = normalize(X, scaler, **scaler_kwargs)
    y, ss_y = normalize(y, scaler, **scaler_kwargs)

    return X, ss_X, y, ss_y
