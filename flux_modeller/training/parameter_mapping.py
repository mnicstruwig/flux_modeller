"""
Helper functions for modelling the mapping from the device design parameters to the flux-linkage curve model parameters.
"""
import numpy as np
from sklearn.model_selection import cross_val_score


def train_regressor_per_output(X, y, regressor, **regressor_kwargs):
    """
    Trains a regressor per output variable in target 'y' using input 'X'.

    Parameters
    ----------
    X : ndarray
        Input training data
    y : ndarray
        Corresponding targets of `X`.
    regressor : cls
        Class of the regressor to be used to train the model. The `fit` function will be called on the instantiated
        object.
    **regressor_kwargs
        Keyword arguments passed to `regressor`.

    Returns
    -------
    list
        List of the trained models, with each element hold the corresponding model to each element of the target `y`.
    """
    vector_length = np.shape(y)[1]

    model_list = []
    for vector_index in range(0, vector_length):
        targets = y[:, vector_index]
        reg = regressor(**regressor_kwargs)
        reg.fit(X, targets)
        model_list.append(reg)

    return model_list


def cross_validation_regressor_per_output(X, y, regressor, cv, give_mean_score=True, **regressor_kwargs):
    """
    Performs cross-validation, training a single regressor per element in the target 'y', using 'X' as input.

    Parameters
    ----------
    X : ndarray
        The training input data
    y : ndarray
        The target data
    regressor : cls
        Class of the regressor to be used to train the model. The `fit` function will be called on the instantiated
        object
    cv : int, cross-validation generator or an iterable.
    give_mean_score : {True, False}
        Whether to output the mean score for all regressors, or an array containing each regressor's score.
    **regressor_kwargs
        Keyword arguments passed to `regressor`.

    Returns
    -------
    float or list
        Float of the mean scores if `give_mean_score` is True, otherwise a list of each regressor's score.
    """
    number_regressors = np.shape(y)[1]  # Calculate the number of required regressors

    score_list = []
    for i in range(number_regressors):
        reg = regressor(**regressor_kwargs)
        targets = y[:, i]
        scores = cross_val_score(reg, X, targets, cv=cv)
        score_list.append(scores)

    if give_mean_score:
        return [score.mean() for score in score_list]

    return score_list
