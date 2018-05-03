"""
Helper functions for scoring the various models used to predict the mapping of the device design parameters to the
flux-linkage curve model parameters.
"""


def score_regressors(regressors, scorer, X, y_expected):
    """
    Scores the regressors using `scorer`.

    Parameters
    ----------
    regressors : list
        List of regressors to score. The `predict` method will be called on each regressor for vectors in `X`.
    scorer : function
        The scorer to use in order to compute the score.
    X : ndarray
        The training input data. Will be used as input by the regressors in `regressor` to make predictions.
    y : ndarray
        The actual targets values corresponding to each vector in `X`.

    Returns
    -------
    list
        An array containing the scores for each regressor.
    """
    scores = []
    for i, reg in enumerate(regressors):
        prediction = reg.predict(X)
        score = scorer(y_expected, prediction)
        scores.append(score)

    return scores
