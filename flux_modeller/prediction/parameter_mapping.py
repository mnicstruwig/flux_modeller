"""
Helper functions for predicting the flux-curve model parameters from the device
design parameters.
"""


def predict_on_sample_dict(trained_models, sample, input_key="x"):
    """
    Predict the targets for input data in dict `sample` at key `input_key` using
    each model in `trained_models`.

    Parameters
    ----------
    trained_models : list
        List of trained models. The `predict` method will be called on each model.
    sample : dict
        Dict containing the input sample data
    input_key: str
        Key in `sample` where the input data is stored.

    Returns
    -------
    ndarray
        Array containing the predicted target for each model

    """
    x = sample[input_key]
    y_pred_arr = []
    for model in trained_models:
        y_pred = model.predict(x)  # Returns an array
        y_pred_arr.append(y_pred[0])  # Extract prediction from array
    return y_pred_arr


def predict_update_on_dict_list(
    trained_models, dict_list, input_key="x", prediction_key="y_pred"
):
    """
    Predict the targets for input data in `dict_list` at key `input_key` using
    each model in `trained_models`.

    Parameters
    ----------
    trained_models : list
        List of trained models. The `predict` method will be called on each model
    dict_list : list
        List of dicts containing the input sample data
    input_key : str
        Key for each dict in `dict_list` where the training data is stored
    prediction_key : str
        Key where the prediction must be stored in each sample in `dict_list`.

    Returns
    -------
    list
        List of dicts with the added prediction at key `prediction_key`.

    """

    for i, sample in enumerate(dict_list):
        sample[prediction_key] = predict_on_sample_dict(
            trained_models, sample, input_key
        )
        dict_list[i] = sample

    return dict_list
