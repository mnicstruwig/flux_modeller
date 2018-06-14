"""
This file contains utilities related to dealing with handling dictionaries in general ways.
"""


def update_apply_dict(dict_list, input_keys, output_key, func, **func_kwargs):
    """
    Apply `func` to each dict in `dict_list` with input data at keys in `input_keys`.
    Store the result at `output_key`. Return the modified `dict_list`.

    The input data at `input_keys` will be passed to the `func` in a list, eg.
    func([sample[input_key1], sample[input_key2]]) <-- calling func

    `input_key` and `output_key` can be the same.

    Parameters
    ----------
    dict_list : list
        List of dictionaries containing the data to be modified.
    input_keys : list
        A list of input keys that will be passed to `func` to use for data, in order.
    output_key : str
        String denoting the key where the result of `func` will be stored.
    func : function
        The function that will be applied to each dictionary
    **func_kwargs
       Keyword arguments of `func`.

    Returns
    -------
    list
        List of dicts with the output of `func` stored under the `output_key` key.
    """

    for sample in dict_list:
        input_data = [sample[key] for key in input_keys]
        result = func(input_data, **func_kwargs)
        sample[output_key] = result

    return dict_list


def upsert_dict_list(dict_list, data, key):
    """
    Insert `data` into a list of dictionaries at a particular `key`.

    Parameters
    ----------
    dict_list : list
        List of dictionaries
    data : array-like (n, d)
        Data that must be inserted into each dict in `dict_list`. Each value of `n` will be upserted to correspond to
        each index of `dict_list`. `n` must be equal to len(dict_list).
    key : str
        Key where data must be inserted.
    Returns
    -------
    list
        List of dicts with inserted data.
    """

    for i, x in enumerate(data):
        sample = dict_list[i]  # Take original sample
        sample[key] = data[i:i + 1]  # Insert data

        dict_list[i] = sample  # Reassign modified sample
    return dict_list
