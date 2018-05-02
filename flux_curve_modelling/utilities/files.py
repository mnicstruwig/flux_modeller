"""
Helper functions for dealing with data-warehousing tasks

Naming convention:
`{name}-{var1}[start:step:end]-{var2}[start:step:end]-{datetime}.{extension}`
"""
import datetime
import time


def _parse_value_to_string(tup):
    """
    Parses a tuple into strings in accordance with data-warehousing naming convention
    Parameters
    ----------
    tup : tuple
        Tuple containing of len 1, 2 or 3.

    Returns
    -------
    str
        The formatted string. Values are enclosed in square brackets and separated with dashes (if more than 1).
    """
    output_string = ''

    # TODO: Thing about how to do this parsing again
    if isinstance(tup, tuple):
        if len(tup) == 3:
            output_string = output_string + '[{}'.format(tup[0])
            output_string = output_string + '-{}'.format(tup[1])
            output_string = output_string + '-{}]'.format(tup[2])
        elif len(tup) == 2:
            output_string = output_string + '[{}'.format(tup[0])
            output_string = output_string + '-{}]'.format(tup[1])

        return output_string
    return '[{}]'.format(tup)


def generate_file_name(name, extension, **var_kwargs):
    """
    Generates a filename based on the file-naming convention:
    `{name}-{var1}[start:step:end]-{var2}[start:step:end]-{datetime}.{extension}`

    Parameters
    ----------
    name : str
        A description of the file
    extension : str
        The extension to use for the file
    **var_kwargs
        Keyword arguments with tuple values where the keyword will be used to label the variable.
        eg. variable_name = (1,5,20), which will append `variable_name[1-5-20]` to the filename.

    Returns
    -------
    The formatted filename
    """
    filename = '{name}'.format(name=name)

    if var_kwargs is not None:
        for key, value in var_kwargs.items():
            filename = filename + '-{key}'.format(key=key) + '{value}'.format(value=_parse_value_to_string(value))

    timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d[%H.%M.%S]')
    filename = filename + '-{datetime}'.format(datetime=dt) + '.{ext}'.format(ext=extension)

    return filename
