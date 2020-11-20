from typing import Dict, Any, Union

def scrape_parameters(column_name: str) -> Dict[str, float]:
    """
    Scrapes parameters from the csv column name generated by Maxwell and read by Pandas

    Parameters
    ----------
    column_name: Str
        The name of the column

    Returns
    -------
    dict
        The extracted parameters as a dictionary.
    """
    winding_num_r_str = column_name.split()[3]
    winding_num_z_str = column_name.split()[4]

    winding_num_r = int(winding_num_r_str.split('\'')[1])
    winding_num_z = int(winding_num_z_str.split('\'')[1])

    return {
        'winding_num_r': winding_num_r,
        'winding_num_z': winding_num_z
    }


def get_parameters_dict(column_name: str, winding_diameter: float) -> Dict[str, float]:
    """
    Create a dictionary containing parameters by scraping the column name output by ANSYS Maxwell, and augmenting with
    `winding_diameter`.

    Parameters
    ----------
    column_name: The name of the column to be scraped
    winding_diameter: The diameter of the winding to be added to the dictionary

    Returns
    -------
    dict
        Dictionary containing the extracted parameters as a dictionary.
    """
    dict_ = scrape_parameters(column_name)

    dict_['winding_diameter'] = winding_diameter
    return dict_
