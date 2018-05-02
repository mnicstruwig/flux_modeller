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