import numpy as np
import functools

from ._constants import GNEWTON
RADIANS_PER_DEG = np.pi / 180.

    
def extrapolate_v_outside_last_radius(
        r: float,
        r_last: float,
        v_last: float
) -> float:
    mass_enclosed = v_last**2 * r_last / GNEWTON
    v = np.sqrt(GNEWTON * mass_enclosed / r)
    return v


def calc_physical_distance_per_pixel(
        distance_to_galaxy: float,
        deg_per_pix: float
) -> float:
    """
    :param distance_to_galaxy: [kpc]
    :param deg_per_pix: this is typically given in the CDELT field in FITS headers
    :return: distance in i=0 plane corresponding to 1 pixel
    """
    radians_per_pix = deg_per_pix * RADIANS_PER_DEG
    distance_per_pix = distance_to_galaxy * radians_per_pix
    return distance_per_pix

