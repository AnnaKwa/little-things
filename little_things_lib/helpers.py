import numpy as np

RADIANS_PER_DEG = np.pi / 180.


def calc_physical_distance_per_pixel(
        distance_to_galaxy,
        deg_per_pix
):
    """
    :param distance_to_galaxy: [kpc]
    :param deg_per_pix: this is typically given in the CDELT field in FITS headers
    :return: distance in i=0 plane corresponding to 1 pixel
    """
    radians_per_pix = deg_per_pix * RADIANS_PER_DEG
    distance_per_pix = distance_to_galaxy * radians_per_pix
    return distance_per_pix
