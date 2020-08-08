import numpy as np
import functools
from copy import copy
from inspect import signature, Parameter
from scipy.ndimage.filters import gaussian_filter

from .constants import GNEWTON
RADIANS_PER_DEG = np.pi / 180.


def create_blurred_mask(galaxy_array):
    # creates a mask for pixels that don't have neighbors within 3 pixels
    # useful for only allowing KNN to fill missing values that have nonzero neighbors
    arr = copy(galaxy_array)
    np.nan_to_num(arr, 0)
    mask = gaussian_filter(np.nan_to_num(arr, 0), sigma=6, order=0)
    mask[mask == 0] = False
    mask[mask > 0] = 1
    return mask


def extrapolate_v_outside_last_radius(
        r,
        r_last,
        v_last
):
    M_enclosed = v_last**2 * r_last / GNEWTON
    v = np.sqrt(GNEWTON * M_enclosed / r)
    return v

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


def auto_assign(func):
    """
    to make initializing classes quicker
    :param func:
    :return:
    """
    # Signature:
    sig = signature(func)
    for name, param in sig.parameters.items():
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise RuntimeError('Unable to auto assign if *args or **kwargs in signature.')
    # Wrapper:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        for i, (name, param) in enumerate(sig.parameters.items()):
            # Skip 'self' param:
            if i == 0: continue
            # Search value in args, kwargs or defaults:
            if i - 1 < len(args):
                val = args[i - 1]
            elif name in kwargs:
                val = kwargs[name]
            else:
                val = param.default
            setattr(self, name, val)
        func(self, *args, **kwargs)
    return wrapper

