import numpy as np
import functools
from scipy.interpolate import interp1d
from typing import Sequence, Tuple

from ._constants import GNEWTON
RADIANS_PER_DEG = np.pi / 180.


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


def _extrapolate_v_outside_last_radius(
        r: float,
        r_last: float,
        v_last: float
) -> float:
    # assume no mass outside last radii
    mass_enclosed = v_last**2 * r_last / GNEWTON
    v = np.sqrt(GNEWTON * mass_enclosed / r)
    return v


def _extrapolate_v_within_first_radius(
        r: float,
        r_first: float,
        v_first: float
) -> float:
    if r_first == 0 or v_first == 0:
        return 0.
    else:
        # assumes constant density within innermost radii
        return v_first * r / r_first


def _interpolate_baryonic_rotation_curve(
        final_radii: Sequence[float],
        rotation_curve_radii: Sequence[float],
        rotation_curve_velocities: Sequence[float]
):
    """Interpolate the baryonic rotation curve data to the set of 
    radii used in the MCMC fit. 

    Args:
        final_radii: radii to interpolate to [kpc]
        rotation_curve_radii: baryonic rotation curve radii [kpc]
        rotation_curve_velocities: baryonic rotation curve [km/s]
    """
    rotation_curve_radii, rotation_curve_velocities = np.array([
        (r, v) for r, v in zip(rotation_curve_radii, rotation_curve_velocities)
            if (np.min(final_radii) <= r <= np.max(final_radii))
    ]).T
    interp_rotation = interp1d(rotation_curve_radii, rotation_curve_velocities)
    interp_radii = [
        r for r in final_radii
        if (np.min(rotation_curve_radii) <= r <= np.max(rotation_curve_radii))
    ]
    v_interp = list(interp_rotation(interp_radii))
    extrap_inside_radii = [r for r in final_radii if r < np.min(rotation_curve_radii)]
    extrap_outside_radii = [r for r in final_radii if r > np.max(rotation_curve_radii)]
    v_extrap_inside = [
        _extrapolate_v_within_first_radius(r, rotation_curve_radii[0], rotation_curve_velocities[0])
        for r in extrap_inside_radii]
    r_last, v_last = rotation_curve_radii[-1], rotation_curve_velocities[-1]
    v_extrap_outside = [_extrapolate_v_outside_last_radius(r, r_last, v_last) for r in extrap_outside_radii]
    return np.array(v_extrap_inside + v_interp + v_extrap_outside)
