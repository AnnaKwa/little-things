from astropy.io import fits
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Sequence
import warnings

from ._constants import RAD_PER_ARCSEC
from ._helpers import (
    calc_physical_distance_per_pixel,
    _interpolate_baryonic_rotation_curve
)

@dataclass
class Galaxy:
    name: str
    distance: float
    observed_2d_vel_field_fits_file: str,
    deg_per_pixel: float
    v_systemic: float
    gas_radii: Sequence[float] = None
    gas_velocities: Sequence[float] = None
    stellar_radii: Sequence[float] = None
    stellar_velocities: Sequence[float] = None
    age: float=None


    """
    name: galaxy name
    distance: distance in kpc
    vlos_2d_data: np array of 1st moment, from the fits data
    deg_per_pixel: degrees per pixel in data array
    age: in Gyr, optional. Used for SIDM model.
    stellar/gas radii: Radii that define the stellar/gas rotation curve. 
        If None, defaults to zero rotation velocity for this component.
    stellar/gas velocities: Stellar/gas rotation curve. 
        If None, defaults to zero rotation velocity for this component.
    """

    def __post_init__(self):
        self.image_xdim, self.image_ydim = self.vlos_2d_data.shape
        self.kpc_per_pixel = calc_physical_distance_per_pixel(
            self.distance, self.deg_per_pixel)
        for attribute in ["gas_radii", "gas_velocities", "stellar_radii", "stellar_velocities"]:
            if getattr(self, attribute) is None:
                setattr(self, attribute, [0.])
        self.observed_2d_vel_field = fits.open(observed_2d_vel_field_fits_file)[0].data


class RingModel:
    def __init__(
            self, 
            ring_param_file: str, 
            fits_xdim: int,
            fits_ydim: int,
            distance: float):
        self.radii_arcsec, self.bbarolo_fit_rotation_curve, inclinations, \
        position_angles, fits_x_centers, fits_y_centers , self.v_systemics = \
            np.loadtxt(ring_param_file, usecols=(1,2,4,5,-4,-3,-2)).T

        self.radii_kpc = self.radii_arcsec * RAD_PER_ARCSEC * distance
        self.inclinations = inclinations * np.pi/180
        self.position_angles = position_angles * np.pi/180

        _check_center_pixels(fits_x_centers, fits_y_centers, fits_xdim, fits_ydim)
        self.x_centers, self.y_centers = _convert_fits_to_array_coords(
            np.array(fits_x_centers), np.array(fits_y_centers), fits_xdim, fits_ydim)

        # interpolation functions for generating 2D velocity fields
        self.interp_ring_parameters = {
            'inc': interp1d(self.radii_kpc, self.inclinations),
            'pos_ang': interp1d(self.radii_kpc, self.position_angles),
            'x_center': interp1d(self.radii_kpc, self.x_centers),
            'y_center': interp1d(self.radii_kpc, self.y_centers)
        }


def _check_center_pixels(
        x_centers: Sequence[int], 
        y_centers: Sequence[int], 
        image_xdim: int, 
        image_ydim: int):
    for xc, yc in zip(x_centers, y_centers):
        if xc > image_xdim or yc > image_ydim:
            raise ValueError("X, Y center pixel values lie outside the image "
                                "dimensions. Check that they are within this range.")
        if (0.25 > abs(1. * xc / image_xdim) ) \
                or (abs(1. * xc / image_xdim) > 0.75) \
                or (0.25 > abs(1. * yc / image_ydim) ) \
                or (abs(1. * yc / image_ydim) > 0.75):
            warnings.warn(f"x, y  center pixel values {xc},{yc} provided are "
                            f"not close to center of image dimensions "
                            f"{image_xdim},{image_xdim}. "
                            f"Is this intended?")


def _convert_fits_to_array_coords(
    fits_x: np.ndarray,
    fits_y: np.ndarray, 
    image_xdim: int, 
    image_ydim: int) -> Tuple[int, int]:
    # because x/y in ds9 fits viewer are different from x/y (row/col)
    # convention used here for numpy arrays
        array_y = image_ydim - fits_x
        array_x = image_xdim - fits_y
        return (array_x, array_y)