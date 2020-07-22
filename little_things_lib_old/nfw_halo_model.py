import numpy as np
from .constants import GNEWTON
import warnings


def get_dens_mass_nfw(
        radii,
        rho_s,  # msun kpc-3
        r_s # kpc
):
    radii = np.array(radii)
    mass_enclosed = 4. * np.pi * rho_s * r_s**3 * \
                    (np.log((r_s + radii) / r_s) -
                        (radii/ (radii + r_s) ))
    return mass_enclosed
