from .constants import GNEWTON
from .helpers import auto_assign
from .nfw_halo_model import get_dens_mass_nfw
import numpy as np


class EmceeParameters:
    @auto_assign
    def __init__(
            self,
            ndim,
            nwalkers,
            nburn,
            niter,
            nthin,
            nthreads
    ):
        pass
"""        if niter % nthin != 0:
            raise ValueError("niter must be divisible by nthin, try another set"
                             "of parameters.")

"""

def convert_to_physical_parameter_space(mcmc_space_parameters):
    theta_mcmc = mcmc_space_parameters
    return 10 ** theta_mcmc[0], 10 ** theta_mcmc[1], theta_mcmc[2]


def convert_to_mcmc_parameter_space(physical_space_parameters):
    theta_physical = physical_space_parameters
    return np.log10(theta_physical[0]), np.log10(theta_physical[1]), theta_physical[2]


def get_mcmc_start_position(
        galaxy,
        number_grid_points_per_param=None
):
    bounds = np.array(
        [galaxy.bounds['rhos'],
         galaxy.bounds['rs'],
         galaxy.bounds['ml'],
         ])

    if not number_grid_points_per_param:
        number_grid_points_per_param = [
            3,  # log10(rhos)
            4,  # log10(rs)
            3  # ML disk
        ]

    boundary_offsets = [0.1 * (bound[1] - bound[0]) for bound in bounds]

    start_grid = [np.linspace(bound[0] + boundary_offset, bound[1] - boundary_offset, num_to_sample)
                  for bound, boundary_offset, num_to_sample
                  in zip(bounds, boundary_offsets, number_grid_points_per_param)]

    # explode to all possible combinations of starting params
    possible_start_combinations_0 = [row.flatten() for row in np.meshgrid(*start_grid)]
    possible_start_combinations = list(zip(*possible_start_combinations_0))
    possible_start_combinations_physical_space = [
        convert_to_physical_parameter_space(parameter_space_point)
        for parameter_space_point in possible_start_combinations
    ]
    # TODO: refactor the lnlike function to take specific parameters and add those as parameters to this function
    lnlike_grid = [
        lnlike(parameter_space_point, galaxy)[0]
        for parameter_space_point in possible_start_combinations_physical_space
    ]
    start_point = possible_start_combinations_physical_space[np.argmax(np.array(lnlike_grid))]
    # random draw to start slightly away (5% of bounds range) from each start point
    start_point_radii = [0.05 * (bound[1]-bound[0]) for bound in bounds]
    start_point_mcmc_space = (np.log10(start_point[0]), np.log10(start_point[1]), start_point[2])
    return start_point_mcmc_space, start_point_radii




def generate_nwalkers_start_points(
        galaxy,
        nwalkers,
        start_point,
        start_point_radii
):
    radii = np.array([[np.random.uniform(low=-start_point_radius, high=start_point_radius) for start_point_radius in start_point_radii]
        for i in range(nwalkers)])
    start_points = np.array(start_point) + radii
    for point in start_points:
        if point[0] < galaxy.bounds['rhos'][0] or point[0] > galaxy.bounds['rhos'][1]:
            print(f"Start point {point[0]} for log10(rhos) is outside prior bounds {galaxy.bounds['rhos']}")
        if point[1] < galaxy.bounds['rs'][0] or point[1] > galaxy.bounds['rs'][1]:
            print(f"Start point {point[1]} for log10(rs) is outside prior bounds {galaxy.bounds['rs']}")
        if point[2] < galaxy.bounds['ml'][0] or point[2] > galaxy.bounds['ml'][1]:
            print(f"Start point {point[2]} for M/L is outside prior bounds {galaxy.bounds['ml']}")
    return start_points



def lnlike(
        params_physical_space,
        galaxy,
):
    rhos, rs, ml_disk = params_physical_space
    #galaxy, sidm_setup, mn, emcee_params, prior_params, reg_params, bounds = args
    # ndim, nwalkers, nburn, niter, nthin, nthreads = unpack_emcee_params(emcee_params)

    v_d = np.sqrt(ml_disk) * galaxy.v_stellar
    v2_baryons = galaxy.v_gas ** 2 + v_d ** 2

    dm_mass_enclosed = get_dens_mass_nfw(galaxy.radii , rhos, rs)
    v2_dm = GNEWTON * dm_mass_enclosed / galaxy.radii

    v_m = np.sqrt(v2_dm + v2_baryons)
    if not np.all([np.isfinite(item) for item in v_m]):
        print('error: something went wrong in lnlike for galaxy ')
        print('rhos, rs, v2_dm =',  rhos, rs, v2_dm)

    # TODO: probably have to interpolate radii and rotation here to make the 2d modeled field
    chisq, model_2d_field = chisq_2d(galaxy, galaxy.radii, v_m, v_err_const=galaxy.v_error_2d, record_array=True)

    return -0.5 * (chisq ), \
           (chisq, np.sqrt(v2_dm), np.sqrt(v2_baryons),
            v_d, v_m, dm_mass_enclosed, model_2d_field)


def lnprior(theta, bounds):
    for item, bound in zip(theta, bounds):
        if not bound[0] <= item <= bound[1]:
            return -np.inf
    return 0.0


def lnprob(theta_mcmc_space, galaxy, save_blob=True):
    bounds = np.array(
        [galaxy.bounds['rhos'],
         galaxy.bounds['rs'],
         galaxy.bounds['ml'],
         ])
    lp = lnprior(theta_mcmc_space, bounds)
    params_physical_space = convert_to_physical_parameter_space(theta_mcmc_space)
    if not np.isfinite(lp):
        return -np.inf ,0
    lnl, bb = lnlike(params_physical_space, galaxy)
    blob = params_physical_space + bb if save_blob is True else None
    return lp + lnl, blob


def chisq_2d(
        galaxy,
        radii_model,
        v_rot_1d_model,
        v_err_2d=None,
        v_err_const=10.,
        record_array=False
):
    """
    :param v_rot_1d_model:
    :param v_los_2d_data:
    :param v_err_2d: if 2d error field not provided, use v_err_const
    :param v_err_const:
    :return:
    """
    try:
        vlos_2d_data = galaxy.vlos_2d_data
    except:
        raise ValueError('Need observed 2d velocity field for galaxy')
    vlos_2d_model = galaxy.create_2d_velocity_field(
        radii_model,
        v_rot=v_rot_1d_model
    )
    if v_err_2d:
        chisq = np.nansum((vlos_2d_data - vlos_2d_model) ** 2 / v_err_2d ** 2)
    else:
        chisq = np.nansum((vlos_2d_data - vlos_2d_model) ** 2 / v_err_const ** 2)
    if record_array:
        return chisq, vlos_2d_model
    else:
        return chisq, None

