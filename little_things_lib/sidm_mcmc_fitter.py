from .constants import GNEWTON
from .sidm_halo_model import (
    NFWMatcher,
    get_dens_mass_sidm,
    get_dens_mass_without_baryon_effect)
from .helpers import auto_assign
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


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


def convert_to_physical_parameter_space(mcmc_space_parameters):
    theta_mcmc = mcmc_space_parameters
    return 10 ** theta_mcmc[0], 10 ** theta_mcmc[1], theta_mcmc[2], theta_mcmc[3]


def convert_to_mcmc_parameter_space(physical_space_parameters):
    theta_physical = physical_space_parameters
    return np.log10(theta_physical[0]), np.log10(theta_physical[1]), theta_physical[2], theta_physical[3]


def get_mcmc_start_position(
        galaxy,
        number_grid_points_per_param=None
):
    bounds = np.array(
        [galaxy.bounds['rho0'],
         galaxy.bounds['sigma0'],
         galaxy.bounds['cross_section'],
         galaxy.bounds['ml'],
         ])

    if not number_grid_points_per_param:
        number_grid_points_per_param = [
            4,  # log10(rho0)
            8,  # log10(sigma0)
            1,  # cross section
            4  # ML disk
        ]

    boundary_offsets = [0.1 * (bound[1] - bound[0]) for bound in bounds]
    # hard code in no offset for cross section
    boundary_offsets[2] = 1e-6

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

    return start_point, start_point_radii


def generate_nwalkers_start_points(
        nwalkers,
        start_point,
        start_point_radii
):
    radii = np.array([[np.random.uniform(low=-start_point_radius, high=start_point_radius) for start_point_radius in start_point_radii]
        for i in range(nwalkers)])
    start_points = np.array(start_point) + radii

    return start_points


def lnlike(
        params_physical_space,
        galaxy,
):
    rho0, sigma0, cross, ml_disk = params_physical_space
    #galaxy, sidm_setup, mn, emcee_params, prior_params, reg_params, bounds = args
    # ndim, nwalkers, nburn, niter, nthin, nthreads = unpack_emcee_params(emcee_params)

    ml_median = galaxy.prior_params['ml_median']
    log10_rmax_spread = galaxy.prior_params['log10_rmax_spread']
    log10_c200_spread = galaxy.prior_params['log10_c200_spread']

    abs_e_V, rel_e_V, vmax_prior, ratio_vmax_prior = galaxy.regularization_params
    v_d = np.sqrt(ml_disk) * galaxy.v_stellar

    v2_baryons = galaxy.v_gas ** 2 + v_d ** 2
    lines = np.array(list(zip(galaxy.radii, v2_baryons * galaxy.radii / GNEWTON)))
    r0 = 3. * sigma0 / np.sqrt(4.* np.pi * GNEWTON * rho0)
    mnorm = rho0 * r0 ** 3

    interp_m = InterpolatedUnivariateSpline(lines[:, 0] / r0, lines[:, 1] / mnorm, k=2)
    rm = lines[0, 0] / r0
    rmm = lines[-1, 0] / r0

    def massB(r):
        if rm < r < rmm:
            m = interp_m(r)
        elif r <= rm:
            m = interp_m(rm) * (r / rm) ** 3
        else:
            m = interp_m(rmm)
        return m

    nfw_matcher = NFWMatcher()
    if cross > 1e-3:
        r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, rho, mass = \
            get_dens_mass_sidm(rho0, sigma0,
                               cross, r0,
                               mnorm, massB,
                               galaxy, nfw_matcher)
    else:
        r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, rho, mass = \
            get_dens_mass_without_baryon_effect(rho0, sigma0, cross, r0, mnorm, galaxy, nfw_matcher)
    v2_dm = []
    for r, m in zip(galaxy.radii, mass):
        if r > r1:
            vd2 = GNEWTON * nfw_matcher.nfw_m_profile(r / rs) * mnfw0 / r
        else:
            vd2 = GNEWTON * m / r
        v2_dm = np.append(v2_dm, vd2)
    v_m = np.sqrt(v2_dm + v2_baryons)
    if not np.all([np.isfinite(item) for item in v_m]):
        print('error_5: something went wrong in lnlike for galaxy ' + galaxy.Galaxy)
        print('rho0, sigma0, r0, r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, v2_dm =', \
              rho0, sigma0, r0, r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, v2_dm)

    # TODO: probably have to interpolate radii and rotation here to make the 2d modeled field
    chisq = chisq_2d(galaxy, galaxy.radii, v_m)

    return -0.5 * (chisq ), \
           (r1, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, chisq, np.sqrt(v2_dm),
            np.sqrt(v2_baryons), v_d, v_m)


def lnprior(theta, bounds):

    for item, bound in zip(theta, bounds):
        if not bound[0] <= item <= bound[1]:
            return -np.inf
    return 0.0


def lnprob(theta_mcmc_space, galaxy):
    bounds = np.array(
        [galaxy.bounds['rho0'],
         galaxy.bounds['sigma0'],
         galaxy.bounds['cross_section'],
         galaxy.bounds['ml'],
         ])
    lp = lnprior(theta_mcmc_space, bounds)
    if not np.isfinite(lp):
        return -np.inf, 0
    params_physical_space = convert_to_physical_parameter_space(theta_mcmc_space)
    lnl, bb = lnlike(params_physical_space, galaxy)
    blob = params_physical_space + bb
    return lp + lnl, blob


def chisq_2d(
        galaxy,
        radii_model,
        v_rot_1d_model,
        v_err_2d=None,
        v_err_const=2.
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
    return chisq

