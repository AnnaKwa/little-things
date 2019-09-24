import os
import emcee

from dm_halo_model import (
    abundance_match_behroozi_2012,
    get_dens_mass_without_baryon_effect)

import numpy as np
import os, emcee, corner, warnings
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import odeint
from scipy.optimize import brentq, minimize
import matplotlib.pyplot as plt
from matplotlib import rcParams
from SPARCdata import unpack_emcee_params
from autocorr import get_autocorr_N

from galaxy import Galaxy


def start_pos(args):
    galaxy, ss, mn, __, __, __, bounds_in = args

    bounds = np.array(bounds_in)
    ng = np.ones(len(bounds_in))

    ng[0] = 4;  # for log10(rate0)
    bounds[0][1] = min(3.5, bounds_in[0][1])

    ng[1] = 8;  # for log10(sigma0)
    if galaxy.Vflat > 0:
        bounds[1][0] = max(np.log10(galaxy.Vflat / 3), bounds_in[1][0])
        bounds[1][1] = min(np.log10(galaxy.Vflat), bounds_in[1][1])
    else:
        bounds[1][0] = bounds_in[1][0]
        bounds[1][1] = bounds_in[1][1]

    ng[2] = 1;  # for cross

    ng[3] = 4;  # for ML_disk
    bounds[3][1] = min(1.0, bounds_in[3][1])
    bounds[3][0] = max(0.1, bounds_in[3][0])

    if sum(galaxy.Data.Bulge) > 0:
        ng[4] = 4;  # for ML_bulge
        bounds[4][0] = max(0.1, bounds_in[4][0])
        bounds[4][1] = min(1.5, bounds_in[4][1])
    else:
        ng[4] = 1;  # for ML_bulge
        bounds[4][0] = bounds[4][1]

    std = [0.1 * (b[1] - b[0]) for b in bounds]
    g1 = [np.linspace(bounds[i][0] + std[i], bounds[i][1] - std[i], ng[i]) \
          for i in range(len(bounds_in))]
    g2 = np.meshgrid(*g1)
    g3 = [g.flatten() for g in g2]
    grid = list(zip(*g3))
    lnlike_grid = [lnlike(ss.unpack(item), args)[0] for item in grid]
    start = grid[np.argmax(lnlike_grid)]
    print(start)
    print(galaxy.Galaxy, ": starting rho0, sigma0, cross, ml_disk, ml_bulge = ", ss.unpack(start), "min chisq/dof",
          np.max(lnlike_grid) * (-2) / (len(galaxy.Data.R) - 3))
    return start, std

class sidm_setup:

    def __init__(self, tilted_ring_model):
        self.GNewton = 4.302113488372941e-06  # G in kpc * (km/s)**2 / Msun
        self.fourpi = 4.0 * np.pi
        self.age = 10.0  # gyrs
        self.gyr = 3.15576e+16
        self.rate_const = 1.5278827817856099e-26 * self.age * self.gyr
        self.tilted_ring_model = tilted_ring_model  # initialize this beforehand using Bbarolo fit params and .set_tilted_ring_parameters

    def unpack(self, theta):
        return 10 ** (theta[0] - theta[1] - theta[2]) / self.rate_const, 10 ** theta[1], \
               10 ** theta[2], theta[3], theta[4]

    def pack(self, rho0, sigma0, cross, ml_disk, ml_bulge):
        return np.log10(self.rate_const * rho0 * sigma0 * cross), np.log10(sigma0), \
               np.log10(cross), ml_disk, ml_bulge

    def unpack_2(self, theta):
        return theta[0] - theta[1] - theta[2] - np.log10(self.rate_const), theta[1], theta[2], theta[3], theta[4]

    def unpack_3(self, theta):
        return 10 ** (theta[0] - theta[1] - theta[2]) / self.rate_const, 10 ** theta[1], \
               10 ** theta[2], 10 ** theta[3], 10 ** theta[4]

    def pack_3(self, rho0, sigma0, cross, ml_disk, ml_bulge):
        return np.log10(self.rate_const * rho0 * sigma0 * cross), np.log10(sigma0), \
               np.log10(cross), np.log10(ml_disk), np.log10(ml_bulge)


class match_nfw:

    def __init__(self):
        self.x_iso_min, self.x_iso_max = 1e-4, 1e2
        self._x_iso = np.logspace(np.log10(self.x_iso_min), np.log10(self.x_iso_max), 100)
        self._y_iso_0 = np.array([1.0, (4.0 * np.pi / 3.0) * self.x_iso_min ** 3])
        self._y_iso = odeint(self.fsidm_no_b, self._y_iso_0, self._x_iso)
        self.density_iso_no_b = InterpolatedUnivariateSpline(self._x_iso, self._y_iso[:, 0])
        self.mass_iso_no_b = InterpolatedUnivariateSpline(self._x_iso, self._y_iso[:, 1])
        self._x = np.logspace(-4, 4, num=100)
        self._y = list(map((lambda x: self.nfw_m_profile(x) * (1 + 1 / x) ** 2), self._x))
        self.r1_over_rs = InterpolatedUnivariateSpline(self._y, self._x)

        self._h0 = 0.671
        self._rhocrit = 277.2 * self._h0 ** 2
        self._c = np.logspace(0, 3, 50)
        self._logrhos = np.log((200. / 3.) * \
                               self._c ** 3 / (np.log(1 + self._c) - self._c / (1 + self._c)) * self._rhocrit)
        self._getc = InterpolatedUnivariateSpline(self._logrhos, self._c)
        self._logrhos_vir = np.log((104.2 / 3.) * \
                                   self._c ** 3 / (np.log(1 + self._c) - self._c / (1 + self._c)) * self._rhocrit)
        self._getcvir = InterpolatedUnivariateSpline(self._logrhos_vir, self._c)

    def fsidm_no_b(self, y, x):  # for scipy.integrate.odeint
        drhodr = - 9.0 * y[1] * y[0] / (4 * np.pi * x ** 2)
        dmassdr = 4 * np.pi * y[0] * x ** 2
        return [drhodr, dmassdr]

    def nfw_m_profile(self, x):
        return np.log(1 + x) - x / (1.0 + x)

    def dutton_c200(self, m200):
        log10c200 = 0.905 - 0.101 * np.log10(m200 / (1e12 / self._h0))
        return 10 ** log10c200



def tophat(x):
    return -2 * np.log(1 / (1 + x ** 40))


def abundance_match_behroozi_2012(Mhalo, z=0, alpha=None):
    """
    do abundance matching from arxiv 1207.6105v1
    alpha can be specified as the faint end slope
    at z = 0, alpha = -1.474

    as of 10/2014, this jives with what's on his website
    """

    if alpha is not None:
        vara = True
    else:
        vara = False

    from numpy import log10, exp
    def f(x, alpha, delta, gamma):
        top = log10(1 + exp(x)) ** gamma
        ibottom = 1 / (1 + exp(10 ** -x)) if x > -2.0 else 0
        return -log10(10 ** (alpha * x) + 1) + delta * top * ibottom

    a = 1. / (1. + z)

    nu = exp(-4 * a ** 2)
    log10epsilon = -1.777 + (-0.006 * (a - 1) - 0.000 * z) * nu - 0.119 * (a - 1)
    epsilon = 10 ** log10epsilon

    log10M1 = 11.514 + (-1.793 * (a - 1) - 0.251 * z) * nu
    M1 = 10 ** log10M1

    if alpha is None:
        alpha = -1.412 + (0.731 * (a - 1)) * nu
    else:
        defalpha = -1.412 + (0.731 * (a - 1)) * nu

    delta = 3.508 + (2.608 * (a - 1) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1) + 0.279 * z) * nu

    if not vara:
        log10Mstar = log10(epsilon * M1) + f(log10(Mhalo / M1), alpha, delta, gamma) - f(0, alpha, delta, gamma)

    else:
        from numpy import array, empty_like
        if type(Mhalo) != type(array([1.0, 2.0, 3.0])):
            if Mhalo >= M1:
                # then I use the default alpha
                log10Mstar = log10(epsilon * M1) + f(log10(Mhalo / M1), defalpha, delta, gamma) - f(0, defalpha, delta,
                                                                                                    gamma)
            else:
                # then I use my alpha
                log10Mstar = log10(epsilon * M1) + f(log10(Mhalo / M1), alpha, delta, gamma) - f(0, alpha, delta, gamma)
        else:
            log10Mstar = empty_like(Mhalo)
            log10Mstar[Mhalo >= M1] = log10(epsilon * M1) + f(log10(Mhalo[Mhalo >= M1] / M1), defalpha, delta,
                                                              gamma) - f(0, defalpha, delta, gamma)
            log10Mstar[Mhalo < M1] = log10(epsilon * M1) + f(log10(Mhalo[Mhalo < M1] / M1), alpha, delta, gamma) - f(0,
                                                                                                                     alpha,
                                                                                                                     delta,
                                                                                                                     gamma)

    return 10 ** log10Mstar


def lnlike(
        params,
        args,
        tilted_ring_model
):
    rho0, sigma0, cross, ml_disk, ml_bulge = params
    galaxy, sidm_setup, mn, emcee_params, prior_params, reg_params, bounds = args
    ndim, nwalkers, nburn, niter, nthin, nthreads = unpack_emcee_params(emcee_params)
    rmax_prior, rmax100, slope, log10rmax_spread, \
    log10c200_spread, tophat_prior, half_width, \
    ml_median, log10ml_spread, bulge_prior, bulge_prior_width = prior_params
    abs_e_V, rel_e_V, vmax_prior, ratio_vmax_prior = reg_params

    v_b = np.sqrt(ml_bulge) * galaxy.Data.Bulge
    v_d = np.sqrt(ml_disk) * galaxy.Data.Disk


    v2_baryons = galaxy.Data.Gas ** 2 + v_d ** 2 + v_b ** 2
    lines = np.array(list(zip(galaxy.Data.R, v2_baryons * galaxy.Data.R / sidm_setup.GNewton)))
    r0 = 3 * sigma0 / np.sqrt(sidm_setup.fourpi * sidm_setup.GNewton * rho0)
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

    r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, rho, mass = get_dens_mass(rho0, sigma0,
                                                                                                          cross, r0,
                                                                                                          mnorm, massB,
                                                                                                          args)
    v2_dm = []
    for r, m in zip(galaxy.Data.R, mass):
        if r > r1:
            vd2 = sidm_setup.GNewton * mn.nfw_m_profile(r / rs) * mnfw0 / r
        else:
            vd2 = sidm_setup.GNewton * m / r
        v2_dm = np.append(v2_dm, vd2)
    v_m = np.sqrt(v2_dm + v2_baryons)
    if not np.all([np.isfinite(item) for item in v_m]):
        print('error_5: something went wrong in lnlike for galaxy ' + galaxy.Galaxy)
        print('rho0, sigma0, r0, r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, v2_dm =', \
              rho0, sigma0, r0, r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, v2_dm)

    '''
    chisq = np.sum( (galaxy.Data.V - v_m) ** 2 / galaxy.Data.e_V ** 2 )
    err_V_sqr = galaxy.Data.e_V ** 2 + abs_e_V**2 + (rel_e_V * galaxy.Data.V) ** 2
    chisq_reg = np.sum( (galaxy.Data.V - v_m) ** 2 / err_V_sqr )
    '''
    chisq = chisq_2d(v_m, tilted_ring_model)

    if rmax_prior:
        chisq_cosmo = (np.log10(rmax100 * (vmax / 100) ** slope / rmax) / log10rmax_spread) ** 2
    else:
        lgrhos = np.log(rhos)
        if lgrhos > mn._logrhos[-1]:
            c200 = mn._c[-1] * (lgrhos / mn._logrhos[-1]) ** 11
        else:
            if lgrhos < mn._logrhos[0]:
                c200 = mn._c[0]
            else:
                c200 = mn._getc(lgrhos)
        m200 = mnfw0 * mn.nfw_m_profile(c200)
        chisq_cosmo = (np.log10(c200 / mn.dutton_c200(m200)) / log10c200_spread) ** 2

    if tophat_prior:
        chisq_cosmo = tophat(chisq_cosmo / half_width ** 2)

    if vmax_prior:
        chisq_cosmo += tophat(np.log(vmax / galaxy.Vflat) / np.log(ratio_vmax_prior))

    ml0 = abundance_match_behroozi_2012(mvir) / (galaxy.Lum * 1e9) if ml_median < 0 else ml_median
    chisq_ml = (np.log10(ml_disk / ml0) / log10ml_spread) ** 2
    if ml_median < 0: chisq_ml = tophat(chisq_ml)

    if bulge_prior:
        chisq_ml += -2 * np.log((np.tanh((ml_bulge - ml_disk) / bulge_prior_width) + 1) / 2)

    return -0.5 * (chisq_reg + chisq_cosmo + chisq_ml), \
           (r1, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, chisq, np.sqrt(v2_dm),
            np.sqrt(v2_baryons), v_b, v_d, v_m)


def lnprior(theta, bounds):
    for item, bound in zip(theta, bounds):
        if not bound[0] <= item <= bound[1]:
            return -np.inf
    return 0.0


def lnprob(theta, args, tilted_ring_model):
    __, sidm_setup, __, __, __, __, bounds = args
    lp = lnprior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf, 0
    params = sidm_setup.unpack(theta)
    lnl, bb = lnlike(params, args, tilted_ring_model)
    blob = params + bb
    return lp + lnl, blob



def chisq_2d(
        tilted_ring_model,
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
    vlos_2d_data = tilted_ring_model.vlos_2d_data
    vlos_2d_model = tilted_ring_model.create_2d_velocity_field(
        tilted_ring_params['radii'],
        v_rot=v_rot_1d_model
    )
    if v_err_2d:
        chisq = np.sum((vlos_2d_data - vlos_2d_model) ** 2 / v_err_2d ** 2)
    else:
        chisq = np.sum((vlos_2d_data - vlos_2d_model) ** 2 / v_err_const ** 2)
    return chisq


def tophat(x):
    return -2 * np.log( 1 / ( 1 + x**40 ) )


def lnlike(params, args):
    rho0, sigma0, cross, ml_disk, ml_bulge = params
    galaxy, sidm_setup, mn, emcee_params, prior_params, reg_params, bounds = args
    ndim, nwalkers, nburn, niter, nthin, nthreads = unpack_emcee_params(emcee_params)
    rmax_prior, rmax100, slope, log10rmax_spread, \
    log10c200_spread, tophat_prior, half_width, \
    ml_median, log10ml_spread, bulge_prior, bulge_prior_width = prior_params
    abs_e_V, rel_e_V, vmax_prior, ratio_vmax_prior = reg_params

    v_b = np.sqrt(ml_bulge) * galaxy.Data.Bulge
    v_d = np.sqrt(ml_disk) * galaxy.Data.Disk
    v2_baryons = galaxy.Data.Gas ** 2 + v_d ** 2 + v_b ** 2
    lines = np.array(list(zip(galaxy.Data.R, v2_baryons * galaxy.Data.R / sidm_setup.GNewton)))
    r0 = 3 * sigma0 / np.sqrt(sidm_setup.fourpi * sidm_setup.GNewton * rho0)
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

    r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, rho, mass = get_dens_mass(rho0, sigma0,
                                                                                                          cross, r0,
                                                                                                          mnorm, massB,
                                                                                                          args)
    v2_dm = []
    for r, m in zip(galaxy.Data.R, mass):
        if r > r1:
            vd2 = sidm_setup.GNewton * mn.nfw_m_profile(r / rs) * mnfw0 / r
        else:
            vd2 = sidm_setup.GNewton * m / r
        v2_dm = np.append(v2_dm, vd2)
    v_m = np.sqrt(v2_dm + v2_baryons)
    if not np.all([np.isfinite(item) for item in v_m]):
        print('error_5: something went wrong in lnlike for galaxy ' + galaxy.Galaxy)
        print('rho0, sigma0, r0, r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, v2_dm =', \
              rho0, sigma0, r0, r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, v2_dm)

    '''
    chisq = np.sum( (galaxy.Data.V - v_m) ** 2 / galaxy.Data.e_V ** 2 )
    err_V_sqr = galaxy.Data.e_V ** 2 + abs_e_V**2 + (rel_e_V * galaxy.Data.V) ** 2
    chisq_reg = np.sum( (galaxy.Data.V - v_m) ** 2 / err_V_sqr )
    '''
    chisq = chisq_2d(v_m, tilted_ring_model)

    if rmax_prior:
        chisq_cosmo = (np.log10(rmax100 * (vmax / 100) ** slope / rmax) / log10rmax_spread) ** 2
    else:
        lgrhos = np.log(rhos)
        if lgrhos > mn._logrhos[-1]:
            c200 = mn._c[-1] * (lgrhos / mn._logrhos[-1]) ** 11
        else:
            if lgrhos < mn._logrhos[0]:
                c200 = mn._c[0]
            else:
                c200 = mn._getc(lgrhos)
        m200 = mnfw0 * mn.nfw_m_profile(c200)
        chisq_cosmo = (np.log10(c200 / mn.dutton_c200(m200)) / log10c200_spread) ** 2

    if tophat_prior:
        chisq_cosmo = tophat(chisq_cosmo / half_width ** 2)

    if vmax_prior:
        chisq_cosmo += tophat(np.log(vmax / galaxy.Vflat) / np.log(ratio_vmax_prior))

    ml0 = abundance_match_behroozi_2012(mvir) / (galaxy.Lum * 1e9) if ml_median < 0 else ml_median
    chisq_ml = (np.log10(ml_disk / ml0) / log10ml_spread) ** 2
    if ml_median < 0: chisq_ml = tophat(chisq_ml)

    if bulge_prior:
        chisq_ml += -2 * np.log((np.tanh((ml_bulge - ml_disk) / bulge_prior_width) + 1) / 2)

    return -0.5 * (chisq_reg + chisq_cosmo + chisq_ml), \
           (r1, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, chisq, np.sqrt(v2_dm),
            np.sqrt(v2_baryons), v_b, v_d, v_m)


def lnprior(theta, bounds):
    for item, bound in zip(theta,bounds):
        if not bound[0] <= item <= bound[1]:
            return -np.inf
    return 0.0


def lnprob(theta, args, tilted_ring_model):
    __, ss, __, __, __, __, bounds = args
    lp = lnprior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf, 0
    params = ss.unpack(theta)
    lnl, bb = lnlike(params, args)
    blob = params +  bb
    return lp + lnl, blob


def process_it(galaxy, sampler, emcee_params, ss, dt_blobs, fig_dir, std):
    ndim, nwalkers, nburn, niter, nthin, nthreads = unpack_emcee_params(emcee_params)
    samples = sampler.flatchain[::nthin]
    np.copyto(galaxy.emcee.samples, samples)
    np.copyto(galaxy.emcee.lnprob, sampler.flatlnprobability[::nthin])
    galaxy.emcee.acor = get_autocorr_N(sampler)
    np.copyto(galaxy.emcee.acceptance, sampler.acceptance_fraction)

    bb = np.array(sampler.blobs, dtype=dt_blobs).T.reshape(-1).view(np.recarray)
    blobs = bb[::nthin]
    del bb
    for name in dt_blobs.names:
        np.copyto(galaxy.emcee[name], blobs[name])

    params = [ss.unpack(sample) for sample in samples]
    p1, p2, p3, p4, p5 = map(lambda v: (v[1], v[2] / v[1] - 1, 1 - v[0] / v[1]), \
                             zip(*np.percentile(params, [16, 50, 84], axis=0)))

    all_labels = ["rho0", "sigma0", "cross", "ml_disk", "ml_bulge"]
    label_names = ["log10(rho0)", "log10(sigma0)", "log10(cross)", "log10(ML disk)", "log10(ML bulge)"]
    labels = [name for (name, d) in zip(all_labels, std) if d > 0]
    labeli = [i for (i, d) in zip(list(range(len(all_labels))), std) if d > 0]
    labeln = [name for (name, d) in zip(label_names, std) if d > 0]
    params = np.log10(np.array([galaxy.emcee[name] for name in labels]).T)
    figure = corner.corner(params, show_titles=True, title_kwargs={"fontsize": 12}, labels=labeln)
    fname = os.path.join(os.getcwd(), fig_dir, "corner_figs", "corner_" + galaxy.Galaxy + ".png")
    plt.savefig(fname)
    plt.close(figure)

    x = galaxy.Data.R
    b = np.zeros((len(galaxy.emcee.samples), len(x)))
    np.copyto(b, [item for item in galaxy.emcee.v_dm])
    v_dm_stats = np.array(list(zip(*np.percentile(b, [16, 50, 84], axis=0))))
    np.copyto(b, [item for item in galaxy.emcee.v_sm])
    v_sm_stats = np.array(list(zip(*np.percentile(b, [16, 50, 84], axis=0))))
    np.copyto(b, [item for item in galaxy.emcee.v_d])
    v_d_stats = np.array(list(zip(*np.percentile(b, [16, 50, 84], axis=0))))
    np.copyto(b, [item for item in galaxy.emcee.v_m])
    v_m_stats = np.array(list(zip(*np.percentile(b, [16, 50, 84], axis=0))))

    fig = plt.figure(figsize=(9, 9))
    rcParams['axes.linewidth'] = 2
    rcParams['font.family'] = 'serif'
    #    rcParams['font.sans-serif'] = ['Tahoma']
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax = fig.add_subplot(111)
    ax.set_title(galaxy.Galaxy, fontsize=20)
    ax.set_xlabel('radius (kpc)', fontsize=20)
    ax.set_ylabel('rotation speed (km/s)', fontsize=20)

    ax.errorbar(x, galaxy.Data.V, yerr=galaxy.Data.e_V, fmt='o', c='black')
    #    ax.scatter(x,v_dm_stats[:,1],c='blue')
    ax.fill_between(x, v_dm_stats[:, 0], v_dm_stats[:, 2], facecolor='blue', alpha=0.99, interpolate=True, label='Dark')
    #    ax.scatter(x,v_sm_stats[:,1],c='green')
    ax.fill_between(x, v_sm_stats[:, 0], v_sm_stats[:, 2], facecolor='green', alpha=0.99, interpolate=True, linewidth=2,
                    label='Baryons')
    #    ax.scatter(x,v_d_stats[:,1],c='grey')
    ax.fill_between(x, v_d_stats[:, 0], v_d_stats[:, 2], facecolor='grey', alpha=0.99, interpolate=True, linewidth=2,
                    label='Disk')
    ax.fill_between(x, v_m_stats[:, 0], v_m_stats[:, 2], facecolor='red', alpha=0.99, interpolate=True, linewidth=2,
                    label='Model')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    handles, plot_labels = ax.get_legend_handles_labels()
    ax.legend(handles, plot_labels, prop={'size': 20}, loc=4)

    fname = os.path.join(os.getcwd(), fig_dir, "vrot_figs", "vrot_" + galaxy.Galaxy + ".png")
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    label_names = ["log10(rho0)", "log10(sigma0)", "log10(cross)", "ML disk", "ML bulge"]
    fig, ax = plt.subplots(len(labels), sharex=True, figsize=(10, 5 * len(labels)))
    steps = np.arange(niter) + 1
    for i, iplot in zip(labeli, range(len(labels))):
        for j in range(nwalkers):
            ax[iplot].plot(steps, sampler.chain[j, :, i])
        ax[iplot].set_ylabel(label_names[i])
        ax[iplot].set_xlabel('step')
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fname = os.path.join(os.getcwd(), fig_dir, "trace_plots", "chains_" + galaxy.Galaxy + ".png")
    plt.savefig(fname)
    plt.close(fig)

    return p1, p2, p3, p4, p5
