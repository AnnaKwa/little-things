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

def piecewise_constant(
        params, #velocity_at_bin_center = params  
        galaxy
):
    vels=[]
    velocity_at_bin_center=params
    r_arr=galaxy.radii
    bin_edges=galaxy.bin_edges
    for ring in r_arr:
        for radius in range(len(bin_edges)):
            if radius!=0: 
                if ring<bin_edges[radius] and ring>bin_edges[radius-1]: #ring is greater than current bin edge, and less than previous bin edge
                    vels.append(velocity_at_bin_center[radius-1])
    return vels


def generate_nwalkers_start_points(
        nwalkers,
        galaxy
):
    start_points=[]
    bin_bounds=list(galaxy.bounds.values())
    for walker in range(len(galaxy.bin_edges)-1):
        lis=[]
        for iteration in range(nwalkers):
            lis.append(np.random.random()*(bin_bounds[iteration][1]-bin_bounds[iteration][0])+bin_bounds[iteration][0])
        start_points.append(lis)
        #start point shifted by bounds. numpy output [0,1) --> multiplied by (max-min) then shifted by min.
        #each parameter can have individual bounds defined. No need to make all have same bounds.
    return start_points


def lnlike(
        params,
        galaxy,
):
    v_m=piecewise_constant(params,galaxy)
    assert v_m.shape==galaxy.bin_edges.shape
    
    if not np.all([np.isfinite(item) for item in v_m]):
        print('error: something went wrong in lnlike for galaxy ')
        print(f'piecewise velocities = {params}')

    # TODO: probably have to interpolate radii and rotation here to make the 2d modeled field
    chisq, model_2d_field = chisq_2d(galaxy, galaxy.radii, v_m, v_err_const=galaxy.v_error_2d, record_array=True)

    return -0.5 * (chisq ), \
           (chisq, v_m, model_2d_field) 


def lnprior(theta, bounds):
    for item, bound in zip(theta, bounds):
        if not bound[0] <= item <= bound[1]:
            return -np.inf
    return 0.0


def lnprob(params, galaxy):
    sorted_keys = sorted(list(galaxy.bounds.keys()))
    bounds = np.array(
        [galaxy.bounds[key] for key in sorted_keys])
    lp = lnprior(params, bounds)
    if not np.isfinite(lp):
        return -np.inf ,0
    lnl, bb = lnlike(params, galaxy)
    blob = tuple(params) + bb
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

