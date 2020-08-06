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
    #print('bin_edges: ',bin_edges)
    #print('r_arr: ',r_arr)
    for ring in r_arr:
        for radius in range(len(bin_edges)):
            if ring<=bin_edges[radius] and ring>bin_edges[radius-1]: #ring is greater than current bin edge, and less than
                vels.append(velocity_at_bin_center[radius-1])       #previous bin edge
    #print("vels: ",np.array(vels))
    return np.array(vels)


def generate_nwalkers_start_points(
        nwalkers,
        galaxy
):
    start_points=[]
    bin_bounds=sorted(list(galaxy.bounds.values()))
    for walker in range(nwalkers):
        lis=[]
        for radial_bin in range(len(galaxy.bin_edges)-1):
            lis.append(np.random.random()*(bin_bounds[radial_bin][1]-bin_bounds[radial_bin][0])+bin_bounds[radial_bin][0])
        start_points.append(lis)
        #start point shifted by bounds. numpy output [0,1) --> multiplied by (max-min) then shifted by min.
        #each parameter can have individual bounds defined. No need to make all have same bounds.
    return np.array(start_points)


def lnlike(
        params,
        galaxy,
):
    v_m=piecewise_constant(params,galaxy)
    #print('v_m: ',v_m)
    #print('v_m len: ',len(v_m))
    #print('radii len: ',len(galaxy.radii))
    assert v_m.shape[0]==(galaxy.radii.shape[0])

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


def lnprob(params, galaxy, save_blob=True):
    sorted_keys = sorted(list(galaxy.bounds.keys()))
    bounds = np.array(
        [galaxy.bounds[key] for key in sorted_keys])
    lp = lnprior(params, bounds)
    if not np.isfinite(lp):
        return -np.inf ,0
    lnl, bb = lnlike(params, galaxy)
    blob = tuple(params) + bb if save_blob is True else None
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

