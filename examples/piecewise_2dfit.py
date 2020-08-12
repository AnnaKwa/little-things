from fit2d import Galaxy, RingModel
from fit2d.mcmc import LinearPrior
from fit2d.mcmc import lnlike_piecewise_model, dynesty_lnlike
from fit2d.models import PiecewiseModel

from astropy.io import fits
import joblib
import dynesty

### SAMPLER PARAMS ###
maxiter=10
maxcall=10
maxiter_init=10
maxcall_init=10
output_results_file = "piecewise_example_results.pkl"

### GALAXY PARAMS ###
name = "UGC3974"
distance = 8000. # [kpc]
observed_2d_vel_field_fits_file = "/home/anna/Desktop/fit2d/data/UGC3974_1mom.fits"
deg_per_pixel=4.17e-4
v_systemic=270. 

### RING PARAMS ###
ring_param_file = "/home/anna/Desktop/fit2d/data/UGC3974_ring_parameters.txt"
# x and y dims are switched in ds9 fits display versus np array shape
fits_ydim, fits_xdim = fits.open(observed_2d_vel_field_fits_file)[0].data.shape

### PIECEWISE MODEL PARAMS ###
num_bins = 10
bounds_min, bounds_max = (0, 100)


### Don't modify anything below here ###

galaxy = Galaxy(
    name=name,
    distance=distance,
    observed_2d_vel_field_fits_file=observed_2d_vel_field_fits_file,
    deg_per_pixel=deg_per_pixel,
    v_systemic=v_systemic, 
)

ring_model = RingModel(
    ring_param_file=ring_param_file,
    fits_xdim=fits_xdim,
    fits_ydim=fits_ydim,
    distance=distance
)

piecewise_model = PiecewiseModel(num_bins=num_bins)
piecewise_model.set_bounds(bounds_min, bounds_max)
piecewise_model.set_bin_edges(rmin=ring_model.radii_kpc[0], rmax=ring_model.radii_kpc[-1])

prior = LinearPrior(bounds=piecewise_model.bounds)
prior_transform = prior.transform_from_unit_cube

lnlike_args = [galaxy, ring_model, piecewise_model]
log_likelihood = dynesty_lnlike(lnlike_piecewise_model, prior.transform_from_unit_cube, lnlike_args)

sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, piecewise_model.num_bins)
sampler.run_nested(
    maxiter=maxiter, 
    maxcall=maxcall,
    maxiter_init=maxiter_init, 
    maxcall_init=maxcall_init,
    print_progress=True, 
    use_stop=False, wt_kwargs={"pfrac": 1.0}
)

with open(output_results_file, "wb") as f:
    joblib.dump(sampler.results, f)