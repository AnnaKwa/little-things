### for dynesty and nestle
prior_transform = lambda cube: sidm_cluster_fit.prior_cube(cube, bounds)
log_likelihood = lambda cube: sidm_cluster_fit.lnlike(ss.unpack(cube), args)[0]

if not dynamic:
    sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim, bound=method, nlive=nlive)
    sampler.run_nested(dlogz=tol, print_progress=True)
else:
    sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim)
    sampler.run_nested(maxiter=10000, print_progress=True, use_stop=False, wt_kwargs={'pfrac': 1.0})
    
res = sampler.results # results from sampler; may want to save it
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1]) # get nested sampling samples and weights
new_samples = dynesty.utils.resample_equal(samples, weights) # resample to get a smaller number of equally weighted samples
samples = new_samples[np.random.choice(len(new_samples), nsamples)] # now we have "nsamples" equally-weighted "samples"


"""
Nested samplers need parameters with bounds from 0 to 1. Hence they are called cubes (more appropriately hypercubes since they nparam-dimensional cubes).
The function prior_cube takes those cube parameters and converts them to parameters with the bounds we have set (given by bounds).
Finally ss.unpack will take those parameters and convert them to the parameters given to lnlike().
This is useful if for example we want to sample in p1=log(rhos*rs), p2=log(rs), p3=M/L_disk, p4=M/L_bulge but lnlike() needs (rhos, rs, M/L_disk, M/L_bulge). 
So, ss.unpack() will take (p1,p2,p3,p4) and convert it to (rhos, rs, M/L_disk, M/L_bulge) by calculating rhos=10*(p1-p2),rs=10**rs,M/L_disk=p3,M/L_bulge=p4.
"""


### prior_cube function
def prior_cube(cube, bounds, return_cube=True):
    """
    Transforming unit cube to have the right bounds.
    """
    for i,b in enumerate(bounds):
        cube[i] = cube[i]*(b[1]-b[0])+b[0]
    return cube if return_cube else None