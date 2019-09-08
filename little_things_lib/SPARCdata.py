import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import differential_evolution
from astroML.linear_model import PolynomialRegression 
#from sklearn.gaussian_process import GaussianProcess # old version, no longer supported 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

def read_SPARC_data(data_dir):

    data_info_location = os.path.join(data_dir,'temp.mrt')

    d = [12, 3, 7, 6, 3, 5, 5, 8, 8, 6, 9, 6, 9, 8, 6, 6, 6, 4, 15]
    dt_in = [('Galaxy', 'U99'), ('T', int), ('D', float), ('e_D', float), \
                   ('f_D', int), ('Inc',  float), ('e_Inc', float), ('Lum', float), \
                   ('e_Lum', float), ('Reff', float), ('SBeff',  float), ('Rdisk', float), \
                   ('SBdisk', float), ('MHI',  float), ('RHI',  float), ('Vflat', float), \
                   ('e_Vflat', float), ('Q',  int), ('Ref',  object)]
    dlong = np.genfromtxt(data_info_location, delimiter=d, skip_header=29, \
            dtype=dt_in, autostrip=True).view(np.recarray)

    return dlong[mask_galaxies(dlong)], dt_in



def mask_galaxies(g):
    mask = np.zeros(len(g), dtype=bool)
    for i, entry in enumerate(g):
#        if (entry.Vflat > 0 and entry.Q < 3): mask[i] = True
#        if (entry.Vflat > 0): mask[i] = True
#        if (entry.Inc >= 30 and entry.Vflat <= 0): mask[i] = True
        if (entry.Inc >= 30 and entry.Q < 3): mask[i]=True
    return mask


def create_galaxies_arr(data, start, end, dt_galaxy):
    d = data[start:end]
    galaxies = np.rec.array( np.empty((len(d),), dtype=dt_galaxy) )    
    for na in d.dtype.names:
        galaxies[na] = d[na]
    return galaxies

def unpack_emcee_params(emcee_params):
    ndim, nwalkers, nburn, niter = emcee_params[:4]
    try:
        nthin = emcee_params[4]
    except:
        nthin = 1
    try:
        nthreads = emcee_params[5]
    except:
        nthreads = 1
    return (ndim, nwalkers, nburn, niter, nthin, nthreads)

def dt(emcee_params, dt_in, model='sidm'):
    nfit = 9
    dtData = np.dtype([('Filename','U200'),('R',object),('R_min',float),('R_max',float),\
                       ('V',object),('e_V',object),('Gas',object),('Disk',object),('Bulge',object)])
    dtReg = np.dtype([('Vmax',float),('Rmax',float),('chi2',float),\
                      ('V',float,(nfit,)),('dV',float,(nfit,)),('ddV',float,(nfit,)),('V_data',object),\
                      ('Vdm_med',object),('Vdm_max',object),('Vdm_min',object),\
                      ('V_spline',object),('Vs',float,(nfit,))])
    dtGP = np.dtype([('Vmax',float),('Rmax',float),('e_Vmax',float),\
                     ('V',float,(nfit,)),('dV',float,(nfit,)),('ddV',float,(nfit,)),\
                     ('e_V',float,(nfit,)),\
                     ('R_data',object),('V_data',object),('e_V_data',object),\
                    ('Vdm_med',object),('Vdm_max',object),('Vdm_min',object)])
    dtSB = np.dtype([('Filename','U200'),('R_min',float),('R_max',float),\
                     ('R',object),('Disk',object),('Bulge',object),('SB0',float),('R0',float)])

    ndim, nwalkers, nburn, niter, nthin, nthreads = unpack_emcee_params(emcee_params)
    nsamples = int(nwalkers * niter / nthin)
    
    dt_blobs_sidm = np.dtype({'names':['rho0', 'sigma0', 'cross', 'ml_disk', 'ml_bulge', 'r1', 'mass_r1', 'rho_r1', 'rhos', 'rs', 'vmax', 'rmax', 'mvir', 'rvir', 'cvir', 'slope_15pRvir', 'chisq', 'v_dm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                     'formats':[float, float, float, float, float, float, float, float, \
                                float, float, float, float, float, float, float, float, float, object, object, object, object, object]})
    
    dt_blobs_burkert = np.dtype({'names':['rho0', 'r0', 'ml_disk', 'ml_bulge', 'rho0rc', 'rc',  'vmax', 'rmax', 'chisq', \
                              'v_dm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                     'formats':[float, float, float, float, float, float, float, float, float, \
                                object, object, object, object, object]})

    dt_blobs_nfw = np.dtype({'names':['rhos', 'rs', 'ml_disk', 'ml_bulge', 'vmax', 'rmax', 'c200', 'm200', 'r200', 'chisq', \
                              'v_dm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                     'formats':[float, float, float, float, float, float, float, float, float, float, \
                                object, object, object, object, object]})

    dt_blobs_tnfw = np.dtype({'names':['rhos', 'rs', 'r0', 'ml_disk', 'ml_bulge', 'rhocrc', 'rc',  'vmax', 'rmax', 'vmax_nfw', 'rmax_nfw', 'c200', 'm200', 'r200', 'chisq', \
                              'v_dm', 'v_cdm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                     'formats':[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, \
                                object, object, object, object, object, object]})

    dt_blobs_hnfw = np.dtype({'names':['rhos', 'rs', 'r0', 's', 'ml_disk', 'ml_bulge', 'd15', 'slope15', 'vmax', 'rmax', 'vmax_nfw', 'rmax_nfw', 'c200', 'm200', 'r200', 'cvir', 'mvir', 'rvir', 'chisq', \
                              'v_dm', 'v_cdm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                     'formats':[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, \
                                object, object, object, object, object, object]})

    dt_blobs_hnfw_mod = np.dtype({'names':['m200', 'r0', 's', 'ml_disk', 'ml_bulge', 'd15', 'slope15', 'vmax', 'rmax', 'vmax_nfw', 'rmax_nfw', 'c200', 'rs', 'r200', 'cvir', 'mvir', 'rvir', 'chisq', \
                              'v_dm', 'v_cdm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                     'formats':[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, \
                                object, object, object, object, object, object]})
    
    dt_blobs_mond = np.dtype({'names':['a0', 'ml_disk', 'ml_bulge', 'chisq', \
                                  'v_dm', 'v_sm', 'v_b', 'v_d', 'v_m'], \
                         'formats':[float, float, float, float,\
                                    object, object, object, object, object]})
    if model == 'sidm':
        dt_blobs = dt_blobs_sidm
    elif model == 'burkert':
        dt_blobs = dt_blobs_burkert
    elif model == 'nfw':
        dt_blobs = dt_blobs_nfw
    elif model == 'tnfw':
        dt_blobs = dt_blobs_tnfw
    elif model == 'hnfw':
        dt_blobs = dt_blobs_hnfw
    elif model == 'hnfw_mod':
        dt_blobs = dt_blobs_hnfw_mod
    elif model == 'mond':
        dt_blobs = dt_blobs_mond
    else:
        print("model not implemented")
        exit()

    dtemcee = np.dtype([('samples', float, (nsamples,ndim)),\
                        ('acor',float,(ndim,2)),('acceptance',float,(nwalkers,)),\
                        ('lnprob',float,(nsamples,))]+\
                       [(item[0],item[1],(nsamples,)) for item in dt_blobs.descr])
    
    dt_galaxy = dt_in + \
                    [('Vb_peak',float),('Vb_peak_location',float),\
                     ('ML_maxdisk',float),('ML_maxbulge',float),\
                     ('Angmom',object),('Angmom_int',float),\
                    ('Data',dtData),('Regr',dtReg),('GP',dtGP),\
                     ('SB',dtSB),('Vb2_interp',object),\
                     ('emcee',dtemcee)]

    return dt_galaxy, dt_blobs



def gen_vdata_poly(entry, npoly):
    
    data=entry.Data 
    Rd=entry.Rdisk
    Rp=entry.Vb_peak_location

    z_sample=data.R
    z_fit=np.array([0.5,1,2,0.5*Rd,Rd,2*Rd,0.5*Rp,Rp,2*Rp])
    sample=data.V
    error=np.sqrt(data.e_V ** 2 + 2 ** 2)
    
    V_spline = InterpolatedUnivariateSpline(z_sample, sample)
    V_spline_fit = V_spline(z_fit)

    n_poly = min( npoly, len(sample)-2 )
    clf=PolynomialRegression(n_poly)
    clf.fit(z_sample[:, None], sample, error)
    y_fit = clf.predict(z_fit[:, None])
    sample_fit = clf.predict(z_sample[:, None])

    z_fine = np.logspace(np.log10(data.R[0]),np.log10(data.R[-1]),100)
    y_fine = clf.predict(z_fine[:, None])
    pos=np.argmax(y_fine)
    vmax=y_fine[pos]
    rmax=z_fine[pos]
    
    chi2_dof = (np.sum(((sample_fit - sample) / error) ** 2)
                    / (len(sample) - n_poly - 1))
    
    dy_pred=(clf.predict(z_fit[:,None]*1.05)-
             clf.predict(z_fit[:,None]*0.95))/(z_fit*0.1)

    ddy_pred=(clf.predict(z_fit[:,None]*1.05)+
              clf.predict(z_fit[:,None]*0.95)-2*y_fit)/((z_fit*0.05) ** 2)

    s2 = sample_fit ** 2 - data.Gas ** 2 - entry.ML_maxbulge * data.Bulge ** 2 - min(entry.ML_maxdisk,0.2) * data.Disk ** 2
    s2 = np.array([max(item,0.0) for item in s2])
    Vdm_max = InterpolatedUnivariateSpline(z_sample,np.sqrt(s2))
    s2 = sample_fit ** 2 - data.Gas ** 2 - entry.ML_maxbulge * data.Bulge ** 2 - min(entry.ML_maxdisk,0.4) * data.Disk ** 2
    s2 = np.array([max(item,0.0) for item in s2])
    Vdm_med = InterpolatedUnivariateSpline(z_sample,np.sqrt(s2))
    s2 = sample_fit ** 2 - data.Gas ** 2 - entry.ML_maxbulge * data.Bulge ** 2 - min(entry.ML_maxdisk,0.6) * data.Disk ** 2
    s2 = np.array([max(item,0.0) for item in s2])
    Vdm_min = InterpolatedUnivariateSpline(z_sample,np.sqrt(s2))

    return vmax,rmax,chi2_dof,y_fit,dy_pred,ddy_pred,sample_fit,\
            Vdm_med,Vdm_max,Vdm_min,V_spline,V_spline_fit


def gen_vdata_gp(entry):
    
    data=entry.Data 
    Rd=entry.Rdisk
    Rp=entry.Vb_peak_location
    
    z_sample = data.R
    z_fit = np.array([0.5,1,2,0.5*Rd,Rd,2*Rd,0.5*Rp,Rp,2*Rp])
    z_fine = np.logspace(np.log10(data.R[0]),np.log10(data.R[-1]),100)
    sample = data.V

#    gp = GaussianProcess(regr='quadratic',corr='absolute_exponential', theta0=1e-1,
#                         thetaL=1e-4, thetaU=10,
#                         normalize=True,
#                         nugget=(error/sample) ** 2,
#                         random_start=10)
    v2_err = np.sum(data.e_V **2 + (0.0*data.V[-1]) **2)/len(data.e_V)
    gp_kernel = RBF(length_scale = Rp)*data.V[-1]**2 + WhiteKernel(noise_level = v2_err)
    gp = GaussianProcessRegressor(kernel = gp_kernel)
    gp.fit(z_sample[:, None], sample)
    y_fit, sigma_fit = gp.predict(z_fit[:, None], return_std=True)
#     sigma_fit = 0.98*np.sqrt(MSE)

    sample_fit, sigma_sample = gp.predict(z_sample[:, None], return_std=True)
#     sigma_sample = 0.98*np.sqrt(MSE)
    
    y_fine, sigma_fine = gp.predict(z_fine[:, None], return_std=True)
#     sigma_fine = 0.98*np.sqrt(MSE)
    pos=np.argmax(y_fine)
    vmax=y_fine[pos]
    rmax=z_fine[pos]
    e_vmax = sigma_fine[pos]

    dy_pred=(gp.predict(z_fit[:,None]*1.05, return_std=False)-
             gp.predict(z_fit[:,None]*0.95, return_std=False))/(z_fit*0.1)

    ddy_pred=(gp.predict(z_fit[:,None]*1.05, return_std=False)+
              gp.predict(z_fit[:,None]*0.95, return_std=False)-2*y_fit)/((z_fit*0.05) ** 2)
    
    s2 = sample_fit ** 2 - data.Gas ** 2 - entry.ML_maxbulge * data.Bulge ** 2 - min(entry.ML_maxdisk,0.2) * data.Disk ** 2
    s2 = np.array([max(item,0.0) for item in s2])
    Vdm_max = InterpolatedUnivariateSpline(z_sample,np.sqrt(s2))
    s2 = sample_fit ** 2 - data.Gas ** 2 - entry.ML_maxbulge * data.Bulge ** 2 - min(entry.ML_maxdisk,0.4) * data.Disk ** 2
    s2 = np.array([max(item,0.0) for item in s2])
    Vdm_med = InterpolatedUnivariateSpline(z_sample,np.sqrt(s2))
    s2 = sample_fit ** 2 - data.Gas ** 2 - entry.ML_maxbulge * data.Bulge ** 2 - min(entry.ML_maxdisk,0.6) * data.Disk ** 2
    s2 = np.array([max(item,0.0) for item in s2])
    Vdm_min = InterpolatedUnivariateSpline(z_sample,np.sqrt(s2))

    return vmax,rmax,e_vmax,y_fit,dy_pred,ddy_pred,sigma_fit,z_sample,sample_fit,sigma_sample,Vdm_med,Vdm_max,Vdm_min
    

def get_disk_params(r, sb):
    nll = lambda a: np.sum((sb-(a[0] * np.exp(-r/a[1])))**2)
    bounds=[(sb[0]/2,sb[0]*100),(r[0]*1e-2,r[-1])]
    result = differential_evolution(nll, bounds, tol = 1e0)
    return result.x


def fill_galaxy(data_dir, entry, n_poly):
    
    data_file_location_v = os.path.join(data_dir,'RC')
    data_file_location_sb = os.path.join(data_dir,'Rotmod')

    entry.SB.Filename = os.path.join(data_file_location_sb,entry.Galaxy+'_rotmod.dat')
    lines = np.loadtxt(entry.SB.Filename, comments="#", delimiter=None, unpack=False, skiprows=0)
    entry.SB.R = lines[:,0]
    entry.SB.Disk = InterpolatedUnivariateSpline(entry.SB.R, lines[:,-2])
    entry.SB.Bulge = InterpolatedUnivariateSpline(entry.SB.R, lines[:,-1])
    entry.SB.R_min = entry.SB.R[0]
    entry.SB.R_max = entry.SB.R[-1]
    entry.SB.SB0, entry.SB.R0 = get_disk_params(np.array(lines[:,0]),np.array(lines[:,-2]))
    
    entry.Data.Filename = os.path.join(data_file_location_v,entry.Galaxy+'.dat')
    lines = np.loadtxt(entry.Data.Filename, comments="#", delimiter=None, unpack=False, skiprows=1)
    lines = np.array([item for item in lines if item[1]>0])    
    entry.Data.R = lines[:,0]
    entry.Data.R_min = np.amin(entry.Data.R)
    entry.Data.R_max = np.amax(entry.Data.R)
    entry.Data.V = lines[:,1]
    entry.Data.e_V = lines[:,2]
    entry.Data.Gas = lines[:,3]
    entry.Data.Disk = lines[:,4]
    temp = entry.Data.R **2 * entry.Data.V * np.array( list ( map( entry.SB.Disk, entry.Data.R ) ) ) * (1e6 * 2 * np.pi)
    entry.Angmom = InterpolatedUnivariateSpline(entry.Data.R, temp)
    entry.Angmom_int = entry.Angmom.integral( max(entry.Data.R_min,entry.SB.R_min), min(entry.Data.R_max,entry.SB.R_max) )
    
    if len(lines[0]) > 5: 
        entry.Data.Bulge = lines[:,5]
        entry.ML_maxbulge = np.amin((entry.Data.V ** 2 - entry.Data.Gas ** 2 - 0.5 * entry.Data.Disk ** 2)/(entry.Data.Bulge ** 2))
        if entry.ML_maxbulge < 0: 
            print(entry.Galaxy,": changing Q value to 10 from ",entry.Q)
            entry.Q = 10
        entry.ML_maxbulge = min(entry.ML_maxbulge, 0.7)
    else:
        entry.Data.Bulge = np.zeros(len(lines[:,0]))
        entry.ML_maxbulge = 1.0
    entry.ML_maxdisk = np.amin((entry.Data.V ** 2 - entry.Data.Gas ** 2 - \
            entry.ML_maxbulge * entry.Data.Bulge ** 2)/(entry.Data.Disk ** 2))
    if entry.ML_maxdisk < 0: 
        print(entry.Galaxy,": changing Q value to 11 from ",entry.Q)
        entry.Q = 11
    entry.ML_maxdisk = min(entry.ML_maxdisk, 0.5)

    vb2 = entry.ML_maxdisk * entry.Data.Disk ** 2 + entry.Data.Gas ** 2 + entry.ML_maxbulge * entry.Data.Bulge ** 2
    get_vb2 = InterpolatedUnivariateSpline(entry.Data.R, vb2)
    entry.Vb2_interp = get_vb2
    rr = np.logspace(np.log10(entry.Data.R[0]),np.log10(entry.Data.R[-1]),100)
    vb2 = list(map(get_vb2,rr))
    location_max = np.argmax(vb2)
    entry.Vb_peak = np.sqrt(vb2[location_max])
    entry.Vb_peak_location = rr[location_max]
    entry.Regr = gen_vdata_poly(entry, n_poly)
    entry.GP = gen_vdata_gp(entry)
    return 1
        
