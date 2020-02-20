import numpy as np
from .constants import GNEWTON
from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq
import warnings


class NFWMatcher:

    def __init__(self):
        self.x_iso_min, self.x_iso_max = 1e-4, 1e2
        self._x_iso = np.logspace(np.log10(self.x_iso_min), np.log10(self.x_iso_max), 100)
        self._y_iso_0 = np.array([1.0, (4.0 * np.pi / 3.0) * self.x_iso_min ** 3])
        self.y_iso = odeint(self.fsidm_no_b, self._y_iso_0, self._x_iso)
        self.density_iso_no_b = InterpolatedUnivariateSpline(self._x_iso, self.y_iso[:, 0])

        self.mass_iso_no_b = InterpolatedUnivariateSpline(self._x_iso, self.y_iso[:, 1])
        self._x = np.logspace(-4, 4, num=100)
        self.y = list(map((lambda x: self.nfw_m_profile(x) * (1 + 1 / x) ** 2), self._x))
        self.r1_over_rs = InterpolatedUnivariateSpline(self.y, self._x)

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


def fsidm(y, x, massB): # for scipy.integrate.odeint
    drhodr = - 9.0 * (massB(x) + y[1]) * y[0] / ( 4 * np.pi * x ** 2 )
    dmassdr = 4 * np.pi * y[0] * x ** 2
    return [drhodr, dmassdr]


def get_dens_mass_sidm(rho0, sigma0, cross, r0, mnorm, massB, galaxy, nfw_matcher):
    mn = nfw_matcher

    x0 = 0.1*galaxy.radii[0]/r0
    y0 = np.array([1.0, (4.*np.pi / 3.0) * x0 ** 3])
    x = np.concatenate(([x0],galaxy.radii/r0))
    y = odeint(fsidm, y0, x, args=(massB,))
    rr = galaxy.rate_constant * cross * sigma0 * y[:,0] * rho0

    if rr[-1] > 1.0:
        x1 = np.logspace(np.log10(x[-1]),np.log10(2.0*x[-1]),5)
        y1 = odeint(fsidm, y[-1,:], x1, args=(massB,))
        while True:
            if galaxy.rate_constant * cross * sigma0 * y1[-1,0] * rho0 < 1.0:
                break
            else:
                x1 = np.logspace(np.log10(x1[-1]),np.log10(2.0*x1[-1]),5)
                y1 = odeint(fsidm, y1[-1,:], x1, args=(massB,))
    elif rr[-1] <= 1.0 and rr[0] > 1.0:
        for i in range(1,len(rr)):
            if (rr[i-1]-1.0)*(rr[i]-1.0) < 0:
                x1 = np.logspace(np.log10(x[i-1]),np.log10(1.1*x[i]),10)
                y1 = odeint(fsidm, y[i-1], x1, args=(massB,))
    else:
        print("error_1. Shouldn't be here.")
        input('Press <ENTER> to continue. The code will exit with errors.')
    rho1 = 1.0/(galaxy.rate_constant * cross * sigma0)
    if np.any([y1[i,0]>y1[i-1,0] for i in range(1,len(x1))]):
        r1 = (x1[0]+x1[-1])/2
        m1 = 1e20
        rs = r1
    else:
        density = InterpolatedUnivariateSpline(x1*r0, y1[:,0]*rho0)
        mass = InterpolatedUnivariateSpline(x1*r0, y1[:,1]*mnorm)
        def rate(x):  return density(x) / rho1
        if (rate(x1[0]*r0)-1)*(rate(x1[-1]*r0)-1) > 0:
            print("error_2 (numerical issues?): ",\
                  r0,rho0,sigma0,x1[0]*r0,x1[-1]*r0,x*r0,rr)
        r1 = brentq((lambda x: rate(x)-1), x1[0]*r0, x1[-1]*r0, rtol=1e-4)
        m1 = mass(r1)
        mr1 = max(m1/(4.*np.pi * density(r1) * r1 ** 3), 0.5)
        rs = r1 / mn.r1_over_rs(mr1)
        if mr1 < 0.5: print("error_3: ",\
                            rho0,sigma0,r1,m1,m1/(4.*np.pi * density(r1) * r1 ** 3))
    if m1 < 0:
        print("error_4 (mass(r1) negative): ",\
              r0,r1,m1,x1*r0,y1[:,0]*rho0,y1[:,1]*mnorm)
    mnfw0 = m1 / mn.nfw_m_profile(r1/rs)
    rmax = 2.163 * rs
    vmax = np.sqrt(GNEWTON * mnfw0 * mn.nfw_m_profile(2.163) / rmax )
    rhos = vmax ** 2 * 2.163 / (GNEWTON * 4.* np.pi * 0.467676 * rs ** 2 )

    lgrhos = np.log(rhos)
    if lgrhos > mn._logrhos_vir[-1]:
        cvir = mn._c[-1]*(lgrhos/mn._logrhos_vir[-1])**11
    else:
        if lgrhos < mn._logrhos_vir[0]:
            cvir = mn._c[0]
        else:
            cvir = mn._getcvir(lgrhos)
    mvir = mnfw0 * mn.nfw_m_profile(cvir)
    rvir = rs*cvir
    r2 = 0.015*rvir
    x2 = r2/r0
    if x2 < x0:
        slope_15pRvir = 0
    elif r2 > r1:
        slope_15pRvir = -(rs+3*r2)/(rs+r2)
    else:
        with warnings.catch_warnings(record=True) as w:
            y2, info = odeint(fsidm, y0, np.array([x0,x2]), args=(massB,), full_output = 1, mxstep=5000)
            slope_15pRvir = fsidm(y2[1,:],x2,massB)[0]*(x2/y2[1,0])
            if len(w): print(info,y2,vmax,rmax,np.log10(rho0),r1,lgrhos,mvir,rvir,cvir,x2*r0,r0)

    return r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, np.delete(y[:,0],0)*rho0, np.delete(y[:,1],0)*mnorm


def get_dens_mass_without_baryon_effect(rho0, sigma0, cross, r0, mnorm, galaxy, nfw_matcher):
    rho1 = 1.0/(galaxy.rate_constant * cross * sigma0)
    def rate(x):
        return rho0 * nfw_matcher.density_iso_no_b(x) / rho1
    if (rate(nfw_matcher.x_iso_min)-1.0)*(rate(nfw_matcher.x_iso_max)-1.0) > 0:
        print("Need to increase mn.x_iso_max; rate(mn.x_iso_max) = ", rate(nfw_matcher.x_iso_max))
        print('Setting r1/r0 =  mn.x_iso_max.')
        r1 = nfw_matcher.x_iso_max
    else:
        r1 = brentq((lambda x: rate(x)-1), nfw_matcher.x_iso_min, nfw_matcher.x_iso_max, rtol=1e-4)
    m1 = nfw_matcher.mass_iso_no_b(r1)
    mr1 = max(m1/(4 * np.pi * nfw_matcher.density_iso_no_b(r1) * r1 ** 3), 0.5)
    r1 *= r0
    rs = r1 / nfw_matcher.r1_over_rs(mr1)
    m1 *= mnorm
    mnfw0 = m1 / nfw_matcher.nfw_m_profile(r1/rs)
    rmax = 2.163 * rs
    vmax = np.sqrt(GNEWTON * mnfw0 * nfw_matcher.nfw_m_profile(2.163) / rmax )
    rhos = vmax ** 2 * 2.163 / (GNEWTON * 4 * np.pi * 0.467676 * rs ** 2 )

    lgrhos = np.log(rhos)
    if lgrhos > nfw_matcher._logrhos_vir[-1]:
        cvir = nfw_matcher._c[-1]*(lgrhos/nfw_matcher._logrhos_vir[-1])**11
    else:
        if lgrhos < nfw_matcher._logrhos_vir[0]:
            cvir = nfw_matcher._c[0]
        else:
            cvir = nfw_matcher._getcvir(lgrhos)
    mvir = mnfw0 * nfw_matcher.nfw_m_profile(cvir)
    rvir = rs*cvir
    x2 = 0.015*rvir/r0
    if x2 < nfw_matcher.x_iso_min:
        slope_15pRvir = 0
    elif x2 > nfw_matcher.x_iso_max:
        slope_15pRvir = -3
    else:
        y2 = np.array([nfw_matcher.density_iso_no_b(x2), nfw_matcher.mass_iso_no_b(x2)])
        slope_15pRvir = nfw_matcher.fsidm_no_b(y2,x2)[0]*(x2/y2[0])

    return r1, mnfw0, m1, rho1, rhos, rs, vmax, rmax, mvir, rvir, cvir, slope_15pRvir, \
            [], nfw_matcher.y

