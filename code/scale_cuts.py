import numpy as np
from scipy import stats, interpolate
import pyccl as ccl
import nx2pt


def cum_chi2(datavec, cov):
    chi2s = np.zeros(len(datavec) - 1)
    for i in range(len(chi2s)):
        d = datavec[:i+1]
        c = cov[:i+1][:,:i+1]
        chi2s[i] = d @ np.linalg.solve(c, d)
    return chi2s


def get_kmax_ell_cut(cosmo, tracer1, tracer2, kmax, thresh=0.95, smooth_lk=5e-2):
    """Calculate max ell for Cl that corresponds to a max k scale."""
    pk = cosmo.parse_pk(p_of_k_a="nonlinear")
    # apply k-cut to power spectrum
    pk_cut = ccl.Pk2D.from_function(lambda k, a: pk(k, a) * stats.logistic.sf(np.log(k / kmax) / smooth_lk), is_logp=False)
    # calculate cls
    ell = np.hstack([np.linspace(2, 102, 50), np.geomspace(104, 1e4, 150)])
    cl = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk)
    cl_cut = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk_cut)
    ratio = cl_cut / cl
    #print(np.min(ratio), np.max(ratio))
    # interpolate ratio to find ell cut
    ell_cut = interpolate.interp1d(ratio, ell)(thresh)
    return ell_cut


def get_kmax_ell_cut_chi2(cosmo, tracer1, tracer2, kmax, bpws, cov, thresh=1, smooth_lk=5e-2):
    """ """
    pk = cosmo.parse_pk(p_of_k_a="nonlinear")
    # apply k-cut to power spectrum
    pk_cut = ccl.Pk2D.from_function(lambda k, a: pk(k, a) * stats.logistic.sf(np.log(k / kmax) / smooth_lk), is_logp=False)
    # calculate cls and make data vectors
    ell = np.hstack([np.linspace(2, 102, 50), np.geomspace(104, 1e4, 150)])
    cl = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk)
    cl = nx2pt.bin_theory_cl(cl, bpws, ell=ell)
    cl_cut = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk_cut)
    cl_cut = nx2pt.bin_theory_cl(cl_cut, bpws, ell=ell)
    # get cut based on chi2 distance
    chi2 = cum_chi2(cl - cl_cut, cov)
    ind_cut = np.nonzero(chi2 > thresh)[0]
    if len(ind_cut) == 0:
        return -1
    return ind_cut[0]


def get_kmax_theta_cut(cosmo, tracer1, tracer2, kmax, thresh=0.95, smooth_lk=5e-2):
    """Calculate min thetas for xi+, xi- that correspond to a max k scale."""
    pk = cosmo.parse_pk(p_of_k_a="nonlinear")
    # apply k-cut to power spectrum
    pk_cut = ccl.Pk2D.from_function(lambda k, a: pk(k, a) * stats.logistic.sf(np.log(k / kmax) / smooth_lk), is_logp=False)
    # calculate cls
    ell = np.hstack([np.linspace(1, 101, 50), np.geomspace(103, 5e5, 250)])
    cl = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk)
    cl_cut = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk_cut)
    # compute xip / xim
    theta = np.geomspace(0.1, 150, 150) / 60  # deg
    ratio_p = cosmo.correlation(ell=ell, C_ell=cl_cut, theta=theta, type="GG+") / cosmo.correlation(ell=ell, C_ell=cl, theta=theta, type="GG+")
    ratio_m = cosmo.correlation(ell=ell, C_ell=cl_cut, theta=theta, type="GG-") / cosmo.correlation(ell=ell, C_ell=cl, theta=theta, type="GG-")
    #print(np.min(ratio_p), np.max(ratio_p))
    #print(np.min(ratio_m), np.max(ratio_m))
    # interpolate ratio to find theta cut
    thresh_p = 2 - thresh if np.max(ratio_p) > 2 - thresh else thresh
    thresh_m = 2 - thresh if np.max(ratio_m) > 2 - thresh else thresh
    theta_cut_p = interpolate.interp1d(ratio_p, theta)(thresh_p) * 60  # arcmin
    theta_cut_m = interpolate.interp1d(ratio_m, theta)(thresh_m) * 60
    return (theta_cut_p, theta_cut_m)


def get_ell_cut_nl(cosmo, tracer1, tracer2, thresh=0.95):
    ell = np.hstack([np.linspace(2, 102, 50), np.geomspace(104, 1e4, 150)])
    pk = cosmo.parse_pk(p_of_k_a="nonlinear")
    pk_lin = cosmo.parse_pk(p_of_k_a="linear")
    cl = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk)
    cl_lin = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk_lin)
    ratio = cl_lin / cl
    # interpolate ratio to find ell cut
    ell_cut = interpolate.interp1d(ratio, ell)(thresh)
    return ell_cut
