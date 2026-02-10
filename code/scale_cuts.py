import numpy as np
from scipy import stats, interpolate
import pyccl as ccl
import sacc
import nx2pt


def cum_chi2(datavec, cov):
    chi2s = np.zeros(len(datavec) - 1)
    for i in range(len(chi2s)):
        d = datavec[:i+1]
        c = cov[:i+1][:,:i+1]
        chi2s[i] = d @ np.linalg.solve(c, d)
    return chi2s


def trunc_pk(pk, kmax, smooth_lk=5e-2):
    """Exponentially suppress given power spectrum for k > kmax."""
    suppression = lambda k: stats.logistic.sf(np.log(k / kmax) / smooth_lk)
    return ccl.Pk2D.from_function(lambda k, a: pk(k, a) * suppression(k), is_logp=False)


def get_kmax_ell_cut(cosmo, tracer1, tracer2, kmax, bpws=None, thresh=0.95, smooth_lk=5e-2):
    """Calculate max ell for Cl that corresponds to a max k scale."""
    pk = cosmo.parse_pk(p_of_k_a="nonlinear")
    # apply k-cut to power spectrum
    pk_cut = trunc_pk(pk, kmax, smooth_lk=smooth_lk)
    # calculate cls
    ell = np.hstack([np.linspace(2, 102, 50), np.geomspace(104, 1e4, 150)])
    cl = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk)
    cl_cut = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk_cut)
    if bpws is None:
        ratio = cl_cut / cl
        # interpolate ratio to find ell cut
        ell_cut = interpolate.interp1d(ratio, ell)(thresh)
        return ell_cut
    else:
        cl = nx2pt.bin_theory_cl(cl, bpws, ell=ell)
        cl_cut = nx2pt.bin_theory_cl(cl_cut, bpws, ell=ell)
        return np.abs(1 - cl_cut / cl) < 1 - thresh


def get_kmax_ell_cut_chi2(cosmo, tracer1, tracer2, kmax, bpws, cov, thresh=1, smooth_lk=5e-2):
    """ """
    pk = cosmo.parse_pk(p_of_k_a="nonlinear")
    # apply k-cut to power spectrum
    pk_cut = trunc_pk(pk, kmax, smooth_lk=smooth_lk)
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
    pk_cut = trunc_pk(pk, kmax, smooth_lk=smooth_lk)
    # calculate cls
    ell = np.hstack([np.linspace(1, 101, 50), np.geomspace(103, 5e5, 250)])
    cl = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk)
    cl_cut = cosmo.angular_cl(tracer1, tracer2, ell, p_of_k_a=pk_cut)
    # compute xip / xim
    theta = np.geomspace(0.1, 250, 200) / 60  # deg
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


def get_snr(s, theory=None, kcut=None, use_model=False):
    """Get data vector SNR, optionally applying k-space scale cuts."""
    if (use_model or kcut is not None) and theory is None:
        raise ValueError("Need to provide a CCLTheory object to use kcut or use_model")
    s = s.copy()
    # remove all bmodes
    for dtype in s.get_data_types():
        if 'b' in dtype:
            s.remove_selection(data_type=dtype)
    ells = np.geomspace(2, 6100, 250)
    model = []
    if kcut is not None:
        for tr1, tr2 in s.get_tracer_combinations():
            ell_cut = get_kmax_ell_cut(theory.cosmo, theory.tracers[tr1]["ccl_tracer"], theory.tracers[tr2]["ccl_tracer"], kcut)
            s.remove_selection(tracers=(tr1, tr2), ell__gt=ell_cut)
    if use_model:
        for tr1, tr2 in s.get_tracer_combinations():
            inds = s.indices(tracers=(tr1, tr2))
            if len(inds) > 0:
                bpws = s.get_bandpower_windows(inds).weight.T
                model.append(nx2pt.bin_theory_cl(theory.get_cl(tr1, tr2, ells), bpws, ell=ells))
        data = np.hstack(model)
        Nd = 0
    else:
        data = s.mean
        Nd = len(data)
    #print(len(s.mean), "data points")
    snr = np.sqrt(data @ np.linalg.solve(s.covariance.covmat, data) - Nd)
    return snr

# vectorize over kcut
get_snr = np.vectorize(get_snr, excluded=(0, 1), otypes=[float])
