import numpy as np
import healpy as hp


def get_mass_map(shear_field, nside, smooth_fwhm=None):
    """Make a KS mass map from a shear field."""
    mmap = hp.alm2map(shear_field.alm[0], nside, fwhm=smooth_fwhm)
    return mmap


def make_healpix_map(nside, ra=None, dec=None, ipix=None, vals=None):
    npix = hp.nside2npix(nside)
    if ipix is None:
        if ra is None or dec is None:
            raise ValueError("Must provide ra and dec or pre-computed pixel indices")

        ipix = hp.ang2pix(nside, ra, dec, lonlat=True)

    return np.bincount(ipix, weights=vals, minlength=npix)


def numdiff(func, params, h=1e-4, **kwrds):
    """
    Calculate the numerical derivative of a function with signature func(params, **kwrds).

    params: dictionary of parameters that differentiated wrt
    kwrds: extra keyword arguments to the fuction
    h: step size of numerical derivative

    Returns dictionary of parameters and derivatives
    """
    deriv = {}
    for p, val in params.items():
        params_hi = params | {p: val+h}
        params_lo = params | {p: val-h}
        deriv[p] = (func(params_hi, **kwrds) - func(params_lo, **kwrds)) / (2*h)
    return deriv

