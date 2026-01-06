import argparse
import glob
from os import path
import sys
import warnings
import numpy as np
from scipy import interpolate
import healpy as hp
from astropy.table import Table
import pymaster as nmt
import nx2pt

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
import ccl_interface
from misc import make_healpix_map
from timer import Timer


def load_cl(file, lmax):
    ell, cl = np.loadtxt(file).T
    if ell[-1] < lmax:
        raise ValueError(f"the provided power spectrum does not go to high enough ell")
    if ell[0] > 0:
        ell = np.hstack([[0], ell])
        cl = np.hstack([[0], cl])
    spline = interpolate.CubicSpline(np.log(1+ell), np.log(1+cl), bc_type="not-a-knot", extrapolate=False)
    log_ell = np.log(1 + np.arange(lmax+1))
    cl = np.expm1(spline(log_ell))
    return cl
    

def gaussian_shear_sim(cl, nside, seed=None):
    # generate Gaussian spin-2 fields from input Cl
    if seed is None: seed = -1
    shear_maps = nmt.synfast_spherical(nside, [cl, 0*cl, 0*cl], [2,], seed=seed)
    return shear_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_cl")
    parser.add_argument("output_dir")
    parser.add_argument("--nside", type=int, default=8192, help="nside (default: 8192)")
    parser.add_argument("-n", "--num-realizations", type=int, default=10, help="number of realizations to generate (default: 10)")

    args = parser.parse_args()

    cl_in = load_cl(args.input_cl, 3*args.nside-1)
    
    seed = len(glob.glob(path.join(args.output_dir, "*.fits")))
    for i in range(args.num_realizations):
        with Timer(f"generating maps ({i+1} / {args.num_realizations})"):
            shear_maps = gaussian_shear_sim(cl_in, args.nside, seed=seed+i)
        hp.write_map(path.join(args.output_dir, f"shear_maps_{seed+i:03}.fits"), shear_maps)


if __name__ == "__main__":
    main()
