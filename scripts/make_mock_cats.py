import argparse
import glob
import os
from os import path
import sys
import numpy as np
import healpy as hp
from astropy.table import Table

sys.path.insert(0, "/projects/ncsa/caps/aaronjo2/shearXlensing/code")
from agora_rotations import get_rotator
from timer import Timer


def rotate_e1e2_rand(e1, e2, seed=None):
    np.random.seed(seed)
    phi = np.random.uniform(0, 2*np.pi, len(e1))
    rot = np.exp(2j * phi) * (e1 + e2*1j)
    return rot.real, rot.imag


def gen_shear_catalog(signal_maps, catalog, seed=None, noise_level=1):
    nside = hp.npix2nside(len(signal_maps[0]))
    cat_inds = hp.ang2pix(nside, catalog["ra"], catalog["dec"], lonlat=True)
    # generate noise
    g1_n, g2_n = rotate_e1e2_rand(catalog["g_1"], catalog["g_2"], seed=seed)
    # signal + noise
    # here we just sample the value of the pixel, should be ok if nside is high enough
    g1 = signal_maps[0][cat_inds] + noise_level * g1_n
    g2 = signal_maps[1][cat_inds] + noise_level * g2_n
    # new catalog
    sim_cat = Table(dict(ra=catalog["ra"], dec=catalog["dec"],
                         weight=catalog["weight"], g_1=g1, g_2=g2))
    return sim_cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir")
    parser.add_argument("cat_dir")
    parser.add_argument("-o", "--output")
    parser.add_argument("--no-ia", action="store_true")
    parser.add_argument("--noise_level", default=1, type=float, help="noise level relative to that of provided catalogs")
    args = parser.parse_args()

    shear_pattern = "shear/fullsky/maps/raytrace_kg1g2_*_fullsky.fits"
    ia_pattern = "shear/fullsky/maps/raytrace_IAkg1g2_*_fullsky*.fits"
    cat_pattern = "DESY3_shearcat_zbin*.fits"

    # find files
    shear_files = sorted(glob.glob(path.join(args.sim_dir, shear_pattern)))
    if not args.no_ia:
        # add in IA
        ia_files = sorted(glob.glob(path.join(args.sim_dir, ia_pattern)))
        if len(ia_files) != len(shear_files):
            raise RuntimeError("there is not a IA map for each shear map")

    # load data catalog
    cat_files = sorted(glob.glob(path.join(args.cat_dir, cat_pattern)))
    if len(cat_files) != len(shear_files):
        raise RuntimeError("there is not a catalog file for each shear map")

    print(f"found {len(shear_files)} z-bins")
    print()

    # loop over all z-bins
    for zbin in range(len(shear_files)):
        print(f"z-bin {zbin+1}")
        print(f"shear map: {shear_files[zbin]}")
        if not args.no_ia:
            print(f"IA map: {ia_files[zbin]}")
        print(f"catalog: {cat_files[zbin]}")
        print("---")

        shear_maps = hp.read_map(shear_files[zbin], field=[1,2])
        if not args.no_ia:
            ia_maps = hp.read_map(ia_files[zbin], field=[1,2])
            shear_maps += ia_maps

        catalog = Table.read(cat_files[zbin])

        for i in range(10):
            print(f"Rotation {i+1}:")
            
            save_name = path.join(args.output, f"desy3_mockcat_zbin{zbin+1}_rot{i+1:02}.fits")
            if os.path.exists(save_name):
                print(f"{save_name} exists, skipping")
                continue

            rot = get_rotator(i)
            # rotate signal maps
            with Timer("rotating signal map"):
                rot_signal = rot.rotate_map_pixel(shear_maps)

            with Timer("generating catalog"):
                sim_cat = gen_shear_catalog(rot_signal, catalog, noise_level=args.noise_level)

            # save catalog
            os.makedirs(path.dirname(save_name), exist_ok=True)
            print("saving to", save_name)
            sim_cat.write(save_name)
