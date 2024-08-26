import glob
import os
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, "/projects/caps/aaronjo2/shearXlensing/code")
from catalogs import DESY3ShearCat


def make_map(nside, ra, dec, vals=None):
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)
    npix = hp.nside2npix(nside)
    return np.bincount(ipix, weights=vals, minlength=npix)


def main():

    nside = 1024
    overwrite = True

    if len(sys.argv) < 2:
        print("Must specify location of DES Y3 calibrated catalogs")
        return

    cat_dir = sys.argv[1]
    cat_files = glob.glob(f"{cat_dir}/*.pkl")

    # save maps at {data_dir}/../maps
    save_dir = '/'.join(cat_dir.rstrip('/').split('/')[:-1]) + "/maps"
    print("saving maps to:", save_dir, '\n')
    os.makedirs(save_dir, exist_ok=True)

    for cat_file in cat_files:
        cat = DESY3ShearCat.load_from_pkl(cat_file)
        print(cat.name)

        # weights map
        w_map = make_map(nside, cat.data["ra"], cat.data["dec"], vals=cat.data["weight"])
        mask = w_map > 0

        # weighted shear maps
        g_maps = [make_map(nside, cat.data["ra"], cat.data["dec"], vals=cat.data[f"g_{i}"] * cat.data["weight"])
                  for i in [1, 2]]
        for g_map in g_maps:
            g_map[mask] = g_map[mask] / w_map[mask]

        # psf elipticity maps
        psf_maps = [make_map(nside, cat.data["ra"], cat.data["dec"],
                    vals=cat.data[f"psf_e{i}"] * cat.data["weight"]) for i in [1, 2]]
        for psf_map in psf_maps:
            psf_map[mask] = psf_map[mask] / w_map[mask]

        save_name = f"{save_dir}/DESY3_zbin{cat.zbin}_nside{nside}"

        hp.write_map(save_name + "_shearmaps.fits", g_maps, column_names=["g1", "g2"], overwrite=overwrite)
        hp.write_map(save_name + "_psfmaps.fits", psf_maps, column_names=["psf_e1", "psf_e2"], overwrite=overwrite)
        hp.write_map(save_name + "_mask.fits", w_map, overwrite=overwrite)


if __name__ == "__main__":
    main()
