import argparse
import glob
import os
from os import path
import sys

import numpy as np
import healpy as hp
from astropy.table import Table

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
from timer import Timer
from misc import make_healpix_map


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("cat_dir")
    parser.add_argument("--nside", default=2048, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("--extra-mask")
    parser.add_argument("--uniform-weights", action="store_true")
    parser.add_argument("--cut-wrt-nmean", default=None, type=float)
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument("--do-psf", action="store_true")
    args = parser.parse_args()

    nside = args.nside
    overwrite = args.overwrite
    cat_dir = args.cat_dir

    cat_files = sorted(glob.glob(path.join(cat_dir, args.pattern)))
    print(f"found {len(cat_files)} catalogs")

    if args.output is None:
        # save maps at {data_dir}/../maps
        save_dir = path.join(cat_dir, "../maps")
    else:
        save_dir = args.output
    print("saving maps to:", save_dir, '\n')
    os.makedirs(save_dir, exist_ok=True)

    if args.extra_mask is not None:
        print("will apply extra mask:", args.extra_mask)
        extra_mask = hp.read_map(args.extra_mask)
        assert hp.npix2nside(len(extra_mask)) == args.nside
    else:
        extra_mask = None

    shot_noise_vals = []
    for cat_file in cat_files:
        print("Catalog:", cat_file)
        name_parts = path.basename(path.splitext(cat_file)[0]).split('_')
        # remove part of file name that refers to "catalog"
        name_parts = [part for part in name_parts if "cat" not in part]

        save_name_base = path.join(save_dir, '_'.join(name_parts) + f"_nside{nside}")
        save_name_mask = save_name_base + "_mask.fits"
        save_name_maps = save_name_base + "_shearmaps.fits"
        save_name_psf = save_name_base + "_psfmaps.fits"

        print("loading catalog...")
        cat = Table.read(cat_file)
        ipix = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)

        # weights map
        print("computing mask...")
        if not args.uniform_weights:
            weights = cat["weight"]
        else:
            weights = np.ones(len(cat))
        # if using an extra mask, modify galaxy weights
        if extra_mask is not None:
            weights *= extra_mask[ipix]
        w_map = make_healpix_map(nside, ipix=ipix, vals=weights)
        mask_b = w_map > 0

        # calculate avg number of galaxies per pixel
        nmap = make_healpix_map(nside, ipix=ipix)
        print(f"number of galaxies per pixel (min / mean / max): {np.min(nmap[mask_b])} / {np.mean(nmap[mask_b]):.2f} / {np.max(nmap[mask_b])}")
        print(f"weighted avg number of galaxies per pixel: {np.average(nmap[mask_b], weights=w_map[mask_b]):.2f}")

        if args.cut_wrt_nmean is not None:
            mask = nmap > np.mean(nmap[mask_b]) * args.cut_wrt_nmean
            w_map = w_map * mask
            mask_b = w_map > 0
            print(f"number of galaxies per pixel (min / mean / max): {np.min(nmap[mask_b])} / {np.mean(nmap[mask_b]):.2f} / {np.max(nmap[mask_b])}")
            print(f"weighted avg number of galaxies per pixel: {np.average(nmap[mask_b], weights=w_map[mask_b]):.2f}")

        if overwrite or not path.exists(save_name_mask):
            hp.write_map(save_name_mask, w_map, overwrite=True)

        # weighted shear maps
        if overwrite or not path.exists(save_name_maps): 
            print("computing shear maps...")
            g_maps = [make_healpix_map(nside, ipix=ipix, vals=cat[f"g_{i}"] * weights) for i in [1, 2]]
            for g_map in g_maps:
                g_map[mask_b] = g_map[mask_b] / w_map[mask_b]
            
            hp.write_map(save_name_maps, g_maps, column_names=["g1", "g2"], overwrite=True)

        # calculate shot noise
        print("computing shot noise...")
        # Following Nicola et al 2020 (eqs 2.2 and 2.24)
        sigma2_e_cat = (cat["g_1"]**2 + cat["g_2"]**2) / 2
        w2_sigma2_map = make_healpix_map(nside, ipix=ipix, vals=weights**2 * sigma2_e_cat)
        shot_noise = hp.nside2pixarea(nside) * np.mean(w2_sigma2_map)
        shot_noise_vals.append(shot_noise)
        print(f"Shot noise in auto pseudo-Cl: {shot_noise:.5e}")

        if args.do_psf:
            if overwrite or not path.exists(save_name_psf):
                # psf elipticity maps
                psf_maps = [make_healpix_map(nside, ipix=ipix, vals=cat[f"psf_e{i}"] * weights) for i in [1, 2]]
                for psf_map in psf_maps:
                    psf_map[mask_b] = psf_map[mask_b] / w_map[mask_b]

                hp.write_map(save_name_psf, psf_maps, column_names=["psf_e1", "psf_e2"], overwrite=True)

    # write shot noise estimates
    with open(path.join(save_dir, f"shot_noise_ests_nside{nside}.txt"), 'w') as f:
        lines = [f"{cat_file}: {shot_noise_vals[i]:.7e}" for i, cat_file in enumerate(cat_files)]
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
