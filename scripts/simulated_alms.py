import argparse
from functools import partial
import os
from os import path
import glob
import numpy as np
import healpy as hp


def get_reduced_shear(filename):
    k, g1, g2 = hp.read_map(filename, field=None)
    g1 /= (1 - k)
    g2 /= (1 - k)
    return g1, g2


field_types = {
        "cmbkappa": {"pattern": "cmbkappa/*cmbkappa*.fits",
                     "load_field": hp.read_map,
                     "spin": 0},
        "shear": {"pattern": "shear/raytrace_kg1g2*_fullsky.fits",
                  "load_field": partial(hp.read_map, field=[1,2]),
                  "spin": 2},
        "reduced_shear": {"pattern": "shear/raytrace_kg1g2*_fullsky.fits",
                          "load_field": get_reduced_shear,
                          "spin": 2},
        "ia": {"pattern": "shear/raytrace_IAkg1g2*_fullsky*.fits",
               "load_field": partial(hp.read_map, field=[1,2]),
               "spin": 2}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate alms from simulated full-sky fields.")
    parser.add_argument("sim_dir")
    parser.add_argument("field_type")
    parser.add_argument("--lmax-save", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.field_type not in field_types:
        raise NotImplementedError(f"{field_type} is not implemented")

    field_info = field_types[args.field_type]
    files = glob.glob(path.join(args.sim_dir, field_info["pattern"]))
    print("found", len(files), args.field_type, "maps")

    for filename in files:
        dirname, basename = path.split(filename)
        dirname = path.join(dirname, "sim_alms")
        os.makedirs(dirname, exist_ok=True)

        basename, ext = path.splitext(basename)
        basename += f"_{args.field_type}_alm" + ext
        save_name = path.join(dirname, basename)

        if path.exists(save_name) and not args.overwrite:
            print(f"{save_name} already exists, skipping")
            continue

        print("loading", filename)
        maps = field_info["load_field"](filename)
        nside = hp.npix2nside(len(maps[0])) if field_info["spin"] != 0 else hp.npix2nside(len(maps))

        print("computing alms at nside", nside)
        if field_info["spin"] > 0:
            #alm = hp.map2alm_spin(maps, spin=field_info["spin"], lmax=2*nside)
            alm = hp.map2alm_spin(maps, spin=field_info["spin"])
        else:
            #alm = hp.map2alm(maps, lmax=2*nside)
            alm = hp.map2alm(maps)

        print("saving to", save_name)
        hp.write_alm(save_name, alm, lmax=args.lmax_save, overwrite=True)
