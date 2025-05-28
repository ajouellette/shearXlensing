import argparse
import os
from os import path
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
import catalogs
from timer import Timer
from misc import make_healpix_map


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("index_file")
    parser.add_argument("--nside", default=2048, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask-thresh", default=0, type=float)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    if not path.isdir(args.output):
        os.makedirs(args.output)

    maglim_cats = catalogs.DESY3MaglimCat(args.index_file)

    shot_noise_ests = []
    for zbin in range(6):
        map_name = path.join(args.output, f"delta_zbin{zbin}_nside{args.nside}.fits")
        mask_name = path.join(args.output, f"mask_zbin{zbin}_nside{args.nside}.fits")
        print("bin", zbin)
        if path.isfile(map_name) and path.isfile(mask_name) and not args.overwrite:
            print("map and mask already exist, skipping")
            continue
        maps = maglim_cats.get_map(args.nside, zbin, mask_thresh=args.mask_thresh)
        print(len(getattr(maglim_cats, f"cat_zbin{zbin}")), "galaxies in bin")
        hp.write_map(map_name, maps["delta"], overwrite=args.overwrite)
        hp.write_map(mask_name, maps["mask"], overwrite=args.overwrite)
        shot_noise_ests.append(maps["shot_noise"])

    print("Shot noise estimates:")
    print(shot_noise_ests)



if __name__ == "__main__":
    main()
