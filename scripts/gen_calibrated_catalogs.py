import argparse
import os
from os import path
import sys
import joblib

sys.path.insert(0, "/projects/ncsa/caps/aaronjo2/shearXlensing/code")
from catalogs import DESY3ShearCat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file")
    parser.add_argument("--sample", default="all", choices=["all", "blue", "red"])
    parser.add_argument("-o", "--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index_file = args.index_file

    if args.output is None:
        # save catalogs at {data_dir}/../calibrated_catalogs
        save_dir = path.join(path.dirname(index_file), '..', "calibrated_catalogs")
    else:
        save_dir = args.output
    print("Will save calibrated catalogs to:", save_dir, '\n')
    os.makedirs(save_dir, exist_ok=True)

    for zbin in [1, 2, 3, 4, None]:
        if zbin is not None:
            print("z-bin", zbin)
        else:
            if args.sample != "all":
                continue
            print("Combined bin")

        if zbin is not None:
            save_file = path.join(save_dir, f"DESY3_shearcat_zbin{zbin-1}.fits")
        else:
            save_file = path.join(save_dir, f"DESY3_shearcat_combined.fits")
        if path.isfile(save_file) and not args.overwrite:
            print(f"{save_file} already exists, skipping")
            continue

        print("loading catalog and calibrating...")
        cat = DESY3ShearCat(index_file, zbin=zbin, sample=args.sample)
        print(len(cat.data), "galaxies")
        print("total multiplicative bias:", cat.R)

        print("writing catalog to", save_file)
        # save catalog as astropy table in FITS format
        cat.data.write(save_file, overwrite=True)

        if zbin is not None:
            print()


if __name__ == "__main__":
    main()
