import os
from os import path
import sys
import joblib

sys.path.insert(0, "/projects/caps/aaronjo2/shearXlensing/code")
from catalogs import DESY3ShearCat


def main():
    if len(sys.argv) < 2:
        print("Must specify location of DES Y3 catalogs index file")
        return

    index_file = sys.argv[1]


    # save catalogs at {data_dir}/../calibrated_catalogs
    save_dir = path.join(path.dirname(index_file), '..', "calibrated_catalogs")
    print("Will save calibrated catalogs to:", save_dir, '\n')
    os.makedirs(save_dir, exist_ok=True)

    for zbin in [1, 2, 3, 4, None]:
        if zbin is not None:
            print("z-bin", zbin)
        else:
            print("Combined bin")

        print("loading catalog and calibrating...")
        cat = DESY3ShearCat(index_file, zbin=zbin)
        print(len(cat.data), "galaxies")
        print("total multiplicative bias:", cat.R)

        if zbin is not None:
            save_file = path.join(save_dir, f"DESY3_shearcat_zbin{zbin-1}.fits")
        else:
            save_file = path.join(save_dir, f"DESY3_shearcat_combined.fits")
        print("writing catalog to", save_file)
        # save catalog as astropy table in FITS format
        cat.data.write(save_file, overwrite=True)

        if zbin is not None:
            print()


if __name__ == "__main__":
    main()
