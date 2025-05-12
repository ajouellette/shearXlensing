import argparse
import glob
from os import path
import numpy as np
import treecorr


def main():
    parser = argparse.ArgumentParser(description="compute real space correlation functions for a set of shear catalogs")
    parser.add_argument("catalog_dir", help="directory that contains shear catalog files")
    parser.add_argument("-o", "--output", help="output file")
    parser.add_argument("--bin-slop", default=0.01, type=float, help="treecorr bin_slop paramerer, default: 0.01")
    parser.add_argument("--curved-sky", action="store_true", help="do curved sky calculations, default: False")
    parser.add_argument("--brute", action="store_true", help="do brute-force calculations, default: False")
    args = parser.parse_args()
    
    catalogs = sorted(glob.glob(path.join(args.catalog_dir, "*.fits")))
    print(f"found {len(catalogs)} redshift bins")

    result = {}

    for i in range(len(catalogs)):
        cat1 = treecorr.Catalog(file_name=catalogs[i], ra_col="ra", dec_col="dec",
                                g1_col="g_1", g2_col="g_2", w_col="weight",
                                ra_units="degrees", dec_units="degrees")
        for j in range(i, len(catalogs)):
            cat2 = treecorr.Catalog(file_name=catalogs[j], ra_col="ra", dec_col="dec",
                                    g1_col="g_1", g2_col="g_2", w_col="weight",
                                    ra_units="degrees", dec_units="degrees")
            
            metric = "Arc" if args.curved_sky else "Euclidean"
            gg = treecorr.GGCorrelation(min_sep=2.5, max_sep=250, nbins=20, sep_units="arcmin",
                                        bin_slop=args.bin_slop, metric=metric, brute=args.brute)
            print(f"computing {i} x {j}")
            gg.process(cat1, cat2=cat2)
            result[f"xip_{i}_{j}"] = gg.xip
            result[f"xim_{i}_{j}"] = gg.xim
            result[f"theta_{i}_{j}"] = gg.meanr

            del cat2, gg
        del cat1

    print("done")
    print("saving to", args.output)
    np.savez(args.output, **result)


if __name__ == "__main__":
    main()
