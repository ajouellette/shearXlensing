import argparse
from functools import partial
import sys
import os
from os import path
import glob
import numpy as np
import healpy as hp
import pymaster as nmt

sys.path.insert(0, "/projects/ncsa/caps/aaronjo2/shearXlensing/code")
from agora_rotations import get_rotator
from timer import Timer


def get_reduced_shear(filename):
    k, g1, g2 = hp.read_map(filename, field=None)
    g1 /= (1 - k)
    g2 /= (1 - k)
    return g1, g2


field_types = {
        "cmbkappa": {"pattern": "cmbkappa/*cmbkappa*.fits",
                     "load_field": hp.read_map,
                     "spin": 0},
        "shear": {"pattern": "shear/fullsky/maps/raytrace_kg1g2*_fullsky.fits",
                  "load_field": partial(hp.read_map, field=[1,2]),
                  "spin": 2},
        "ia": {"pattern": "shear/fullsky/maps/raytrace_IAkg1g2*_fullsky*.fits",
               "load_field": partial(hp.read_map, field=[1,2]),
               "spin": 2},
        "reduced_shear": {"pattern": "shear/fullsky/maps/raytrace_reduced_shear_des_*_fullsky.fits",
                          "load_field": partial(hp.read_map, field=None),
                          "spin": 2},
        "reduced_shear_ia": {"pattern": "shear/fullsky/maps/raytrace_reduced_shear_ia_des_*_fullsky.fits",
                             "load_field": partial(hp.read_map, field=None),
                             "spin": 2},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate alms from simulated full-sky fields.", epilog=f"Possible fields: {list(field_types.keys())}")
    parser.add_argument("sim_dir")
    parser.add_argument("field_type")
    parser.add_argument("-o", "--output", help="output dir")
    parser.add_argument("--no-rot", action="store_true")
    parser.add_argument("--lmax-save", type=int, default=8000, help="default: %(default)s")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nmt-niter", default=3, type=int, help="default: %(default)d")
    args = parser.parse_args()

    if args.field_type not in field_types:
        raise NotImplementedError(f"{field_type} is not implemented")

    field_info = field_types[args.field_type]
    files = glob.glob(path.join(args.sim_dir, field_info["pattern"]))
    if args.output is None:
        save_dir = path.join(path.dirname(path.join(args.sim_dir, field_info["pattern"])), "../alms")
    else:
        save_dir = args.output
    print("will save alms to", save_dir)
    print("found", len(files), args.field_type, "maps")

    for filename in files:
        basename, _ = path.splitext(path.basename(filename))
        basename += "_rot_alms.npz"
        save_name = path.join(save_dir, basename)

        if path.exists(save_name) and not args.overwrite:
            print(f"{save_name} already exists, skipping")
            continue

        print("loading", filename)
        maps = field_info["load_field"](filename)
        map_info = nmt.NmtMapInfo(None, [maps.shape[-1],])
        alm_info = nmt.NmtAlmInfo(map_info.get_lmax())
        lmax_save = min(map_info.get_lmax(), args.lmax_save)
        
        rot_alms = dict()
        with Timer(f"computing alms at nside {map_info.nside}"):
            alm = nmt.map2alm(maps, field_info["spin"], map_info, alm_info, n_iter=args.nmt_niter)
            alm = hp.resize_alm(alm, map_info.get_lmax(), map_info.get_lmax(), lmax_save, lmax_save)
            # make sure alms are np.complex128 to avoid healpy bug(?)
            if isinstance(alm, list):
                alm = [x.astype(np.complex128) for x in alm]
            else:
                alm = alm.astype(np.complex128)
            print([hp.Alm.getlmax(len(x)) for x in alm] if isinstance(alm, list) else hp.Alm.getlmax(len(alm)))
            rot_alms['1'] = alm

        # don't need maps after calculating alms
        del maps
        
        if not args.no_rot:
            # compute all rotations
            for rot_i in range(1, 10):
                with Timer(f"computing rotation {rot_i}"):
                    rot = get_rotator(rot_i)
                    if field_info["spin"] > 0:
                        rot_alm = [rot.rotate_alm(alm_i) for alm_i in alm]
                    else:
                        rot_alm = rot.rotate_alm(alm)
                    rot_alms[str(rot_i+1)] = rot_alm

        print("saving to", save_name)
        np.savez(save_name, **rot_alms)
