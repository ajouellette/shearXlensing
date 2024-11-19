import argparse
import os
from os import path
import time

import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.table import Table
import nx2pt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir")
    parser.add_argument("--nside", default=2048, type=int)
    parser.add_argument("--overwrite", action="store_true")
    #parser.add_argument("
    args = parser.parse_args()

    cmbk_pattern = "cmbkappa/mockobs/kmapxx_k{dataset}_50{rot:02}_1.fits"
    cmbk_mask = "cmbkappa/mockobs/mask2048_binary_inner_smoothmask_apod_v2.fits"
    shear_pattern = "shear/mock_cats/desy3_mockcat_zbin{zbin}_rot{rot:02}.fits"
    wksp_cache = "/home/aaronjo2/scratch/nmt_workspaces"
    output_dir = path.join(args.sim_dir, "cross_corrs")
    os.makedirs(output_dir, exist_ok=True)

    bpw_edges = nx2pt.get_bpw_edges(30, 3800, 24, "linear")
    bins = nx2pt.get_nmtbins(args.nside, bpw_edges)
    ell_eff = bins.get_effective_ells()

    datasets = ["GMV", "PP"]
    zbins = list(range(1, 5))
    rotations = list(range(1, 11))

    cmbk_mask = hp.ud_grade(hp.read_map(path.join(args.sim_dir, cmbk_mask)), args.nside)
    
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        for zbin in zbins:
            print(f"z-bin: {zbin}")
            output_file = path.join(output_dir, f"cmbk_{dataset}_shear_zbin{zbin}_nside{args.nside}_masked_on_input_cls.npz")
            if path.exists(output_file) and not args.overwrite:
                print(f"{output_file} already exists, skipping")
                continue
            
            cls = []
            for rot in rotations:
                print(f"---\nRotation {rot}\n---")
                print("reading data...")
                t1 = time.perf_counter()
                cmbk = hp.read_map(path.join(args.sim_dir, cmbk_pattern.format(dataset=dataset, rot=rot)))
                shear = Table.read(path.join(args.sim_dir, shear_pattern.format(zbin=zbin, rot=rot)))
                t2 = time.perf_counter()
                print(f"  ({t2 - t1:.1f} s)")
                print("constructing tracers...")
                t1 = time.perf_counter()
                cmbk = nx2pt.MapTracer(f"CMB kappa ({dataset})", [cmbk], cmbk_mask**2, masked_on_input=True)
                print(cmbk)
                shear = nx2pt.CatalogTracer(f"DES shear (bin {zbin})", [shear["ra"], shear["dec"]], shear["weight"],
                                            3*args.nside-1, fields=[shear["g_1"], shear["g_2"]])
                print(shear)
                t2 = time.perf_counter()
                print(f"  ({t2 - t1:.1f} s)")
                print("getting workspace...")
                t1 = time.perf_counter()
                wksp = nx2pt.get_workspace(cmbk.field, shear.field, bins, wksp_cache=wksp_cache)
                t2 = time.perf_counter()
                print(f"  ({t2 - t1:.1f} s)")
                print("computing Cl...")
                t1 = time.perf_counter()
                cl = nmt.compute_coupled_cell(cmbk.field, shear.field)
                cl = wksp.decouple_cell(cl)
                t2 = time.perf_counter()
                print(f"  ({t2 - t1:.1f} s)")
                cls.append(cl)
    
            cls = np.array(cls)
            bpws = wksp.get_bandpower_windows()

            print("saving to", output_file)
            np.savez(output_file, cls=cls, ell_eff=ell_eff, bpws=bpws)
