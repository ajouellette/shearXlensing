import argparse
from functools import partial
import os
from os import path
import sys

import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.table import Table
import nx2pt

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
from timer import Timer
from agora_rotations import get_rotator


def get_g_mock_cat(sim_dir, nside, rot):
    pattern = "shear/mock_cats/desy3_mockcat_zbin{zbin}_rot{rot:02}.fits"
    lmax = 3*nside - 1
    tracers = []
    for zbin in range(1, 5):
        cat = Table.read(path.join(sim_dir, pattern.format(zbin=zbin, rot=rot)))
        tracer = nx2pt.CatalogTracer(f"shear zbin {zbin} rot {rot}", [cat["ra"], cat["dec"]],
                                     cat["weight"], lmax, fields=[cat["g_1"], cat["g_2"]])
        tracers.append(tracer)
    return tracers


def get_g_mock_map(sim_dir, nside, rot):
    pattern = "shear/mock_maps/desy3_zbin{zbin}_rot{rot:02}_nside{nside}_{map}.fits"
    shot_noise_file = path.join(sim_dir, f"shear/mock_maps/shot_noise_ests_nside{nside}.txt")
    shot_noise_ests = {}
    if path.exists(shot_noise_file):
        with open(shot_noise_file) as f:
            for line in f.readlines():
                key, val = line.split(": ")
                shot_noise_ests[key] = float(val)
    tracers = []
    for zbin in range(1, 5):
        mask = hp.read_map(path.join(sim_dir, pattern.format(zbin=zbin, rot=rot, nside=nside, map="mask")))
        maps = hp.read_map(path.join(sim_dir, pattern.format(zbin=zbin, rot=rot, nside=nside, map="shearmaps")), field=None)
        tracer = nx2pt.MapTracer(f"shear zbin {zbin} rot {rot}", maps, mask, beam=hp.pixwin(nside))
        for key in shot_noise_ests:
            if f"zbin{zbin}_rot{rot:02}" in key:
                tracer.shot_noise = shot_noise_ests[key]
                break
        if not hasattr(tracer, "shot_noise"):
            print("Warning: no shot noise estimate")
        tracers.append(tracer)
    return tracers


def get_g_true_map(sim_dir, nside, rot, zbin):
    pattern = "shear/mock_cats/desy3_mockcat_zbin{zbin}_rot{rot:02}.fits"
    tracers = []
    return tracers


def get_k_mock_map(sim_dir, nside, rot, binary_mask=False, trunc_mask=False, new=False):
    if nside != 2048:
        raise NotImplementedError
    pattern = "cmbkappa/mockobs/kmapxx_k{dataset}_50{rot:02}_1.fits"
    cmbk_mask = "cmbkappa/mockobs/mask2048_binary_inner_smoothmask_apod_v2.fits"
    cmbk_mask_trunc = "cmbkappa/mockobs/mask2048_binary_inner_smoothmask_apod_v2_trunc2.5.fits"
    cmbk_mask = hp.read_map(path.join(sim_dir, cmbk_mask))
    if trunc_mask:
        apply_mask = hp.read_map(path.join(sim_dir, cmbk_mask_trunc))
    else:
        apply_mask = cmbk_mask
    if binary_mask:
        apply_mask = apply_mask == 1
    tracers = []
    for dataset in ["GMV", "PP"]:
        if new:
            if dataset == "PP":
                continue
            cmbk = hp.read_map(path.join(sim_dir, pattern.replace("mockobs", "agorarec").format(dataset=dataset, rot=rot)))
        else:
            cmbk = hp.read_map(path.join(sim_dir, pattern.format(dataset=dataset, rot=rot)))
        tracer = nx2pt.MapTracer(f"CMB kappa ({dataset})", [cmbk * apply_mask], cmbk_mask**2 * apply_mask,
                                masked_on_input=True)
        tracers.append(tracer)
    return tracers


def get_k_true_map(sim_dir, nside, rot, masked=False, binary_mask=False, new=False):
    cmbk_mask = "cmbkappa/mockobs/mask2048_binary_inner_smoothmask_apod_v2.fits"
    if nside != 2048:
        raise NotImplementedError
    klm = "cmbkappa/sim_alms/agora_raytrace16384_cmbkappa_highzadded_lowLcorrected_cmbkappa_alm.fits"
    if new:
        with np.load(path.join(sim_dir, "cmbkappa/agorarec/klm_rotated_v2.npz")) as f:
            klm = f[str(rot)]
    else:
        with np.load(path.join(sim_dir, "cmbkappa/sim_alms/rotated_alms.npz")) as f:
            klm = f[str(rot)]
    cmbk_true = hp.alm2map(klm, nside)
    if masked:
        cmbk_mask = hp.read_map(path.join(sim_dir, cmbk_mask))
        if binary_mask:
            cmbk_mask = cmbk_mask == 1
    else:
        cmbk_mask = np.ones_like(cmbk_true)
    tracer = nx2pt.MapTracer("simulated CMB kappa", [cmbk_true], cmbk_mask)
    return [tracer]


fields = {
        "shear_mock_cat": get_g_mock_cat,
        "shear_mock_map": get_g_mock_map,
        "shear_true": get_g_true_map,

        "cmbk_mock": get_k_mock_map,
        "cmbk_mock_bmask": partial(get_k_mock_map, binary_mask=True),
        "cmbk_mock_trunc": partial(get_k_mock_map, trunc_mask=True),
        "cmbk_true": get_k_true_map,
        "cmbk_true_masked": partial(get_k_true_map, masked=True),
        "cmbk_true_bmasked": partial(get_k_true_map, masked=True, binary_mask=True),

        "cmbk_mock_new": partial(get_k_mock_map, new=True),
        "cmbk_mock_new_trunc": partial(get_k_mock_map, new=True, trunc_mask=True),
        "cmbk_true_new": partial(get_k_true_map, new=True),
        "cmbk_true_new_masked": partial(get_k_true_map, new=True, masked=True),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog=f"Possible fields: {list(fields.keys())}")
    parser.add_argument("sim_dir")
    parser.add_argument("field1")
    parser.add_argument("field2")
    parser.add_argument("--nside", default=2048, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lmin", type=int, default=30)
    parser.add_argument("--lmax", type=int, default=3500)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--binning", default="linear", choices=["linear", "sqrt", "log"])
    args = parser.parse_args()

    if args.field1 not in fields:
        raise ValueError(f"unknown field {args.field1}")
    if args.field2 not in fields:
        raise ValueError(f"unknown field {args.field2}")

    wksp_cache = "/scratch/aaronjo2/nmt_workspaces"
    output_dir = path.join(args.sim_dir, "cross_corrs")
    os.makedirs(output_dir, exist_ok=True)

    bpw_edges = nx2pt.get_bpw_edges(args.lmin, args.lmax, args.n_bins, args.binning)
    bins = nx2pt.get_nmtbins(args.nside, bpw_edges)
    ell_eff = bins.get_effective_ells()

    output_file = path.join(output_dir, f"{args.field1}_{args.field2}_nside{args.nside}_cls.npz")
    if path.exists(output_file) and not args.overwrite:
        print(f"{output_file} already exists, skipping")
        sys.exit()

    rotations = list(range(1, 11))
    cls_all = []
    bpws_all = []
    for rot_i, rot in enumerate(rotations):
        print(f"---\nRotation {rot}\n---")

        with Timer("loading fields..."):
            fields1 = fields[args.field1](args.sim_dir, args.nside, rot)
            print("Field 1:", fields1)
            if args.field2 == args.field1:
                fields2 = fields1
            else:
                fields2 = fields[args.field2](args.sim_dir, args.nside, rot)
            print("Field 2:", fields2)
        
        # calculate all cross-spectra
        cls_rot = []
        bpws_rot = []
        covs_rot = []
        for i, field1 in enumerate(fields1):
            cls = []
            bpws = []
            covs = []
            for j, field2 in enumerate(fields2):
                with Timer(f"computing Cl {field1.name} x {field2.name}..."):
                    wksp = nx2pt.get_workspace(field1.field, field2.field, bins, wksp_cache=wksp_cache)
                    cl = nmt.compute_coupled_cell(field1.field, field2.field)

                    #if rot_i == 0:
                    #    with Timer("computing cov..."):
                    #        # compute covariance for single realization (spin-0 approx)
                    #        cov_wksp = nx2pt.get_cov_workspace(field1.field, field2.field, wksp_cache=wksp_cache)
                    #        pcl1 = nmt.compute_coupled_cell(field1.field, field1.field)
                    #        fsky1 = np.mean(field1.mask**2)
                    #        pcl2 = nmt.compute_coupled_cell(field2.field, field2.field)
                    #        fsk2 = np.mean(field2.mask**2)
                    #        fsky_cross = np.mean(field1.mask * field2.mask)

                    #        cov = nmt.gaussian_covariance(cov_wksp, *[0,]*4, pcl1[:1] / fsky1, *[cl[:1] / fsky_cross,]*2, pcl2[:1] / fsky2, wksp)
                    #        covs.append(cov)

                    if field1 is field2 and hasattr(field1, "shot_noise"):
                        print("subtracting shot noise...")
                        if field1.spin == 0:
                            cl -= field1.shot_noise
                        else:
                            shot_noise = np.zeros_like(cl)
                            shot_noise[0] = field1.shot_noise * np.ones(len(cl[0]))
                            shot_noise[-1] = field1.shot_noise * np.ones(len(cl[0]))
                            cl -= shot_noise
                    cl = wksp.decouple_cell(cl)
                cls.append(cl)
                bpws.append(wksp.get_bandpower_windows())
            cls_rot.append(cls)
            bpws_rot.append(bpws)
            covs_rot.append(covs)
        cls_all.append(cls_rot)
        bpws_all.append(bpws_rot)
            
    # save cross-spectra and bandpowers
    cls_all = np.array(cls_all)
    bpws_all = np.array(bpws_all)
    covs_all = np.array(covs_rot)
    print("saving to", output_file)
    np.savez(output_file, cls=cls_all, ell_eff=ell_eff, bpws=bpws_all)
