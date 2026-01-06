import argparse
import glob
from os import path
import sys
import warnings
import numpy as np
from scipy import interpolate
import healpy as hp
from astropy.table import Table
import pymaster as nmt
import nx2pt

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
import ccl_interface
from misc import make_healpix_map
from timer import Timer


def load_cl(file, lmax):
    ell, cl = np.loadtxt(file).T
    if ell[-1] < lmax:
        raise ValueError(f"the provided power spectrum does not go to high enough ell")
    if ell[0] > 0:
        ell = np.hstack([[0], ell])
        cl = np.hstack([[0], cl])
    spline = interpolate.CubicSpline(np.log(1+ell), np.log(1+cl), bc_type="not-a-knot", extrapolate=False)
    log_ell = np.log(1 + np.arange(lmax+1))
    cl = np.expm1(spline(log_ell))
    return cl
    

def sample_shear_cat(maps, cat, noise_level=None):
    nside = hp.npix2nside(maps.shape[1])
    if f"ipix_{nside}" in cat.colnames:
        ipix = cat[f"ipix_{nside}"]
    else:
        ipix = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)
        cat[f"ipix_{nside}"] = ipix
    if noise_level is not None:
       phi = np.random.uniform(0, 2*np.pi, len(cat))
       rot = np.exp(2j * phi) * (cat["g_1"] + cat["g_2"]*1j)
       g1_noise = noise_level * rot.real
       g2_noise = noise_level * rot.imag
    else:
        g1_noise = 0
        g2_noise = 0
    cat["g_1"] = maps[0][ipix] + g1_noise
    cat["g_2"] = maps[1][ipix] + g2_noise
    return


def make_shear_maps(cat, nside):
    if f"ipix_{nside}" in cat.colnames:
        ipix = cat[f"ipix_{nside}"]
    else:
        ipix = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)
        cat[f"ipix_{nside}"] = ipix
    wg_maps = np.array([make_healpix_map(nside, ipix=ipix, vals=cat["weight"] * cat[f"g_{i}"])
                        for i in [1, 2]])
    sigma2_cat = (cat["g_1"]**2 + cat["g_2"]**2) / 2
    w2_sigma2_map = make_healpix_map(nside, ipix=ipix, vals=cat["weight"]**2 * sigma2_cat)
    shot_noise = hp.nside2pixarea(nside) * np.mean(w2_sigma2_map)
    #shot_noise_iv = hp.nside2pixarea(nside) * np.mean(make_healpix_map(nside, ipix=ipix, vals=cat["weight"])**2 
    #                                                  / w2_sigma2_map)
    return wg_maps, shot_noise


def process_sims_map_method(cat, sim_maps, nside, bins):
    result = dict(ell_eff=bins.get_effective_ells(), cls_out=[], cls_out_tot=[], shot_noise=[])
    # precompute wksp
    cat[f"ipix_{nside}"] = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)
    wmap = make_healpix_map(nside, ipix=cat[f"ipix_{nside}"], vals=cat["weight"])
    field = nmt.NmtField(wmap, None, spin=2, n_iter=0)
    wksp = nx2pt.get_workspace(field, field, bins, wksp_cache="/scratch/aaronjo2/nmt_workspaces")
    result["bpws"] = wksp.get_bandpower_windows()

    for i, maps in enumerate(sim_maps):
        with Timer(f"processing sim {i+1} / {len(sim_maps)}"):
            maps = hp.read_map(maps, field=None)
            sample_shear_cat(maps, cat)
            wg_maps, shot_noise = make_shear_maps(cat, nside)
            field = nmt.NmtField(wmap, wg_maps, spin=2, masked_on_input=True, n_iter=0)
            pcl = nmt.compute_coupled_cell(field, field)
            template = np.hstack([[0,0], np.ones(pcl.shape[1]-2)])
            nl = [shot_noise * template, np.zeros(pcl.shape[1]), np.zeros(pcl.shape[1]), shot_noise * template]
            cl_tot = wksp.decouple_cell(pcl)
            cl = wksp.decouple_cell(pcl - nl)
            result["cls_out_tot"].append(cl_tot)
            result["cls_out"].append(cl)
            result["shot_noise"].append(wksp.decouple_cell(nl))
    
    return result


def process_sims_cat_method(cat, sim_maps, nside, bins):
    result = dict(ell_eff=bins.get_effective_ells(), cls_out=[])
    # precompute wksp
    field = nmt.NmtFieldCatalog([cat["ra"], cat["dec"]], cat["weight"], None,
                                3*nside-1, spin=2, lonlat=True)
    wksp = nx2pt.get_workspace(field, field, bins, wksp_cache="/scratch/aaronjo2/nmt_workspaces")
    result["bpws"] = wksp.get_bandpower_windows()

    for i, maps in enumerate(sim_maps):
        with Timer(f"processing sim {i+1} / {len(sim_maps)}"):
            maps = hp.read_map(maps, field=None)
            sample_shear_cat(maps, cat)
            field = nmt.NmtFieldCatalog([cat["ra"], cat["dec"]], cat["weight"], [cat["g_1"], cat["g_2"]],
                                        3*nside-1, spin=2, lonlat=True)
            pcl = nmt.compute_coupled_cell(field, field)
            cl = wksp.decouple_cell(pcl)
            result["cls_out"].append(cl)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog")
    parser.add_argument("sim_maps")
    parser.add_argument("output")
    parser.add_argument("--subsample", type=float, default=1, help="subsample catalog by this factor (default: %(default).1f)")
    parser.add_argument("--nside", type=int, default=2048, help="process catalogs at this nside (default: %(default)d)")
    parser.add_argument("--shape-noise", type=float, default=0, help="fraction of catalog shape noise to include in mock catalogs (default: %(default).1f)")
    parser.add_argument("--no-weights", action="store_true", help="do not include catalog weights in mock catalogs")
    parser.add_argument("-c", "--cat_method", action="store_true", help="use catalog based Cls")

    args = parser.parse_args()

    # binning TODO: don't hard-code
    #bpw_edges = [  20.,   50.,   80.,  110.,  140.,  170.,  200.,  230.,  260.,
    #              300.,  347.,  400.,  462.,  534.,  616.,  712.,  822.,  949.,
    #             1096., 1265., 1461., 1687., 1948., 2250., 2598., 3000., 3464., 4000.]
    #bpw_edges = np.hstack([np.linspace(2, 250, 9), np.geomspace(281, 3*2048-1, 23)]).astype(int)
    bpw_edges = np.hstack([np.linspace(2, 120, 5).astype(int),
                           np.geomspace(150, 3*args.nside-1, (np.log(3*args.nside / 150) / 0.19).astype(int)).astype(int)])
    bins = nx2pt.get_nmtbins(args.nside, bpw_edges)
    ell_eff = bins.get_effective_ells()
    
    sim_files = sorted(glob.glob(path.join(args.sim_maps, "*.fits")))
    print(f"Found {len(sim_files)} full-sky simulated maps")

    cat = Table.read(args.catalog)
    if args.subsample < 1:
        print("subsampling catalog by factor", args.subsample)
        inds = np.random.choice(len(cat), int(args.subsample * len(cat)), replace=False)
        cat = cat[inds]
    if args.no_weights:
        print("using uniform catalog weights")
        cat["weight"] = np.ones(len(cat))

    if args.cat_method:
        print("Using catalog-based method")
        result = process_sims_cat_method(cat, sim_files, args.nside, bins)
    else:
        print("Using map-based method")
        result = process_sims_map_method(cat, sim_files, args.nside, bins)

    
    np.savez(args.output, **result)


if __name__ == "__main__":
    main()
