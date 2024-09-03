import os
import sys
import yaml
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib


def get_ell_bins(config):
    """Generate ell bins from config."""
    nside = config["nside"]
    ell_min = config["ell_min"]
    dl = config["delta_ell"]
    ell_bins = np.arange(ell_min, 3*nside, dl)
    return ell_bins


def get_tracer(config, key):
    """Load tracer information."""
    nside = config["nside"]
    name = config[key]["name"]
    data_dir = config[key]["data_dir"]
    if "bins" in config[key].keys():
        bins = config[key]["bins"]
    else:
        bins = 1
    if "use_mask_squared" in config[key].keys():
        use_mask_squared = config[key]["use_mask_squared"]
    else:
        use_mask_squared = False
    if "correct_qu_sign" in config[key].keys():
        correct_qu_sign = config[key]["correct_qu_sign"]
    else:
        correct_qu_sign = False

    print(name, f"({bins} bins)" if bins > 1 else '')

    tracer_bins = []
    for bin in range(bins):
        bin_name = name if bins == 1 else f"{name} (bin {bin})"
        map_file = data_dir + '/' + config[key]["map"].format(bin=bin, nside=nside)
        mask_file = data_dir + '/' + config[key]["mask"].format(bin=bin, nside=nside)
        if "beam" in config[key].keys():
            beam_file = data_dir + '/' + config[key]["beam"].format(bin=bin, nside=nside)
        else:
            beam = np.ones(3*nside)

        maps = np.atleast_2d(hp.read_map(map_file, field=None))
        if correct_qu_sign and len(maps) == 2:
            maps = np.array([-maps[0], maps[1]])

        mask = hp.read_map(mask_file)
        if use_mask_squared: mask = mask**2

        nmt_field = nmt.NmtField(mask, maps, beam=beam)
        tracer = dict(name=bin_name, nmt_field=nmt_field)
        tracer_bins.append(tracer)

    return tracer_bins


def get_workspace(wksp_dir, nmt_field1, nmt_field2, ell_bins):
    """Get the NmtWorkspace for given fields and bins (with caching)."""
    # hash based on masks, beams, and bins
    hash_key = joblib.hash([nmt_field1.get_mask(), nmt_field1.beam,
                            nmt_field2.get_mask(), nmt_field2.beam, ell_bins])
    wksp_file = f"{wksp_dir}/cl/{hash_key}.fits"

    try:
        wksp = nmt.NmtWorkspace.from_file(wksp_file)
    except RuntimeError:
        os.makedirs(f"{wksp_dir}/cl", exist_ok=True)
        bins = nmt.NmtBins.from_edges(ell_bins[:-1], ell_bins[1:])
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, bins)
        wksp.write_to(wksp_file)

    return wksp


def get_cov_workspace(wksp_dir, nmt_field1a, nmt_field2a, nmt_field1b=None, nmt_field2b=None):
    """
    Get the NmtCovarianceWorkspace object needed to calculate the covariance between the
    cross-spectra (field1a, field2a) and (field1b, field2b).
    """
    if nmt_field1b is None and nmt_field2b is None:
        nmt_field1b = nmt_field1a
        nmt_field2b = nmt_field2a
    elif nmt_field1b is None or nmt_field2b is None:
        raise ValueError("Must provide either 2 or 4 fields")

    # hash based on masks and beams
    hash_key = joblib.hash([nmt_field1a.get_mask(), nmt_field1a.beam, nmt_field2a.get_mask(), nmt_field2a.beam,
                            nmt_field1b.get_mask(), nmt_field1b.beam, nmt_field2b.get_mask(), nmt_field2b.beam])
    wksp_file = f"{wksp_dir}/cov/{hash_key}.fits"

    try:
        wksp = nmt.NmtCovarianceWorkspace.from_file(wksp_file)
    except RuntimeError:
        os.makedirs(f"{wksp_dir}/cov", exist_ok=True)
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        wksp.write_to(wksp_file)

    return wksp


def save_sacc(config):
    pass


def main():
    with open(sys.argv[1]) as f:
        config = yaml.full_load(f)

    print(config)

    tracer_keys = [key for key in config.keys() if key.startswith("tracer") ]
    print(f"Found {len(tracer_keys)} tracers")
    tracers = dict()
    for tracer_key in tracer_keys:
        tracer = get_tracer(config, tracer_key)
        tracers[tracer_key] = tracer

    print(tracers)

    xspectra = config["calculate_xspecta"]
    print(f"Found {len(xspectra)} x-spectra to calculate")

    for xspec in xspectra:
        print(xspec)
        if xspec[0] not in tracer_keys or xspec[1] not in tracer_keys:
            raise ValueError(f"Unknown tracer in x-spectrum {xspec}, make sure all tracers are defined in the yaml file")

        tracer1 = tracers[xspec[0]]
        tracer2 = tracers[xspec[1]]

        ell_bins = get_ell_bins(config)
        bins = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:])

        wksp_dir = config["workspace_dir"]
        print("Getting workspace")
        wksp = get_workspace(wksp_dir, tracer1["nmt_field"], tracer2["nmt_field"], ell_bins)


if __name__ == "__main__":
    main()
