#!/usr/bin/env python
import argparse
import datetime
import re
from os import path
import sys

import numpy as np
import sacc
import yaml

import nx2pt
sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
import ccl_interface


wl_keys = ["wl", "tracer_wl", "gs"]  # prefixes that indicate a galaxy shear tracer
ck_keys = ["ck", "tracer_ck"]  # prefixes that indicate a CMB lensing tracer


def get_trailing_int(s):
    """
    Extract a trailing integer from the given string.

    taken from https://stackoverflow.com/questions/7085512/check-what-number-a-string-ends-with-in-python
    """
    m = re.search(r"\d+$", s)
    return int(m.group()) if m is not None else None


def standardize_sacc(s, fix_bin_numbers=False):
    """Standardize a sacc file to work with cosmosis analysis pipelines."""
    # Rename tracers to standard names
    tracer_names = list(s.tracers.keys())
    for tracer in tracer_names:
        bin_number = get_trailing_int(tracer)
        if bin_number is None:
            bin_number = 0
        else:
            if fix_bin_numbers:
                bin_number -= 1
        new_name = tracer
        for key in wl_keys:
            if tracer.startswith(key):
                new_name = f"wl_{bin_number}"
                break
        for key in ck_keys:
            if tracer.startswith(key):
                new_name = f"ck_{bin_number}"
                break
        if tracer != new_name:
            print(f"Renaming '{tracer}' to '{new_name}'")
            s.rename_tracer(tracer, new_name)
    # re-order (ck, wl) to (wl, ck)
    for data_point in s.data:
        tracers = data_point.tracers
        if len(tracers) == 2:
            if tracers[0].startswith("ck") and tracers[1].startswith("wl"):
                new_tracers = (tracers[1], tracers[0])
                print(f"Switching '{tracers}' to '{new_tracers}'")
                data_point.tracers = new_tracers


def new_sacc_with_tracers(s):
    """Create an "empty copy" of s."""
    s_new = sacc.Sacc()
    # copy metadata
    for key in s.metadata.keys():
        s_new.metadata[key] = s.metadata[key]
    # copy over tracer objects
    for tracer in s.tracers.values():
        s_new.add_tracer_object(tracer)
    return s_new


def add_cov_from_sacc(s, s_other):
    """Get a covariance from another sacc file."""
    # sanity checks
    if s.tracers.keys() != s_other.tracers.keys():
        raise ValueError("two sacc files do not have the same tracers")
    if s.get_tracer_combinations() != s_other.get_tracer_combinations():
        raise ValueError("two sacc files do not have the same tracer combinations")
    if s.get_data_types() != s_other.get_data_types():
        raise ValueError("two sacc files do not have the same data types")
    if len(s.mean) != len(s_other.mean):
        raise ValueError("two sacc files do not have equal data vector lengths")

    # copy over covariance
    s.add_covariance(s_other.covariance.covmat, overwrite=True)


def correct_cmbk_tf(s, tf):
    """Correct cross-spectra for a CMB lensing transfer function."""
    if "ck_0" not in s.tracers.keys():
        return s
    s_new = new_sacc_with_tracers(s)
    # keep track of standard deviations in order to rescale covariance matrix
    stds_old = np.sqrt(np.diag(s.covariance.covmat))
    stds_new = []
    for comb in s.get_tracer_combinations():
        for dtype in s.get_data_types(comb):
            ell, cl, inds = s.get_ell_cl(dtype, *comb, return_ind=True)
            bpw = s.get_bandpower_windows(inds)
            # spectra that do not involve CMB lensing or involve B-modes do not get modified
            if 'b' in dtype or "ck_0" not in comb:
                s_new.add_ell_cl(dtype, *comb, ell, cl, window=bpw)
                stds_new.append(stds_old[inds])
            else:
                # can't do a CMB lensing auto
                if comb[0] == comb[1]:
                    raise NotImplementedError("Cannot correct a CMB lensing auto-spectrum")
                # check that banpowers are correct
                if not (ell == tf["ell"]).all():
                    raise ValueError(f"bandpowers of cross-spectrum {comb} and transfer function do not match")
                # do correction
                cl_corr = cl / tf["tf"]
                s_new.add_ell_cl(dtype, *comb, ell, cl_corr, window=bpw)
                std_corr = np.abs(cl_corr) * np.sqrt((stds_old[inds] / cl)**2 + (tf["tf_err"] / tf["tf"])**2)
                stds_new.append(std_corr)
    # rescale covariance
    stds_new = np.hstack(stds_new)
    cov_new = s.covariance.covmat * np.outer(stds_new, stds_new) / np.outer(stds_old, stds_old)
    s_new.add_covariance(cov_new)
    # add a metadata flag
    s_new.metadata["cmbk_tf_corrected"] = True
    return s_new


def calc_cov_margem(s, theory, method="des"):
    """Compute covariance term due to marginalizing over shear m bias."""
    cov = np.zeros((len(s.mean), len(s.mean)))
    # loop over all power spectra in the sacc file
    for tracers1 in s.get_tracer_combinations():
        for dtype1 in s.get_data_types(tracers1):
            # skip all B-modes
            if 'b' in dtype1:
                continue
            inds1 = s.indices(tracers=tracers1, data_type=dtype1)
            for tracers2 in s.get_tracer_combinations():
                for dtype2 in s.get_data_types(tracers2):
                    if 'b' in dtype2:
                        continue
                    inds2 = s.indices(tracers=tracers2, data_type=dtype2)
                    # get bandpower windows
                    bpws = s.get_bandpower_windows(inds1).weight.T
                    bpws_b = s.get_bandpower_windows(inds2).weight.T
                    # compute block
                    block = theory.get_cov_marg_m(*tracers1, *tracers2,
                                                   bpws=bpws, bpws_b=bpws_b, method=method)
                    cov[np.ix_(inds1, inds2)] = block
    return cov


def calc_cov_ng(s, theory, kind="ssc"):
    """Compute covariance term due to marginalizing over shear m bias."""
    if kind not in ["ssc", "cng", "both"]:
        raise ValueError
    cov = np.zeros((len(s.mean), len(s.mean)))
    # loop over all power spectra in the sacc file
    for tracers1 in s.get_tracer_combinations():
        for dtype1 in s.get_data_types(tracers1):
            # skip all B-modes
            if 'b' in dtype1:
                continue
            ell_a, _, inds1 = s.get_ell_cl(dtype1, *tracers1, return_ind=True)
            for tracers2 in s.get_tracer_combinations():
                for dtype2 in s.get_data_types(tracers2):
                    if 'b' in dtype2:
                        continue
                    ell_b, _, inds2 = s.get_ell_cl(dtype2, *tracers2, return_ind=True)
                    # compute block
                    if kind == "ssc" or kind == "both":
                        block = theory.get_cov_ssc(*tracers1, ell_a, *tracers2, ell_b)
                        if kind == "both":
                            block += theory.get_cov_cng(*tracers1, ell_a, *tracers2, ell_b)
                    else:
                        block = theory.get_cov_cng(*tracers1, ell_a, *tracers2, ell_b)
                    cov[np.ix_(inds1, inds2)] = block
    return cov


def add_non_gauss_cov(s, theory, kind="ssc"):
    # sanity checks
    for tracer in s.tracers.keys():
        if tracer not in theory.tracers.keys():
            raise ValueError("Theory object does not have the same tracers as the sacc file.")
    s_new = s.copy()
    # calculate new covariance
    cov = s.covariance.covmat + calc_cov_ng(s, theory, kind=kind)
    s_new.add_covariance(cov, overwrite=True)
    # add a metadata flag
    s_new.metadata["non_gaussian_cov"] = True
    return s_new


def marginalize_m(s, theory):
    """Correct for any multiplicative biases and marginalize over their uncertainty."""
    # sanity checks
    for tracer in s.tracers.keys():
        if tracer not in theory.tracers.keys():
            raise ValueError("Theory object does not have the same tracers as the sacc file.")
    s_new = new_sacc_with_tracers(s)
    # copy over Cls, correcting for multiplicative biases
    # any Cl involving B-modes is copied over unchanged
    for comb in s.get_tracer_combinations():
        for dtype in s.get_data_types(comb):
            ell, cl, inds = s.get_ell_cl(dtype, *comb, return_ind=True)
            bpw = s.get_bandpower_windows(inds)
            if 'b' in dtype:
                s_new.add_ell_cl(dtype, *comb, ell, cl, window=bpw)
            else:
                tr1, tr2 = comb
                m1 = theory.tracers[tr1]["args"].get("m_bias", 0)
                m2 = theory.tracers[tr2]["args"].get("m_bias", 0)
                cl /= (1 + m1) * (1 + m2)
                s_new.add_ell_cl(dtype, *comb, ell, cl, window=bpw)
    # calculate new covariance
    cov = s.covariance.covmat + calc_cov_margem(s, theory)
    s_new.add_covariance(cov)
    # add a metadata flag
    s_new.metadata["marginalized_over_m_bias"] = True
    return s_new


def marginalize_dz(s):
    raise NotImplementedError


def main():
    description = """
    Post-process a sacc file that contains the cross-spectra of a set of tracers.

    Can do one or more of the following:
    - take the covariance from another sacc file (must have the exact same tracers and cross-spectra)
    - correct cross-spectra for a CMB lensing transfer function
    - marginalize over shear multiplicative bias
    - calculate non-Gaussian covariance terms (SSC + connected trispectrum)

    Also does some tracer standardization before writing the output sacc file.

    If marginalizing over shear bias and/or computing non-Gaussian terms, then a theory.yaml file is needed.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sacc_file")
    parser.add_argument("-o", "--output", help="where to write new sacc file")
    parser.add_argument("-p", "--inplace", action="store_true", help="modify sacc file in-place")
    parser.add_argument("--from-sacc", help="use covariance from another sacc file")
    parser.add_argument("--cmbk-tf", help="correct for a CMB lensing transfer function")
    parser.add_argument("-m", "--marg-shear-bias", action="store_true", help="marginalize over shear multiplicative bias")
    parser.add_argument("--marg-method", default="des", choices=["des", "kids"], help="method used to marginalize over shear bias")
    parser.add_argument("-t", "--theory", help="file describing tracers and how to calculate theory spectra")
    parser.add_argument("--non-gaussian", action="store_true", help="compute non-Gaussian covariance terms (SSC + cNG)")
    parser.add_argument("--ssc", action="store_true", help="compute SSC term")
    parser.add_argument("--cng", action="store_true", help="compute cNG term")
    
    theory = None
    args = parser.parse_args()
    if args.output is None and not args.inplace:
        raise ValueError("Must provide an output location")

    s = sacc.Sacc.load_fits(args.sacc_file)
    # start with standardizing tracers
    standardize_sacc(s)

    # load covariance from another file
    if args.from_sacc is not None:
        print("Loading covariance from", args.from_sacc)
        s_other = sacc.Sacc.load_fits(args.from_sacc)
        standardize_sacc(s_other)
        add_cov_from_sacc(s, s_other)

    # correct for a CMB lensing transfer function
    if args.cmbk_tf is not None:
        print("Correcting for CMB lensing transfer function...")
        tf = dict()
        with np.load(args.cmbk_tf) as f:
            for key in f.keys():
                tf[key] = f[key]
        s = correct_cmbk_tf(s, tf)

    # shear magnification bias marginalization
    if args.marg_shear_bias:
        print("Marginalizing over shear multiplicative bias...")
        if theory is None and args.theory is not None:
            print("Loading tracer info from", args.theory)
            with open(args.theory) as f:
                config = yaml.safe_load(f)
                theory = ccl_interface.CCLTheory(config)
        else:
            raise ValueError("Must provide tracer info to calculate analytical covariances")
        s = marginalize_m(s, theory, method=args.marg_method)

    # non-Gaussian covariance terms
    if args.non_gaussian or args.ssc or args.cng:
        if args.ssc and not args.cng:
            kind = "ssc"
            print("Computing super-sample covariance terms...")
        elif args.cng and not args.ssc:
            kind = "cng"
            print("Computing connected non-Gaussian covariance terms...")
        else:
            kind = "both"
            print("Computing non-Gaussian covariance terms (SSC + cNG)...")
        if theory is None and args.theory is not None:
            print("Loading tracer info from", args.theory)
            with open(args.theory) as f:
                config = yaml.safe_load(f)
                theory = ccl_interface.CCLTheory(config)
        else:
            raise ValueError("Must provide tracer info to calculate analytical covariances")
        s = add_non_gauss_cov(s, theory, kind=kind)

    def update_timestamp(s, modified=False):
        if modified:
            s.metadata["modified"] = datetime.date.today().isoformat()
            return
        for key in s.metadata.keys():
            if key in ["creation", "created", "creation_date", "creation_time"]:
                s.metadata[key] = datetime.date.today().isoformat()
                return
        s.metadata["created"] = datetime.date.today().isoformat()

    # save sacc file
    print("Writing sacc file...")
    if args.output is not None:
        update_timestamp(s, modified=False)
        s.save_fits(args.output, overwrite=True)
    else:
        update_timestamp(s, modified=True)
        s.save_fits(args.sacc_file, overwrite=True)


if __name__ == "__main__":
    main()
