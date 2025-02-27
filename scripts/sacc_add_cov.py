import argparse

import numpy as np
import sacc

import nx2pt
import ccl_interface


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
    s.add_covariance(s_other.covariance.covmat)


def marginalize_m(s, theory):
    """Compute covariance term due to marginalizing over shear m bias."""
    # sanity checks
    if s.tracers.keys() != theory.tracers.keys():
        raise ValueError("Theory object does not have the same tracers as the sacc file.")

    cov = np.zeros((len(s.mean), len(s.mean)))
    # loop over all power spectra in the sacc file
    for tracers1 in s.get_tracer_combinations():
        for dtype1 in s.get_data_types(tracers1):
            # skip all B-modes
            if 'b' in dtype:
                continue
            inds1 = s.indices(tracers=tracers1, data_type=dtype1)
            for tracers2 in s.get_tracer_combinations():
                for dtype2 s.get_data_types(tracers2):
                    if 'b' in dtype2:
                        continue
                    inds2 = s.indices(tracers=tracers2. data_type=dtype2)
                    # get bandpower windows
                    bpws = s.get_bandpower_windows(inds1).weight.T
                    bpws_b = s.get_bandpower_windows(inds2).weight.T
                    # compute block
                    block = theory.get_cov_marge_m(*tracers1, *tracers2,
                                                   bpws=bpws, bpws_b=bpws_b)
                    cov[np.ix_(inds1, inds2)] = block
    return cov


def marginalize_dz(s):
    raise NotImplementedError


def main():
    description = """
    Add a covariance to a sacc file that contains the cross-spectra of a set of tracers.

    Can do one or more of the following:
    - take the covariance from another sacc file (must have the exact same tracers and cross-spectra)
    - marginalize over shear multiplicative bias
    - calculate non-Gaussian covariance terms (SSC + connected trispectrum)

    If marginalizing over shear bias and/or computing non-Gaussian terms, then a theory.yaml file is needed.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sacc_file")
    parser.add_argument("-o", "--output", help="where to write new sacc file")
    parser.add_argument("--from-sacc", help="use covariance from another sacc file")
    parser.add_argument("-m", "--marg-shear-bias", action="store_true", help="marginalize over shear multiplicative bias")
    parser.add_argument("-t", "--theory", help="file describing tracers and how to calculate theory spectra")
    parser.add_argument("--non-gaussian", action="store_true", help="compute non-Gaussian covariance terms")

    args = parser.parse_args()
    print(args)

    s = sacc.Sacc.load_fits(args.sacc_file)

    # load covariance from another file
    if args.from_sacc is not None:
        s_other = sacc.Sacc.load_fits(args.from_sacc)
        add_cov_from_sacc(s, s_other)

    # load tracer info and compute theory quantities
    if args.theory is not None:
        with open(args.theory) as f:
            config = yaml.safe_load(f)
            theory = ccl_interface.CCLTheory(config)
    else:
        theory = None

    if args.marg_shear_bias:
        if theory is None:
            raise ValueError("Must provide tracer info to calculate analytical covariances")
        cov_m = marginalize_m(s, theory)
        full_cov = s.covariance.covmat + cov_m
        s.add_covariance(full_cov, overwrite=True)


    # save sacc file
    if args.output is not None:
        s.save_fits(args.output)
    else:
        s.save_fits(args.sacc_file, overwrite=True)


if __name__ == "__main__":
    main()
