import argparse
import sys
from os import path
import numpy as np
import pyccl as ccl
from joblib import Parallel, delayed, parallel_config, cpu_count

sys.path.insert(0, path.join(path.dirname(path.realpath(__file__)), "../code"))
import chains


def get_spk(params, kgrid, return_spk_h=False):
    cosmo_params = dict(Omega_c=params["omega_m"]-params["omega_b"], Omega_b=params["omega_b"],
                        h=params["h"], n_s=params["n_s"], A_s=params["a_s"], m_nu=0.06)
    cosmo_dm = ccl.Cosmology(**cosmo_params, matter_power_spectrum="camb",
                             extra_parameters={"camb": {"halofit_version": "mead2020"}})
    cosmo_fb = ccl.Cosmology(**cosmo_params, matter_power_spectrum="camb",
                             extra_parameters={"camb": {"halofit_version": "mead2020_feedback",
                                                        "HMCode_logT_AGN": params["logt_agn"]}})
    spk = cosmo_fb.nonlin_power(kgrid, 1) / cosmo_dm.nonlin_power(kgrid, 1)
    if not return_spk_h:
        return spk
    spk_h = cosmo_fb.nonlin_power(kgrid / params["h"], 1) / cosmo_dm.nonlin_power(kgrid / params["h"], 1)
    return spk, spk_h


def calc_sk_samples(chain, kgrid=np.geomspace(0.01, 10, 50), do_kh=False, samples=2000, max_inner_threads=1):
    if samples < len(chain.weights):
        use_samples = np.random.choice(len(chain.weights), size=samples, replace=False, p=chain.weights)
    else:
        use_samples = np.arange(len(chain.weights))
    spks = []
    Om = chain.getParams().omega_m[use_samples]
    Ob = chain.getParams().omega_b[use_samples]
    h0 = chain.getParams().h0[use_samples]
    n_s = chain.getParams().n_s[use_samples]
    A_s = 1e-9 * chain.getParams().a_s_1e9[use_samples]
    logTagn = chain.getParams().logt_agn[use_samples]

    n_jobs = cpu_count() // max_inner_threads
    with parallel_config(backend="loky", verbose=10, n_jobs=n_jobs, inner_max_num_threads=max_inner_threads):
        spks = Parallel()(
                delayed(get_spk)(dict(omega_m=Om[i], omega_b=Ob[i], h=h0[i], n_s=n_s[i], a_s=A_s[i], logt_agn=logTagn[i]),
                    kgrid, return_spk_h=do_kh)
                for i in range(len(use_samples)))
    weights = chain.weights[use_samples]
    weights /= np.sum(weights)
    k_h_grids = np.expand_dims(kgrid, 0) / np.expand_dims(h0, 1)
    res = {"k": kgrid, "k_h": k_h_grids, "weights": weights}
    if not do_kh:
        spk = np.array(spks)
        return res | {"spk": np.array(spk)}
    spk, spk_h = spks
    return res | {"spk": np.array(spk), "spk_h": np.array(spk_h)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chain")
    parser.add_argument("-o", "--output")
    parser.add_argument("-n", "--n-samples", type=int, default=2000)
    parser.add_argument("--do-kh", action="store_true")
    parser.add_argument("--max-inner-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    chain = chains.load_cosmosis_chain(args.chain)
    if (output := args.output) is None:
        output = path.basename(args.chain).split('.')[0] + "_spk.npz"
    if path.isfile(output) and not args.overwrite:
        print(f"output file {output} already exists")
        sys.exit(1)
    
    res = calc_sk_samples(chain, samples=args.n_samples, do_kh=args.do_kh, max_inner_threads=args.max_inner_threads)
    print("saving to", output)
    np.savez(output, **res)


if __name__ == "__main__":
    main()
