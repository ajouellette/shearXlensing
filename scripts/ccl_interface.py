import sys
import yaml
import numpy as np
from scipy import interpolate
import fitsio
import pyccl as ccl


def get_cosmo(params):
    return ccl.Cosmology(**params)
    

def get_dndz_fits(fits_file, section):
    data = fitsio.FITS(fits_file)
    name, bin_name = section.split()
    z = data[name]["Z_MID"][:]
    dndz = data[name][bin_name][:]
    data.close()
    return z, dndz


def apply_photoz_bias(dndz, delta_z):
    dndz_f = interpolate.interp1d(*dndz, kind="cubic", fill_value=0, bounds_error=False)
    z = dndz[0]
    dndz_new = dndz_f(z - delta_z)
    return (z, dndz_new)


def get_ia_parameterization(args):

    def get_nla(params):
        return lambda z: params["A"]

    def get_nla_z(params):
        A, eta, z_piv = params["A"], params["eta"], params["z_piv"]
        return lambda z: A * ((1 + z) / (1 + z_piv))**eta

    ia_types = {"nla": get_nla, "nla-z": get_nla_z}

    if args["kind"] not in ia_types.keys():
        raise ValueError("unrecognized IA type")

    return ia_types[args["kind"]](args["params"])


def get_weak_lensing_tracer(cosmo, args, ia=None):
    file_name = args["dndz"]["file"]
    if file_name.endswith(".fits"):
        dndz = get_dndz_fits(file_name, args["dndz"]["section"])
    else:
        raise NotImplementedError
    delta_z = args.get("delta_z", 0)
    if delta_z != 0:
        dndz = apply_photoz_bias(dndz, delta_z)
    if ia is not None:
        z = dndz[0]
        ia = (z, ia(z))

    return cosmo.WeakLensingTracer(dndz=dndz, ia_bias=ia)


def get_cmb_lensing_tracer(cosmo, args):
    return cosmo.CMBLensingTracer(**args)
    
    
tracer_types = {
        "CMBLensing": get_cmb_lensing_tracer,
        "WeakLensing": get_weak_lensing_tracer
}


class CCLTheory:

    halomodel_defaults = {
            "mdef": ccl.halos.MassDef200c,
            "hmf": ccl.halos.MassFuncBocquet16,
            "hbias": ccl.halos.HaloBiasTinker10,
        }

    def __init__(self, config):
        self.config = config
        if "cosmo" in config.keys():
            self.cosmo = get_cosmo(config["cosmo"])
        else:
            self.cosmo = ccl.CosmologyVanillaLambdaCDM()

        if "ia" in config.keys():
            self.ia_z = get_ia_parameterization(config["ia"])
        else:
            self.ia_z = None

        if "halomodel" in config.keys():
            mdef = config["halomodel"].get("mdef", halomodel_defaults["mdef"])
            hmf = config["halomodel"].get("", halomodel_defaults["mdef"])
            hbias = config["halomodel"].get("mdef", halomodel_defaults["mdef"])

        self.tracers = config["tracers"]
        for trn, tracer in self.tracers.items():
            if tracer["type"] not in tracer_types.keys():
                raise ValueError(f"unknown tracer type for {trn}")
            ccl_tracer = tracer_types[tracer["type"]](self.cosmo, tracer["args"])
            tracer["ccl_tracer"] = ccl_tracer
            
    def get_cl(self, tracer1, tracer2, ell, use_hm=False):
        tr1 = self.tracers[tracer1]["ccl_tracer"]
        tr2 = self.tracers[tracer2]["ccl_tracer"]
        m1 = self.tracers[tracer1]["args"].get("m_bias", 0)
        m2 = self.tracers[tracer2]["args"].get("m_bias", 0)
        cl = (1 + m1) * (1 + m2) * self.cosmo.angular_cl(tr1, tr2, ell)
        return cl

    def get_cov_marg_m(self, tracer1a, tracer2a, tracer1b, tracer2b, ell):
        sigma_m1a = self.tracers[tracer1a]["args"].get("sigma_m", 0)
        sigma_m2a = self.tracers[tracer2a]["args"].get("sigma_m", 0)
        sigma_m1b = self.tracers[tracer1b]["args"].get("sigma_m", 0)
        sigma_m2b = self.tracers[tracer2b]["args"].get("sigma_m", 0)

        prior_factor = sigma_m1a * sigma_m2a + sigma_m1b * sigma_m2b

        cl_a = self.get_cl(tracer1a, tracer2a, ell)
        cl_b = self.get_cl(tracer1b, tracer2b, ell)

        cov = prior_factor * np.outer(cl_a, cl_b)
        return cov


    def get_cov_ssc(self, tracer1a, tracer2a, tracer1b, tracer2b):
        pass

    def get_cov_cng(self, tracer1a, tracer2a, tracer1b, tracer2b):
        pass


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    theory = CCLTheory(config)

    print(theory.tracers)
