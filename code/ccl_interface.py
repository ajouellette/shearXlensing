import sys
import yaml
import numpy as np
from scipy import interpolate
import fitsio
import healpy as hp
import pyccl as ccl
import nx2pt


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
    return cosmo.CMBLensingTracer(z_source=args["z_source"])
    
    
tracer_types = {
        "CMBLensing": get_cmb_lensing_tracer,
        "WeakLensing": get_weak_lensing_tracer
}


def load_tracer():
    pass


class CCLHaloModel:
    defaults = {
            "mdef": ccl.halos.MassDef200c,
            "hmf": ccl.halos.MassFuncTinker10,
            "hbias": ccl.halos.HaloBiasTinker10,
            "conc": ccl.halos.ConcentrationDuffy08,
        }

    k_arr = np.logspace(-4, 2, 150)
    lk_arr = np.log(k_arr)
    a_arr = np.linspace(0.01, 1, 25)

    suppress_1h = lambda a: 0.1
    smooth_transition = lambda a: 0.7

    def __init__(self, cosmo, mdef=None, hmf=None, hbias=None, conc=None):
        self.cosmo = cosmo
        self.mdef = self.defaults["mdef"] if mdef is None else mdef
        self.hmf = (self.defaults["hmf"] if hmf is None else hmf)(mass_def=self.mdef)
        self.hbias = (self.defaults["hbias"] if hbias is None else hbias)(mass_def=self.mdef)
        self.conc = (self.defaults["conc"] if conc is None else conc)(mass_def=self.mdef)

        self.hmc = ccl.halos.HMCalculator(mass_function=self.hmf, halo_bias=self.hbias, mass_def=self.mdef)
        
        # TODO: these quantities should be tracer dependent
        # this current setup works for lensing only tracers, but will need to change to support galaxies
        self.nfw = ccl.halos.HaloProfileNFW(mass_def=self.mdef, concentration=self.conc)
        self.pMM = ccl.halos.Profile2pt()

        self.tk_1h = ccl.halos.pk_4pt.halomod_Tk3D_1h(self.cosmo, self.hmc, self.nfw,
                                                      lk_arr=self.lk_arr, a_arr=self.a_arr)
        self.tk_ssc = ccl.halos.pk_4pt.halomod_Tk3D_SSC(cosmo=self.cosmo, hmc=self.hmc, prof=self.nfw,
                                                        lk_arr=self.lk_arr, a_arr=self.a_arr)


class CCLTheory:

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
            mdef = config["halomodel"].get("mdef", None)
            hmf = config["halomodel"].get("hmf", None)
            hbias = config["halomodel"].get("hbias", None)
        else:
            mdef = hmf = hbias = None

        self.hm = CCLHaloModel(self.cosmo, mdef, hmf, hbias)

        # load tracers
        self.tracers = config["tracers"]
        for trn, tracer in self.tracers.items():
            if tracer["type"] not in tracer_types.keys():
                raise ValueError(f"unknown tracer type for {trn}")
            ccl_tracer = tracer_types[tracer["type"]](self.cosmo, tracer["args"])
            mask_file = tracer["args"].get("sky_mask", None)
            if mask_file is not None:
                tracer["sky_mask"] = hp.read_map(mask_file).astype(bool)
            tracer["ccl_tracer"] = ccl_tracer
            
    def get_cl(self, tracer1, tracer2, ell, use_hm=False, bpws=None):
        """Compute the cross-spectrum between tracer1 and tracer2."""
        tr1 = self.tracers[tracer1]["ccl_tracer"]
        tr2 = self.tracers[tracer2]["ccl_tracer"]
        m1 = self.tracers[tracer1]["args"].get("m_bias", 0)
        m2 = self.tracers[tracer2]["args"].get("m_bias", 0)
        cl = (1 + m1) * (1 + m2) * self.cosmo.angular_cl(tr1, tr2, ell)
        if bpws is not None:
            cl = nx2pt.bin_theory_cl(cl, bpws)
        return cl

    def get_cov_marg_m(self, tracer1a, tracer2a, tracer1b, tracer2b, bpws, bpws_b=None):
        """Compute the extra covariance term due to analytically marginalizing over multiplicative bias."""
        sigma_m1a = self.tracers[tracer1a]["args"].get("sigma_m", 0)
        sigma_m2a = self.tracers[tracer2a]["args"].get("sigma_m", 0)
        sigma_m1b = self.tracers[tracer1b]["args"].get("sigma_m", 0)
        sigma_m2b = self.tracers[tracer2b]["args"].get("sigma_m", 0)

        prior_factor = sigma_m1a * sigma_m1b + sigma_m1a * sigma_m2b + \
                       sigma_m2a * sigma_m1b + sigma_m2a * sigma_m2b

        if bpws_b is None: bpws_b = bpws
        ell = np.arange(max(bpws.shape[1], bpws_b.shape[1]))

        cl_a = self.get_cl(tracer1a, tracer2a, ell, bpws=bpws)
        cl_b = self.get_cl(tracer1b, tracer2b, ell, bpws=bpws_b)

        cov = prior_factor * np.outer(cl_a, cl_b)
        return cov

    def get_fsky(self, tracer1a, tracer2a=None, tracer1b=None, tracer2b=None):
        if tracer2a is None: tracer2a = tracer1a
        if tracer1b is None: tracer1b = tracer1a
        if tracer2b is None: tracer2b = tracer2a
        masks = [self.tracers[tr]["sky_mask"]
                 for tr in [tracer1a, tracer2a, tracer1b, tracer2b]]
        return np.mean(masks[0] * masks[1] * masks[2] * masks[3])

    def get_cov_ssc(self, tracer1a, tracer2a, ells_a, tracer1b=None, tracer2b=None, ells_b=None):
        """Compute the super-sample covariance term."""
        fsky = self.get_fsky(tracer1a, tracer2a, tracer1b, tracer2b)
        sigma2_B = self.cosmo.sigma2_B_disc(fsky=fsky)
        if tracer1b is None: tracer1b = tracer1a
        if tracer2b is None: tracer2b = tracer2a
        tracers = [self.tracers[tr]["ccl_tracer"]
                   for tr in [tracer1a, tracer2a, tracer1b, tracer2b]]
        cov_ssc = self.cosmo.angular_cl_cov_SSC(tracer1=tracers[0], tracer2=tracers[1],
                    tracer3=tracers[2], tracer4=tracers[3], ell=ells_a, ell2=ells_b,
                    t_of_kk_a=self.hm.tk_ssc, sigma2_B=sigma2_B,
                    integration_method="spline").T
        return cov_ssc

    def get_cov_cng(self, tracer1a, tracer2a, ells_a, tracer1b=None, tracer2b=None, ells_b=None):
        """Compte the connected non-Gaussian covariance term."""
        fsky = self.get_fsky(tracer1a, tracer2a, tracer1b, tracer2b)
        if tracer1b is None: tracer1b = tracer1a
        if tracer2b is None: tracer2b = tracer2a
        tracers = [self.tracers[tr]["ccl_tracer"]
                   for tr in [tracer1a, tracer2a, tracer1b, tracer2b]]
        cov_ng = self.cosmo.angular_cl_cov_cNG(tracer1=tracers[0], tracer2=tracers[1],
                    tracer3=tracers[2], tracer4=tracers[3], ell=ells_a, ell2=ells_b,
                    t_of_kk_a=self.hm.tk_1h, fsky=fsky,
                    integration_method="spline").T
        return cov_ng
        

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    theory = CCLTheory(config)
    print(theory.tracers)
