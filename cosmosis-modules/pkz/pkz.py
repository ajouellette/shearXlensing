import numpy as np
from scipy import interpolate
from cosmosis.datablock import names, option_section


def setup(options):
    pass


def execute(block, config):
    # get current power spectrum
    pk = block["matter_power_nl", "p_k"]
    z = block["matter_power_nl", "z"]
    k = block["matter_power_nl", "k_h"]

    # get sampled parameters
    kp = block["pkz", "k_p"]
    z_p = block["pkz", "z_p"]
    amp = block["pkz", "a"]
    alpha = block["pkz", "alpha"]
    beta = block["pkz", "beta"]

    # compute new power spectrum
    pk *= amp * (k / kp)**alpha * (z / zp)**beta
    block["matter_power_nl"] = pk

    return 0
