import numpy as np
import getdist


def get_latex_labels(param_names):
    """Get LaTex labels for cosmosis parameter names."""
    latex = dict(tau="$\\tau$", n_s="$n_s$", omega_m="$\\Omega_m$", omega_b="$\\Omega_b$", h0="$h$",
                 a_s_1e9="$10^9 A_s$", a_lens="$A_\\text{lens}$", sigma_8="$\\sigma_8$", s_8="$S_8$",
                 a_mod="$A_\\text{mod}$", logt_agn="$\\log T_\\text{AGN}$",
                 a1="$A_1$", a2="A_2", alpha1="$\\alpha_1$", alpha2="$\\alpha_2$", bias_ta="$b_\\text{ta}$")

    return map(lambda p: latex.get(p, p), param_names)


def load_cosmosis_chain(fname, label=None):
    """Load a cosmosis chain from a file."""
    with open(fname) as f:
        params = f.readline().lstrip("#").lower().split()
        params = [p if '--' not in p else p.split('--')[1] for p in params]
        sampler = f.readline().split('=')[1].strip()
    print("sampler:", sampler)
    data = np.loadtxt(fname, comments='#')

    if sampler == "polychord":
        return load_cosmosis_polychord(params, data, label=label)
    elif sampler == "nautilus":
        return load_cosmosis_nautilus(params, data, label=label)
    elif sampler == "fisher":
        return load_cosmosis_fisher(params, data, label=label)
    else:
        raise NotImplementedError


def load_cosmosis_polychord(params, data, label=None):
    print(len(data), "samples")
    samples = getdist.MCSamples(samples=data[:,:-4], weights=data[:,-1],
                                names=params[:-4], labels=get_latex_labels(params[:-4]),
                                label=label, sampler="nested")
    return samples


def load_cosmosis_nautilus(params, data, label=None):
    print(len(data), "samples")
    samples = getdist.MCSamples(samples=data[:,:-3], weights=np.exp(data[:,-3]),
                                names=params[:-3], labels=get_latex_labels(params[:-3]),
                                label=label, sampler="nested")
    return samples


def load_cosmosis_fisher(params, data, label=None):
    pass
