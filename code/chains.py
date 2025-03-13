import numpy as np
import getdist


def get_latex_labels(param_names):
    """Get LaTex labels for cosmosis parameter names."""
    # cosmology
    latex = dict(tau="$\\tau$", n_s="$n_s$", omega_m="$\\Omega_m$", omega_b="$\\Omega_b$", h0="$h$",
                 ombh2="$\\omega_b$", ommh2="$\\omega_m$", omch2="$\\omega_c$",
                 a_s="$A_s$", log1e10as="$\\ln \\left(10^{10} A_s\\right)$", a_s_1e9="$10^9 A_s$",
                 sigma_8="$\\sigma_8$", s_8="$S_8$", w="$w$", wa="$w_a$")
    # intrinsic alignment
    latex = latex | dict(a1="$A_1$", a2="A_2", alpha1="$\\alpha_1$", alpha2="$\\alpha_2$", bias_ta="$b_\\text{ta}$")
    # nusiance parameters
    latex = latex | {f"m{i}": f"$m_{i}$" for i in range(10)}
    latex = latex | {f"bias_{i}": f"$\\Delta z_{i}$" for i in range(10)}
    # miscellaneous
    latex = latex | dict(a_lens="$A_\\text{lens}$", a_mod="$A_\\text{mod}$", logt_agn="$\\log T_\\text{AGN}$")

    return map(lambda p: latex.get(p, p), param_names)


def load_cosmosis_chain(fname, label=None):
    """Load a cosmosis chain from a file."""
    # read the header
    header = []
    with open(fname) as f:
        while True:
            line = f.readline()
            if not line or not line.startswith('#'):
                break
            else:
                header.append(line.strip())
    # cosmosis parameter names are case-insensitive
    params = header.pop(0).lstrip("#").lower().split()  
    # for simplicity, get rid of section names
    # assumes that all parameter names are unique
    params = [p if '--' not in p else p.split('--')[1] for p in params]

    sampler = header.pop(1).split('=')[1].strip()
    if sampler not in samplers.keys():
        raise NotImplementedError(f"unknown sampler {sampler}")
    print("sampler:", sampler)

    data = np.loadtxt(fname, comments='#')

    return samplers[sampler](header, params, data, label=label)


def load_cosmosis_polychord(header, params, data, label=None):
    print(len(data), "samples")
    samples = getdist.MCSamples(samples=data[:,:-4], weights=data[:,-1],
                                names=params[:-4], labels=get_latex_labels(params[:-4]),
                                label=label, sampler="nested")
    return samples


def load_cosmosis_nautilus(header, params, data, label=None):
    print(len(data), "samples")
    samples = getdist.MCSamples(samples=data[:,:-3], weights=np.exp(data[:,-3]),
                                names=params[:-3], labels=get_latex_labels(params[:-3]),
                                label=label, sampler="nested")
    return samples


def load_cosmosis_fisher(header, params, data, label=None):
    pass


samplers = {
        "nautilus": load_cosmosis_nautilus,
        "polychord": load_cosmosis_polychord,
        "fisher": load_cosmosis_fisher
    }
