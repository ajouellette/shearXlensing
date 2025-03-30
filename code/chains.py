import numpy as np
import getdist
from getdist.gaussian_mixtures import GaussianND


def get_latex_labels(param_names):
    """Get LaTex labels for cosmosis parameter names."""
    # cosmology
    latex = dict(tau="\\tau", n_s="n_s", omega_m="\\Omega_m", omega_b="\\Omega_b", h0="h",
                 ombh2="\\omega_b", ommh2="\\omega_m", omch2="\\omega_c",
                 a_s="A_s", log1e10as="\\ln \\left(10^{10} A_s\\right)", a_s_1e9="10^9 A_s",
                 sigma_8="\\sigma_8", s_8="S_8", wa="w_a")
    # intrinsic alignment
    latex = latex | dict(a1="A_1", a2="A_2", alpha1="\\alpha_1", alpha2="\\alpha_2", bias_ta="b_\\text{ta}")
    # nusiance parameters
    latex = latex | {f"m{i}": f"m_{i}" for i in range(10)}
    latex = latex | {f"bias_{i}": f"\\Delta z_{i}" for i in range(10)}
    # miscellaneous
    latex = latex | dict(a_lens="A_\\text{lens}", a_mod="A_\\text{mod}", logt_agn="\\log T_\\text{AGN}")
    
    if isinstance(param_names, str):
        return latex.get(param_names, param_names)

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
    # get all occurances of each parameter to check for duplicates
    p_inds = {param: [i for i, p in enumerate(params) if param == p] for param in set(params)}

    sampler = header.pop(0).split('=')[1].strip()
    if sampler not in samplers.keys():
        raise NotImplementedError(f"unknown sampler {sampler}")
    print("sampler:", sampler)

    data = np.loadtxt(fname, comments='#')
    remove_inds = []
    # remove duplicated parameters
    for param, inds in p_inds.items():
        if len(inds) > 1:
            is_equal = np.all([np.isclose(data[:,inds[0]], data[:,i]).all() for i in inds[1:]])
            if not is_equal:
                raise RuntimeError(f"found different parameters with the same name: {param}")
            remove_inds += inds[1:]
    print(remove_inds)
    mask = np.ones(len(params), dtype=bool)
    mask[remove_inds] = False
    data = data[:,mask]
    params = [p for i, p in enumerate(params) if i not in remove_inds]

    print(len(params), data.shape)

    chain = samplers[sampler](params, data, header=header, label=label)
    # try to add derived parameters
    if isinstance(chain, getdist.MCSamples):
        param_list = chain.paramNames.list()
        # get S8
        if "omega_m" in param_list and "sigma_8" in param_list and "s_8" not in param_list:
            s_8 = chain.getParams().sigma_8 * np.sqrt(chain.getParams().omega_m/0.3)
            chain.addDerived(s_8, "s_8", label=get_latex_labels("s_8"))
    return chain


def load_nested(params, data, header=None, label=None):
    # find weights column
    ind = -1
    for key in ["weight", "log_weight"]:
        if key in params:
            ind = params.index(key)
            break
    if ind != -1:
        params.pop(ind)
        weights = data[:,ind] if 'log' not in key else np.exp(data[:,ind])
        data = np.delete(data, ind, axis=1)
    else:
        print("warning: did not find sample weights, using equal weights")
        weights = np.ones(len(data))

    if header is not None:
        n_varied = None
        for line in header:
            if "n_varied" in line:
                n_varied = int(line.split('=')[1])
                break
        varied_params = params[:n_varied]
        # mark derived params
        derived_params = [p+'*' for p in params[n_varied:]]
    else:
        varied_params = params
        derived_params = []

    samples = getdist.MCSamples(samples=data, weights=weights,
                                names=varied_params + derived_params, labels=get_latex_labels(params),
                                label=label, sampler="nested")
    return samples


def load_fisher(params, data, header=None, label=None):
    mean = np.zeros(len(params))
    if header is not None:
        # get mean vector from header
        for i, line in enumerate(header[-len(params):]):
            mean[i] = float(line.split('=')[1])
    # parameter covariance
    pcov = np.linalg.inv(data)
    gaussian = GaussianND(mean, pcov, names=params, labels=get_latex_labels(params),
                          label=label)
    return gaussian


samplers = {
        "nautilus": load_nested,
        "polychord": load_nested,
        "fisher": load_fisher
    }
