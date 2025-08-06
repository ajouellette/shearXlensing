import numpy as np
import h5py
from nautilus.bounds import NautilusBound
import getdist
from getdist.gaussian_mixtures import GaussianND


def get_latex_labels(param_names):
    """Get LaTex labels for cosmosis parameter names."""
    # cosmology
    latex = dict(omega_m="\\Omega_m", omega_b="\\Omega_b", omega_nu="\\Omega_\\nu", omega_k="\\Omega_k",
                 ombh2="\\omega_b", ommh2="\\omega_m", omch2="\\omega_c", omnuh2="\\omega_\\nu",
                 h0="h", hubble="H_0", cosmomc_theta="100\\theta_\\text{MC}", n_s="n_s", tau="\\tau",
                 a_s="A_s", log1e10as="\\ln \\left(10^{10} A_s\\right)", a_s_1e9="10^9 A_s",
                 sigma_8="\\sigma_8", s_8="S_8", wa="w_a", mnu="m_\\nu")
    # intrinsic alignment
    latex = latex | dict(a1="A_1", a2="A_2", alpha1="\\alpha_1", alpha2="\\alpha_2", bias_ta="b_\\text{ta}")
    # nusiance parameters
    latex = latex | {f"m{i}": f"m_{i}" for i in range(10)}
    latex = latex | {f"bias_{i}": f"\\Delta z_{i}" for i in range(10)}
    # miscellaneous
    latex = latex | dict(a_lens="A_\\text{lens}", a_mod="A_\\text{mod}", logt_agn="\\log T_\\text{AGN}",
                         a_planck="A_\\text{planck}")
    
    if isinstance(param_names, str):
        return latex.get(param_names, param_names)

    return map(lambda p: latex.get(p, p), param_names)


def load_cosmosis_chain(fname, label=None, quiet=False):
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
    if not quiet:
        print("sampler:", sampler)

    data = np.atleast_2d(np.loadtxt(fname, comments='#'))
    remove_inds = []
    # remove duplicated parameters
    for param, inds in p_inds.items():
        if len(inds) > 1:
            is_equal = np.all([np.isclose(data[:,inds[0]], data[:,i]).all() for i in inds[1:]])
            if not is_equal:
                raise RuntimeError(f"found different parameters with the same name: {param}")
            remove_inds += inds[1:]
    mask = np.ones(len(params), dtype=bool)
    mask[remove_inds] = False
    data = data[:,mask]
    params = [p for i, p in enumerate(params) if i not in remove_inds]

    chain = samplers[sampler](params, data, header=header, label=label)
    if isinstance(chain, getdist.MCSamples) and not quiet:
        print(chain.getNumSampleSummaryText())

    # try to add derived parameters
    if isinstance(chain, getdist.MCSamples):
        param_list = chain.paramNames.list()
        # likelihoood values
        if "like" not in param_list and "post" in param_list and "prior" in param_list:
            like = chain.getParams().post - chain.getParams().prior
            chain.addDerived(like, "like")
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
        # find allowed ranges of varied parameters
        ranges = {}
        in_values_section = False
        for line in header:
            if "START_OF_VALUES_INI" in line:
                in_values_section = True
                continue
            if in_values_section and '=' in line:
                param = line.split('=')[0].lstrip('#').strip()
                if param in varied_params:
                    values = list(map(float, line.split('=')[1].split()))
                    ranges[param] = [values[0], values[-1]]
            if "END_OF_VALUES_INI" in line:
                break
    else:
        varied_params = params
        derived_params = []
        ranges = None

    samples = getdist.MCSamples(samples=data, weights=weights, ranges=ranges,
                                names=varied_params + derived_params,
                                labels=get_latex_labels(params),
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


def load_maxlike(params, data, header=None, label=None):
    data = np.squeeze(data)
    result = {p: data[i] for i, p in enumerate(params)}
    return result


samplers = {
        "nautilus": load_nested,
        "polychord": load_nested,
        "fisher": load_fisher,
        "maxlike": load_maxlike
    }


def shell_bound_occupation(nautilus_file, fractional=True):
    """Determine how many points of each shell are also part of each bound.
    Parameters
    ----------
    fractional : bool, optional
        Whether to return the absolute or fractional dependence. Default
        is True.
    Returns
    -------
    m : numpy.ndarray
        Two-dimensional array with occupation numbers. The element at index
        :math:`(i, j)` corresponds to the occupation of points in shell
        shell :math:`i` that also belong to bound :math:`j`. If
        `fractional` is True, this is the fraction of all points in shell
        :math:`i` and otherwise it is the absolute number.
    """
    bounds = []
    points = []
    with h5py.File(nautilus_file) as s:
        i = 1
        while f"bound_{i}" in s.keys():
            bounds.append(NautilusBound.read(s[f"bound_{i}"]))
            points.append(s[f"sampler/points_{i}"][:])
            i += 1

    m = np.zeros((len(bounds), len(bounds)), dtype=int)
    for i, pnts in enumerate(points):
        for k, bound in enumerate(bounds):
            m[i, k] = np.sum(bound.contains(pnts))
    if fractional:
        m = m / np.diag(m)[:, np.newaxis]
    return m
