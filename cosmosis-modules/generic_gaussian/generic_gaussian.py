from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np


class GenericGaussian(GaussianLikelihood):
    like_name = "gaussian_params"

    def __init__(self, options):
        self.param_names = options.get_string("param_names").split()
        print("Will use parameters:", self.param_names)
        super().__init__(options)

    def build_data(self):
        if self.options.has_value("mean"):
            mean = self.options.get_double_array_1d("mean")
        print(mean)
        assert len(mean) == len(self.param_names)
        return None, mean

    def build_covariance(self):
        cov_file = self.options.get_string("cov_file")
        cov = np.genfromtxt(cov_file)
        assert cov.shape == 2*(len(self.param_names),)
        return cov

    def extract_theory_points(self, block):
        theory = []
        for p in self.param_names:
            section, name = p.split('/')
            if section == "cosmological_parameters":
                if not block.has_value(section, "a_s_1e9"):
                    block[section, "a_s_1e9"] = 1e9 * block[section, "a_s"]
            theory.append(block.get_double(section, name))
        return np.array(theory)


setup, execute, cleanup = GenericGaussian.build_module()
