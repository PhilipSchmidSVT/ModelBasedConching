"""
Library for differentiation of functions in context of
parameter sensitivity studies
"""
import numpy as np


class Differentiator:
    "Base class for differentiation"

    def __init__(self, gen_fun, **kwargs):
        self.gen_fun = gen_fun
        self.kwargs = kwargs

    def evaluate_function_at_time(self, params, t):
        res = self.gen_fun(t, self.kwargs["N"], params, self.kwargs["K"])
        return res

    def evaluate_function_at_times(self, params, t):
        res = list(
            map(
                lambda i_t: self.gen_fun(
                    [i_t], self.kwargs["N"], params, self.kwargs["K"]
                ),
                t,
            )
        )
        return np.array(res).squeeze()

    def plot_sensitivity_over_time(self, t_span, params, h=1e-16, t_lines=None):
        import matplotlib.pyplot as plt

        t_sampling = np.linspace(min(t_span), max(t_span), 100)
        sensitivity = self.get_parameter_sensitivity_at_times(t_sampling, params, h)
        n_species = sensitivity.shape[2]
        fig, ax = plt.subplots(1, n_species)
        for i_par in range(0, n_species):
            ax[i_par].plot(t_sampling, sensitivity[:, i_par, :])
            if t_lines is not None:
                [ax[i_par].axvline(t) for t in t_lines]
            ax[i_par].set_title(f"Par #{i_par}")
            ax[i_par].set_xlabel("time in h")
            ax[i_par].set_ylabel(r"$dc_{i}/dp$")
        fig.tight_layout()
        return fig


class Centered_Difference(Differentiator):
    "Do centered finite differences to calculate parameter sensitivities"

    def __init__(self, gen_fun, **kwargs):
        super().__init__(gen_fun, **kwargs)

    def perturb_element(self, params, i_par, h) -> tuple:
        param_low = params[i_par] - h
        param_high = params[i_par] + h
        params_low = params.copy()
        params_high = params.copy()
        params_low[i_par] = param_low
        params_high[i_par] = param_high
        return (params_low, params_high)

    def perturb_param_set(self, params, h) -> list:
        perturbed_params = [
            self.perturb_element(params, i_par, h) for i_par in range(0, len(params))
        ]
        return perturbed_params

    def get_parameter_sensitivity_at(self, t, params, h):
        # Get set of perturbed parameters
        perturbed_params = self.perturb_param_set(params, h)
        # Get finite difference of parsets
        sols = [
            [self.evaluate_function_at_time(par, t) for par in pardouble]
            for pardouble in perturbed_params
        ]
        delta_sols = list(map(lambda i_set: (i_set[1] - i_set[0]) / (2 * h), sols))
        return np.array(delta_sols)

    def get_parameter_sensitivity_at_times(self, t, params, h):
        sensitivity = list(
            map(lambda i_t: self.get_parameter_sensitivity_at([i_t], params, h), t)
        )
        return np.array(sensitivity).squeeze()


class Complex_Step_Differentiation(Differentiator):
    def __init__(self, gen_fun, **kwargs):
        super().__init__(gen_fun, **kwargs)

    def perturb_element(self, params, i_par, h):
        params[i_par] = complex(params[i_par], h)
        return params

    def perturb_param_set(self, params, h):
        perturbed_params = [
            self.perturb_element(params.copy(), i_par, h)
            for i_par in range(0, len(params))
        ]
        return perturbed_params

    def get_parameter_sensitivity_at(self, t, params, h):
        """
        Compute the complex step derivative of gen_fun at time t
        using stepsize h.
        Ref:
        blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/
        dFdp = Im(F(p0 + ih))/h + O(h**2)
        """
        perturbed_params = self.perturb_param_set(params, h)
        sol_ph = list(
            map(lambda p: self.evaluate_function_at_time(p, t), perturbed_params)
        )
        dFdp = [sol.imag / h for sol in sol_ph]
        return np.array(dFdp)

    def get_parameter_sensitivity_at_times(self, t, params, h):
        sensitivities = list(
            map(lambda i_t: self.get_parameter_sensitivity_at([i_t], params, h), t)
        )
        # Output is time x parameter x species
        return np.array(sensitivities).squeeze()


if __name__ == "__main__":
    pass
