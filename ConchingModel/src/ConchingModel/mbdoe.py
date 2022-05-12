"""
Library for applying model based design of experiments to the conching
problem. A use case is optimising the choice of sampling times for higher
precision of parameter estimations.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from ConchingModel import ConcheClass, DifferentiationModules, OptimizerClass
from ConchingModel.Data import SimData
from scipy import optimize


class ODOE:
    "Base class for Optimized Design of Experiments"

    def __init__(
        self, params: list, diff_module: DifferentiationModules.Differentiator
    ) -> None:
        """
        Initialization of a ODOE object
        Arguments:
            params [float] -> mass transport coefficients.
            diff_module Differentiatr -> Differentiation module to use.
        Returns:
            None
        """

        self.params = params
        self.diff_module = diff_module

    @staticmethod
    def get_d_optimality(var_covar_inv):
        "Objective function is the trace of the variance_covariance matrix"
        d_optimality = np.linalg.det(var_covar_inv)
        return d_optimality

    @staticmethod
    def get_e_optimality(var_covar_inv):
        "Objective function is the eigenvalues of the variance_covariance matrix"
        e_optimality = np.linalg.eigvals(var_covar_inv)
        return max(e_optimality)

    @staticmethod
    def get_a_optimality(var_covar_inv):
        "Objective function is the trace of the variance_covariance matrix"
        a_optimality = np.trace(var_covar_inv)
        return a_optimality

    def get_var_covar_mat(self, t_s: list, h: float) -> np.array:
        """
        Calculate the variance covariance matrix of the parameter values
        at times t_s
        Arguments:
            t_s [float] -> list of sampling times != 0
            h float -> stepsize
        Returns:
            var_covar_mat -> nd.array: The variance covariance matrix of
                                         parameters at t_s.
        """
        sensitivity = self.diff_module.get_parameter_sensitivity_at_times(
            t_s, self.params, h
        )
        # Equalize sensitivity format to julialike
        n_phases = sensitivity.shape[2]
        sens_col = [sensitivity.T[i_phase] for i_phase in range(0, n_phases)]
        sens_row = [i_phase.T for i_phase in sens_col]
        QrTQr = list(itertools.product(sens_row, sens_col))
        dot_product = [y @ x for x, y in QrTQr]
        var_covar = np.array(dot_product).sum(axis=0)
        return var_covar

    def get_inv_var_covar(self, var_covar: np.array) -> np.array:
        """Get the inverse of the var_covar matrix"""
        try:
            var_covar_inv = np.linalg.inv(var_covar)
        except np.linalg.LinAlgError:
            var_covar_inv = np.linalg.pinv(var_covar)
        return var_covar_inv

    def run_forward_sim(self, meta_info, mtc, t, N0, K) -> SimData.SimData:
        "Run a fast forward simulation of the Conche system and save output as SimData"
        conche = ConcheClass.ConcheClass(meta_info)
        sim_out = conche.run_sim(t, N0, mtc, K)
        data_out = SimData.SimData(sim_out, t, conche=conche)
        return data_out

    def run_parameter_estim(self, data: SimData.SimData, per_dev_rel, bounds, K):
        "Estimate parameters on perturbed data set"
        perturbed_data = data.get_perturbed_data(per_dev_rel)
        data.conche.meta_data.timeseries = perturbed_data
        opt = OptimizerClass.OptimizerClass(data.conche.meta_data)
        opt_out = opt.run_local_optimization(bounds, np.mean(bounds, axis=1), K)
        return opt_out

    def plot_comparison_par_set(self, param_set: list):
        """param_sets are estimated parameters from different
        experimental conditions"""
        assert isinstance(param_set, (list, tuple))
        par = np.array(param_set)
        figs = []
        for ipar in range(0, par.shape[2]):
            fig, ax = plt.subplots()
            for iset in par[:, :, ipar]:
                ax.hist(iset, alpha=0.5, bins=50)
            ax.set_title(f"Distribution of Parameter #{ipar}")
            ax.set_xlabel("Parameter value")
            ax.set_ylabel("#")
            fig.tight_layout()
            figs.append(fig)
        return figs


class SamplingTime(ODOE):
    """
    Class interface for conducting the optimal experimental design studies
    """

    def cost_fun_t(self, t_s, h, score_fun):
        "Minimize cost function by adjusting sampling times"
        var_covar = self.get_var_covar_mat(t_s, h)
        var_covar_inv = self.get_inv_var_covar(var_covar)
        score = score_fun(var_covar_inv)
        return score

    def optimize_samplingtimes(
        self,
        score_fun,
        h: float,
        t_init: np.ndarray,
        min_diff=0.1,
        method="differential_evolution",
    ) -> optimize.OptimizeResult:
        """
        Get the optimized sampling times for a given set of mass transport
        coefficients and conche setup
        """

        def cb(xk):
            out = self.cost_fun_t(xk, h, score_fun)
            print(f"current score: {out}")
            return False

        # Bounds section
        const_min_dt = self.construct_ascending_order_constraint(t_init, min_diff)
        bounds = [(0.01, max(t_init) * 1.01) for _ in range(0, len(t_init))]
        if method == "differential_evolution":
            opt_res = optimize.differential_evolution(
                self.cost_fun_t,
                bounds=bounds,
                args=(h, score_fun),
                constraints=(const_min_dt),
                workers=1,
            )
        elif method == "shgo":
            opt_res = optimize.shgo(self.cost_fun_t, bounds=bounds, args=(h, score_fun))
        elif method == "dual_annealing":
            opt_res = optimize.dual_annealing(
                self.cost_fun_t, bounds, args=(h, score_fun)
            )
        elif method == "brute":
            opt_res = optimize.brute(self.cost_fun_t, bounds, args=(h, score_fun))
        return opt_res

    def construct_ascending_order_constraint(
        self, t_init, min_diff: float
    ) -> optimize.LinearConstraint:
        "Constraint that ensures only ascending order in results."
        const_A = np.eye(len(t_init)) - np.eye(len(t_init), k=-1)
        min_dt_lb = np.ones(len(t_init)) * min_diff
        min_dt_lb[0] = 0
        min_dt_ub = np.ones(len(t_init)) * np.inf
        const_min_dt = optimize.LinearConstraint(
            const_A, min_dt_lb, min_dt_ub, keep_feasible=True
        )
        return const_min_dt

    def optimize_samplingtimes_locally(self, score_fun, h, t_init, min_diff=0.1):
        "Do sampling_time optimization locally"
        opti_options = {"xtol": 1e-32}
        bounds = [(2 * h, max(t_init)) for _ in range(0, len(t_init))]
        const_min_dt = self.construct_ascending_order_constraint(t_init, min_diff)
        opt_res = optimize.minimize(
            self.cost_fun_t,
            t_init,
            args=(h, score_fun),
            bounds=bounds,
            options=opti_options,
            constraints=(const_min_dt),
            method="trust-constr",
        )
        return opt_res


class deltaSamplingTime(ODOE):
    "Estimate optimal sampling times by optimizing for the time interval between samplings"

    def cost_fun_dt(self, dt, h, score_fun):
        "Optimize for sampling time intervals"
        t = np.cumsum(dt)
        var_covar = self.get_var_covar_mat(t, h)
        var_covar_inv = self.get_inv_var_covar(var_covar)
        score = score_fun(var_covar_inv)
        return score

    def optimize_samplingtimes(self, score_fun, h, dt_init, min_diff=0.1):
        "Carry out optimization of sampling time intervals."
        max_t = np.max(np.cumsum(dt_init))
        n_t = len(dt_init)
        bounds = [(min_diff, max_t / n_t) for i_t in range(0, n_t)]
        opt_res = optimize.differential_evolution(
            self.cost_fun_dt, bounds, args=(h, score_fun)
        )
        return opt_res


if __name__ == "__main__":
    pass
