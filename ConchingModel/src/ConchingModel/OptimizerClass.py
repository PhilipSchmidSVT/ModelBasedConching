"""
Class for the estimation of parameters of the conching model from data.
Model is fit to data by a least squares approach.
Available Classes are:
OptimizerClass: Baseclass for optimization. Optimizes for mass transport coeffs.
InitialStateOptimizer: Inherits OptimizerClass. Additionally optimizes for
  inital sampling time t=0.
PLOptim: Optimizer that handles optimization for the Profile Likelihood analysis
  optimization. Fixes a parameter to a specific value while others are allowed
  to vary.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from itertools import accumulate, pairwise

import ConchingModel.ConcheClass as ConcheClass
import ConchingModel.Data.Timeseries as Timeseries
import matplotlib.pyplot as plt
import numpy as np
from ConchingModel.Data.MetaData import aMetaData
from ConchingModel.GPR import GPRAnalysis, GPRPredictor, GPRVisualizer
from ConchingModel.ParameterEstimation.Targets import OptimizationTarget, TargetHandler
from ConchingModel.Visualizations.timeseries import tsVizErrorbar
from scipy import optimize
from scipy.optimize.optimize import OptimizeResult
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


@dataclass
class Optimizer:
    """Base class for parameter estimation."""

    meta_data: aMetaData = field(repr=False)
    exp_data: Timeseries.aTimeseries = field(repr=False)
    targets: TargetHandler = field(repr=False)
    conche: ConcheClass = field(init=False, repr=False)

    def __post_init__(self):
        self.conche = self.meta_data.spawn_conche()

    def select_properties(self):
        non_targets = {"mtc", "K", "c0"} - self.targets.var_name
        result = {}
        for item in non_targets:
            if item == "K":
                try:
                    result["K"] = self.meta_data.K
                except KeyError:
                    result["K"] = None
            elif item == "c0":
                try:
                    result["c0"] = self.meta_data.N0
                except KeyError:
                    result["c0"] = None
            else:
                pass
        return result

    def run_optimization(
        self, sampling_times: list[float], **fixed_parameters
    ) -> OptimizeResult:
        """
        Estimate parameters and initial condition by least squares method.

        Args:
            bounds (list[list[float]]): Bounds for mass transport coefficients,
              partition coefficients and initial state.

        Returns:
            OptimizeResult: Result of the parameter estimation.
        """
        result = optimize.least_squares(
            self.residuals,
            self.targets.mean_x0,
            bounds=self.targets.bounds_2_elements,
            args=(sampling_times,),
            xtol=None,
            ftol=None,
            max_nfev=1000,
            kwargs=fixed_parameters,
        )
        return result

    def slice_params(self, params: list) -> dict:
        """Assign the params handed to residuals to a dict with their respective labels."""
        par_indices = list(
            [0, *accumulate([target.len for target in self.targets.targets])]
        )
        start_and_end = list(pairwise(par_indices))
        sliced_params = [params[s:e] for s, e in start_and_end]
        result = {
            target.var_name: param
            for target, param in zip(self.targets.targets, sliced_params)
        }
        return result

    def residuals(
        self, parameters: list, sampling_times: list, **fixed_parameters
    ) -> np.ndarray:
        """
        Calculate the squared residuals normalized by the variance of timeseries data.

        Args:
            parameters (list): Parameters that were passed to construct simulation data.
            sampling_times (list): Sampling times of the simulated data.
            fixed_parameters (dict): Parameters that are not optimization targets but fixed.

        Returns:
            np.ndarray: Residuals between experimental data and simulated data.
        """
        parameters = (
            self.slice_params(parameters) | fixed_parameters | self.select_properties()
        )
        sim_data = self.conche.run_sim(
            sampling_times, parameters["c0"], parameters["mtc"], parameters["K"]
        )
        score = (
            (self.exp_data.c_mean - sim_data) ** 2
        ) / self.exp_data.c_var_for_residuals
        return np.sum(score, axis=1)

    def to_combined_particle_phase(self, data):
        """Single particle phases to combined particle phase."""
        substance_amount = data.T * self.meta_data.phase_mass
        particle_concentration = np.sum(substance_amount.T[1:, :], axis=0) / np.sum(
            self.meta_data.phase_mass[1:]
        )
        return np.array([data[0, :], particle_concentration])

    def plot_comparison(self, popt: list, ax=None, **fixed_parameters) -> plt.Figure:
        """Plot comparison between best estimate and experimental data."""
        if ax is None:
            fig, ax = plt.subplots()

        fig = tsVizErrorbar(self.exp_data)(ax=ax, fmt="o", prefix="Experiment: ")
        ts = self.sim_best_estimate(popt, fixed_parameters)
        fig = tsVizErrorbar(ts)(ax=ax, fmt="--", prefix="Simulation: ")
        return fig

    def sim_best_estimate(self, popt, fixed_parameters):
        "Sim parameter estimates with smooth time"
        params = (
            self.slice_params(popt.tolist())
            | fixed_parameters
            | self.select_properties()
        )
        smooth_time = np.linspace(
            min(self.meta_data.sampling_times), max(self.meta_data.sampling_times), 100
        )
        sim_data = self.conche.run_sim(
            smooth_time, params["c0"], params["mtc"], params["K"]
        )
        meta_copy = copy.deepcopy(self.meta_data)
        meta_copy.sampling_times = smooth_time.tolist()
        ts = Timeseries.aTimeseries.from_sim(meta_copy, sim_data)
        return ts


class GlobalOptimizer(Optimizer):
    "Run parameter calibration using global searches."

    def run_optimization(
        self, sampling_times: list[float], fixed_parameters=None
    ) -> OptimizeResult:
        bounds = list(zip(*self.targets.bounds_2_elements))
        result = optimize.differential_evolution(
            self.residuals,
            bounds,
            args=(sampling_times, fixed_parameters),
            workers=-1,
            updating="deferred",
            init="sobol",
            maxiter=1000,
        )
        return result

    def get_par_dict(self, parameters, fixed_parameters=None):
        if fixed_parameters is None:
            fixed_parameters = {}
        par_dict = (
            self.slice_params(parameters) | self.select_properties() | fixed_parameters
        )
        return par_dict

    def residuals(
        self, parameters: list, sampling_times: list, fixed_parameters=None
    ) -> np.ndarray:
        """
        Calculate the squared residuals normalized by the variance of timeseries data.

        Args:
            parameters (list): Parameters that were passed to construct simulation data.
            sampling_times (list): Sampling times of the simulated data.
            fixed_parameters (dict): Parameters that are not optimization targets but fixed.

        Returns:
            np.ndarray: Residuals between experimental data and simulated data.
        """
        par_dict = self.get_par_dict(parameters, fixed_parameters=fixed_parameters)
        sim_data = self.conche.run_sim(
            sampling_times, par_dict["c0"], par_dict["mtc"], par_dict["K"]
        )
        score = (
            (self.exp_data.c_mean - sim_data) ** 2
        ) / self.exp_data.c_var_for_residuals
        return np.sum(score)


@dataclass
class OptWithGPR(GlobalOptimizer):
    """
    This class does parameter estimation by using weights from a
    Gaussian Process Regression of the underlying data.
    """

    gprs: list[GPRPredictor] = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    conche: ConcheClass = field(init=False, repr=False)

    def __post_init__(self):
        bounds = (1e-10, 1e10)
        self.gprs = list(
            map(
                GPRPredictor,
                GPRAnalysis(self.exp_data)
                .add_kernel(Matern(length_scale_bounds=bounds))
                .add_kernel(WhiteKernel(noise_level_bounds=bounds))
                .run_analysis(),
            )
        )
        self.weights = self.get_weights()
        self.conche = self.meta_data.spawn_conche()

    def get_weights(self):
        "Caculate the weights based on c_mean of a phase."
        weights = []
        for gpr, phase in zip(self.gprs, self.exp_data.phases):
            phase_weight = [
                gpr.get_probability(time, val)
                for time, val in zip(phase.sampling_times, phase.c_mean)
            ]
            weights.append(phase_weight)
        return np.array(weights)

    def residuals(
        self, parameters: list, sampling_times: list, fixed_parameters=None
    ) -> float:
        """
        Calculate the squared residuals normalized by the variance of timeseries data.

        Args:
            parameters (list): Parameters that were passed to construct simulation data.
            sampling_times (list): Sampling times of the simulated data.
            fixed_parameters (dict): Parameters that are not optimization targets but fixed.

        Returns:
            np.ndarray: Residuals between experimental data and simulated data.
        """
        if fixed_parameters is None:
            fixed_parameters = {}
        par_dict = (
            self.slice_params(parameters) | self.select_properties() | fixed_parameters
        )
        sim_data = self.conche.run_sim(
            sampling_times, par_dict["c0"], par_dict["mtc"], par_dict["K"]
        )
        ssr = (
            (self.exp_data.c_mean - sim_data) ** 2
        ) / self.exp_data.c_var_for_residuals
        score = ssr * self.weights
        if self.check_for_rapid_change(sampling_times, par_dict):
            score = np.inf
        return np.sum(score)

    def check_for_rapid_change(self, sampling_times: list[float], params: dict):
        smooth_time = np.linspace(sampling_times[0], sampling_times[1], 100).tolist()
        sim_data = self.conche.run_sim(
            smooth_time, params["c0"], params["mtc"], params["K"]
        )
        result = False
        for phase in sim_data:
            dc = np.abs(max(phase) - phase[0])
            if dc / phase[0] > 1.5:
                result = True
                break
        return result

    def plot_comparison(self, popt: list, ax=None, **fixed_parameters) -> plt.Figure:
        fig = super().plot_comparison(popt, ax, **fixed_parameters)
        viz = GPRVisualizer(self.exp_data, [ipredictor.gpr for ipredictor in self.gprs])
        fig = viz.plot(fig.get_axes())
        return fig


class MasterOptimizer:
    """
    This optimizer has a single optimization target. Slave optimizers are
    performing parameter calibration themselves with free parameters. The Master
    Optimizers optimization target is applied to all Slave optimiziers. This is
    basically nested optimization.
    """

    def __init__(self, meta_data: aMetaData, target: OptimizationTarget, cb_file=None):
        self.target = target
        self.meta_data = meta_data
        self.optimizers = []
        self.cb_file = cb_file

    def add_optimizer(self, optimizer: Optimizer) -> MasterOptimizer:
        "Add a slave optimizer."
        self.optimizers.append(optimizer)
        return self

    def residuals(self, param) -> float:
        "Calculate residuals over all slave optimizers."
        master_target = {self.target.var_name: param}
        res = (
            optimizer.run_optimization(
                self.meta_data.sampling_times, fixed_parameters=master_target
            ).fun
            for optimizer in self.optimizers
        )
        return sum(res)

    def callback(self, xk: np.ndarray, convergence):
        residuals = self.residuals(xk)
        out = np.append(xk, residuals)
        if not self.cb_file is None:
            with open(self.cb_file, "ab") as out_file:
                np.savetxt(out_file, out)
        return False

    def run_optimization(self):
        "Do parameter calibration."
        bounds = self.target.bounds
        opt_res = optimize.differential_evolution(
            self.residuals,
            bounds,
            init="sobol",
            updating="deferred",
            workers=-1,
            callback=self.callback,
        )
        return opt_res


class PLOptim(Optimizer):
    "Optimizer for Profile Likelihood Analysis"

    def create_constraint(self, grid: list, i_grid: int, val: float):
        "Create a constraint that fixes a parameter to a certain value."
        A = np.zeros((len(grid), len(grid)))
        A[i_grid, i_grid] = 1
        lb = val
        ub = val
        constraint = optimize.LinearConstraint(A, lb, ub, keep_feasible=False)
        return constraint

    def constrained_optimization_local(self, constraint, bounds, p0, K):
        "Do constrained optimization locally."
        opti_options = {
            "xtol": 1e-32,
            "maxiter": 5000,
        }
        args_ = (K,)
        opt_out = optimize.minimize(
            self.residuals,
            p0,
            args=args_,
            constraints=(constraint,),
            bounds=bounds,
            options=opti_options,
            method="trust-constr",
        )
        return opt_out.fun, opt_out.x

    def constrained_optimization_global(self, constraint, bounds, K):
        """
        Find a global optimum for the parameters using a global optimization
        algorithm

        Args:
            constraint (scipy.optimize.LinearConstraint): Constrain on parameter
            bounds (array_like): bounds for the optimization
            K (array_like): partition_coefficients

        Returns:
            [tuple]: Objective function result and respective parameters
        """
        args_ = (K,)
        opt_out = optimize.differential_evolution(
            self.residuals, bounds, args=args_, constraints=(constraint,), workers=1
        )
        return opt_out.fun, opt_out.x

    def run_PL_optim(self, p0, bounds, K, resolution):
        "Main method of the PL optimizer. Yields the final result"
        # pick gridpoints
        grid = [np.linspace(bound[0], bound[1], resolution) for bound in bounds]
        pl_out = []
        for i_grid, currgrid in enumerate(grid):
            intermediate_list = []
            for currval in currgrid:
                constraint = self.create_constraint(grid, i_grid, currval)
                opt_out_fun, opt_out_x = self.constrained_optimization_local(
                    constraint, bounds, p0, K
                )
                intermediate_list.append((currval, opt_out_fun, *opt_out_x))
                p0 = opt_out_x
            pl_out.append(intermediate_list)
        return pl_out


class PLOptim_with_K(Optimizer):
    "Optimizer for Profile Likelihood Analysis"

    def create_constraint(self, grid: list, i_grid: int, val: float):
        "Create a constraint that fixes a parameter to a certain value."
        A = np.zeros((len(grid), len(grid)))
        A[i_grid, i_grid] = 1
        lb = val
        ub = val
        constraint = optimize.LinearConstraint(A, lb, ub, keep_feasible=False)
        return constraint

    def constrained_optimization_local(self, constraint, bounds, p0, K):
        "Do constrained optimization locally."
        opti_options = {
            "xtol": 1e-32,
            "maxiter": 5000,
        }
        opt_out = optimize.minimize(
            self.residuals,
            p0,
            constraints=(constraint,),
            bounds=bounds,
            options=opti_options,
            method="trust-constr",
        )
        return opt_out.fun, opt_out.x

    def constrained_optimization_global(self, constraint, bounds, K):
        """
        Find a global optimum for the parameters using a global optimization
        algorithm

        Args:
            constraint (scipy.optimize.LinearConstraint): Constrain on parameter
            bounds (array_like): bounds for the optimization
            K (array_like): partition_coefficients

        Returns:
            [tuple]: Objective function result and respective parameters
        """
        opt_out = optimize.differential_evolution(
            self.residuals, bounds, constraints=(constraint,), workers=1
        )
        return opt_out.fun, opt_out.x

    def run_PL_optim(self, p0, bounds, K, resolution):
        "Main method of the PL optimizer. Yields the final result"
        # pick gridpoints
        grid = [np.linspace(bound[0], bound[1], resolution) for bound in bounds]
        pl_out = []
        for i_grid, currgrid in enumerate(grid):
            intermediate_list = []
            for currval in currgrid:
                constraint = self.create_constraint(grid, i_grid, currval)
                opt_out_fun, opt_out_x = self.constrained_optimization_local(
                    constraint, bounds, p0, K
                )
                intermediate_list.append((currval, opt_out_fun, *opt_out_x))
                p0 = opt_out_x
            pl_out.append(intermediate_list)
        return pl_out


class Optimizers(Enum):
    "Enumeration of possible Optimizers."
    OPTIMIZER = Optimizer
    GLOBALOPTIMIZER = GlobalOptimizer
    PLOPTIM = PLOptim
    PLOPTIM_WITH_K = PLOptim_with_K
    OPTWITHGPR = OptWithGPR


if __name__ == "__main__":
    pass
