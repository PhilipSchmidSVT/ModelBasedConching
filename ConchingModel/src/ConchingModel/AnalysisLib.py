" Handles heavy lifting for analysis types. "
from __future__ import annotations

import copy
from enum import Enum
import itertools
import json
from dataclasses import dataclass, field

import ConchingModel.GPR as GPR
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ConchingModel.Data import MetaData, SimData, Timeseries
from ConchingModel.OptimizerClass import Optimizers
from ConchingModel.ParameterEstimation.Targets import (
    InitialState,
    TargetHandler,
    Targets,
)
from numpy.core.function_base import linspace
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from skopt.sampler import Lhs
from skopt.space import Space


class Analysis:
    """
    Analysis base-class. Sub-Analyses build on this class and expand it.
    """

    def __init__(self, meta_data: MetaData.aMetaData, kwargs: dict, forward_sim=None):
        """
        The analysis class requires a definition of the abstract experiment.
        It uses the DataClass.Data container for this reason. Conches are
        derived from a factory method in this container.
        Args:
            meta_info (DataClass.Data): The meta info used to spawn the
            conche and inform the experiment.
        """
        self.meta_data = copy.deepcopy(meta_data)
        self.kwargs = copy.deepcopy(kwargs[self.__class__.__name__])
        self.forward_sim = copy.deepcopy(forward_sim)
        self.result = {"fig": []}

    def setup_conche(self):
        "Wrapping of conche creation."
        return self.meta_data.spawn_conche()

    def setup_optimizer(self):
        "Wrapping of optimizer creation."
        targets = self.setup_optimization_targets()
        optimizer = self.choose_optimizer()
        print(f"Using Optimizer: {optimizer}")
        return optimizer.value(self.meta_data, self.meta_data.timeseries, targets)

    def choose_optimizer(self):
        optimizer = (
            self.kwargs.get("optimizer")
            if self.kwargs.get("optimizer") is not None
            else "GLOBALOPTIMIZER"
        )
        result = Optimizers[optimizer]
        return result

    def setup_optimization_targets(self) -> TargetHandler:
        targets = [Targets[target] for target in self.kwargs["targets"]]
        result = []
        for target, bound in zip(targets, self.kwargs["bounds"]):
            if target.value == InitialState and bound == "new":
                curr_target = target.value.new(
                    self.meta_data,
                    additional_phase=[0, 1e-8]
                    if self.kwargs["add_empty_sugar"]
                    else None,
                )
            elif target.value == InitialState and bound == "free":
                n_phases = len(self.meta_data.timeseries.phases)
                bounds = [[0.0, 1000.0] for _ in range(0, n_phases)]
                if self.kwargs["add_empty_sugar"]:
                    bounds.append([0, 1e-8])
                curr_target = target.value.set(bounds)
            elif target.value == InitialState and bound == "direct":
                curr_target = target.value.direct(
                    self.meta_data,
                    additional_phase=[0, 1e-8]
                    if self.kwargs["add_empty_sugar"]
                    else None,
                )
            else:
                curr_target = target.value(bound)
            result.append(curr_target)
        return TargetHandler(result)

    def run_analysis(self):
        "abstract method for running any analysis."

    def run_and_save(self, base_path: str):
        "Instruction to run and save an analysis."
        self.run_analysis()
        self.save(base_path)

    def save(self, base_path: str):
        "Instructions to save an analysis."
        try:
            save_object = AnalysisSaveObject(self.result, self)
            save_object.save(base_path)
        except:
            print("Saving encountered Exception")

    @staticmethod
    def sample_parameter_space(bounds: tuple[float], n_samples: int, method: str):
        """
        Sample parameters inside bounds for exploratory study.

        Args:
            bounds (tuple): tuple of tuples containing lower and upper bound
              on parameter space.
            n_samples (int): number of parameter samples to draw.
            method (str): which method to use for sampling
        """
        p_space = Space(bounds)
        lhs = Lhs(lhs_type="classic", criterion=None)
        if method == "lhs":
            samples = lhs.generate(p_space.dimensions, n_samples)
            # flatten
            samples = [[p[i_p] for p in samples] for i_p in range(0, len(samples[0]))]
        elif method == "linspace":
            samples = [
                np.linspace(i_bound[0], i_bound[1], n_samples) for i_bound in bounds
            ]
        elif method == "uniform":
            samples = [
                np.concatenate(
                    [np.random.uniform(i_bound[0], i_bound[1], 1) for i_bound in bounds]
                )
                for i_set in range(0, n_samples)
            ]
        elif method == "normal":
            samples = (
                np.array(
                    [
                        [
                            np.random.normal(np.mean(bound), np.std(bound) / 3, 1)
                            for bound in bounds
                        ]
                        for i_set in range(0, n_samples)
                    ]
                )
                .squeeze()
                .T
            )
        return np.array(samples)

    def run_ensemble_sim(self, param_sets: tuple[float], t: list[float]) -> tuple:
        """
        Simulate the conching process for a set of parameters, resulting in
        n solutions for n parameter sets.

        Args:
            param_sets (list): List of parameter sets
            K (float): Partition coefficient
            N0 (np.ndarray): Initial states of the conche

        Returns:
            ensemble_sols (tuple): Collection of simulation solutions
        """
        ensemble = tuple(
            map(
                lambda p: self.setup_conche().run_sim(
                    t, self.forward_sim["N0"], p, self.forward_sim["partition_coeff"]
                ),
                param_sets,
            )
        )
        return ensemble

    def combine_plots(self, data: list, fig=None):
        "Having two plots, combine their data or create a new plot from data."
        COLORS = ["#00BBBB", "#BBBB00", "#BB00BB"]
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]
        ax.set_prop_cycle(None)
        for i_data in data:
            for i_phase, phase in enumerate(i_data):
                ax.plot(range(0, len(phase)), phase, color=COLORS[i_phase], alpha=0.3)
        return fig


class GlobalParameterEstimation(Analysis):
    "Conduct global parameter estimation on experimental data."

    def run_analysis(self):
        opti = self.setup_optimizer()
        result_opt = opti.run_optimization(self.meta_data.sampling_times)
        self.set_plot_params()
        result_fig = self.format_plot(opti.plot_comparison(result_opt.x))
        self.result = {"opt_out": result_opt, "fig": result_fig}
        return self

    @staticmethod
    def set_plot_params() -> plt.Figure:
        "Set default plot parameters."
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams["figure.figsize"] = [16, 9]
        plt.rc("text", usetex=True)

    @staticmethod
    def format_plot(fig):
        "Format the plots accordingly."
        ax = fig.get_axes()
        for iax in ax:
            iax.spines["right"].set_visible(False)
            iax.spines["top"].set_visible(False)
        return fig


class GaussianProcessRegression(Analysis):
    "Perform a Guassian Process Regression on timeseries data."

    def run_analysis(self):
        "Run the Gaussian Process fitter and return the fit model."
        ts = self.meta_data.timeseries
        bounds = (1e-10, 1e10)
        gpr_fitter = (
            GPR.GPRAnalysis(ts)
            .add_kernel(Matern(length_scale_bounds=bounds))
            .add_kernel(WhiteKernel(noise_level_bounds=bounds))
        )
        return gpr_fitter.run_analysis()


class CleanedGlobalParameterEstimation(GlobalParameterEstimation):
    """Do data cleanup before parameter estimation."""

    def setup_optimizer(self):
        targets = self.setup_optimization_targets()
        optimizer = self.choose_optimizer().value(
            self.meta_data, self.meta_data.timeseries.no_outliers, targets
        )
        print(f"Using optimizer: {optimizer}")
        return optimizer, targets

    def run_analysis(self):
        optimizer, targets = self.setup_optimizer()
        self.announce(targets.targets)
        result_opt = optimizer.run_optimization(self.meta_data.sampling_times)
        result_fig = self.plot(result_opt)
        self.result = {"opt_out": result_opt, "fig": result_fig}
        return self

    def announce(self, targets):
        print("Running parameter estimation for targets:")
        for target in targets:
            print(target)
        print(f"For substance {self.meta_data.substance}")

    def plot(self, result_opt, ax=None):
        "Create a plot comparing experimental data and simulation result."
        if ax is None:
            _, ax = plt.subplots()

        optimizer, _ = self.setup_optimizer()
        self.set_plot_params()
        result = self.format_plot(optimizer.plot_comparison(result_opt.x, ax=ax))
        return result


class LocalParameterEstimation(GlobalParameterEstimation):
    "Run local parameter estimation for mtc."

    def run_analysis(self):
        opt_out = self.setup_optimizer().run_local_optimization(
            self.kwargs["bounds"], np.mean(self.kwargs["bounds"], axis=1)
        )
        fig = self.setup_optimizer().make_comparison_plot(opt_out.x)
        self.result = {"opt_out": opt_out, "fig": fig}
        return self.result


class ParDistFromParSampling(Analysis):
    def run_analysis(self):
        "Run analysis from start to finish."
        samples = self.get_parameter_estimates()
        sims = self.run_sims(samples)
        comparison = [self.TSComparison(sim, self.meta_data.timeseries) for sim in sims]
        accepted = [
            sim for sim in comparison if sim.is_accepted(cutoff=self.kwargs["cutoff"])
        ]
        figs = [self.plot_hist(accepted, key) for key in sims[0].params.keys()]
        self.result = {"accepted": accepted, "fig": figs}
        return self

    def get_parameter_estimates(self):
        """
        Get the parameter estimates by optimizing and sampling a normal
        distribution around the samples.
        """
        targets = self.setup_optimization_targets()
        opti = self.setup_optimizer()
        opt_out = opti.run_optimization(self.meta_data.sampling_times)
        mean = opt_out.x
        samples = np.vstack(
            [
                np.random.normal(param, 0.25 * param, self.kwargs["nsamples"])
                for param in mean
            ]
        ).T.tolist()
        samples = [targets.label_parameters(sample) for sample in samples]
        return samples

    def run_sims(self, params: list[dict]) -> list[Sim]:
        "Run the simulation for each parameter set."
        sims = [
            Timeseries.aTimeseries.from_sim(
                self.meta_data,
                self.setup_conche().run_sim(
                    self.meta_data.timeseries.sampling_times, p["c0"], p["mtc"], p["K"]
                ),
            )
            for p in params
        ]
        result = [self.Sim(sim, paramset) for sim, paramset in zip(sims, params)]
        return result

    def plot_hist(self, sims: list[ParDistFromParSampling.Sim], key: str):
        "Plot a histogram of accepted data parameters."
        if sims:
            pars = [sim.ts_sim.params[key] for sim in sims]
            npar = len(pars[0])
            fig, ax = plt.subplots(1, npar, sharey=False)
            for ipar in range(0, npar):
                ax[ipar].hist([par[ipar] for par in pars], bins="auto", density=True)
                ax[ipar].set_xlabel(f"Parameter value of {key}")
            ax[0].set_ylabel("Number of estimates")
            ax[0].set_title(f"Distribution of parameter {key}")
            return fig
        else:
            return None

    class TSComparison:
        "Holds the comparison of the simulated and experimental timeseries data."

        def __init__(
            self, ts_sim: ParDistFromParSampling.Sim, ts_exp: Timeseries.aTimeseries
        ):
            self.ts_sim = ts_sim
            self.ts_exp = ts_exp

        @property
        def ntimes(self):
            "How many sampling times are included."
            return len(self.ts_sim.sim.sampling_times)

        @property
        def distance(self):
            """Return the deviation of every sampling time from the exp data."""
            return np.sqrt(
                (np.array(self.ts_sim.sim.c_mean) - np.array(self.ts_exp.c_mean)) ** 2
            )

        @property
        def summed_distance(self):
            "Sum of the individual deviations along a phase."
            return np.sum(self.distance, axis=1)

        def is_accepted(self, cutoff: int = 2):
            "Returns true if this parameter sample is within the bounds."
            return np.all(
                self.summed_distance
                < cutoff * self.ntimes * np.array(self.ts_exp.c_std_mean)
            )

    @dataclass
    class Sim:
        sim: Timeseries.aTimeseries
        params: dict


class ResidualMap(Analysis):
    "Calculate the residuals in the vicinity of a specific model realization."

    def run_analysis(self):
        "Create the residual map around a best fit of the model given bounds."
        opti = self.setup_optimizer()
        global_min = opti.run_global_optimization(
            self.kwargs["bounds"], self.kwargs["partition_coeff"]
        )
        brute_out = opti.run_brute_search(
            self.center_bounds_on_global_min(global_min), self.kwargs["partition_coeff"]
        )
        fig = self.create_fig(global_min, brute_out["grid"], brute_out["jout"])
        self.result = {"grid": brute_out["grid"], "jout": brute_out["jout"], "fig": fig}
        return self.result

    def center_bounds_on_global_min(self, global_min, offset=0.5):
        "Center the residual map around the optimized parameters."
        centered_bounds = [
            [boundary * (1 - offset), boundary * (1 + offset)]
            for boundary in global_min.x
        ]
        return centered_bounds

    def create_fig(self, global_min, grid, jout):
        "Create the contour plot for the residual map."
        fig, ax = plt.subplots()
        cont = ax.contourf(grid[0], grid[1], jout, alpha=0.5)
        ax.scatter(global_min.x[0], global_min.x[1], c="r")
        fig.colorbar(cont, ax=ax)
        ax.set_xlabel(r"$\beta_{cbcp}$")
        ax.set_ylabel(r"$\beta_{cba}$")
        return fig


class DataUncertaintyAnalysis(Analysis):
    """
    Analyse model behaviour under different data uncertainty.
    Noise is applied to pure data in increasing amounts and the parameter
    estimatin from this noisy data is observed.
    """

    def run_analysis(self):
        """
        Run the full uncertainty analysis. n_sets of noisy data are produced
        from known parameters p and K. These parameters are then estimated by
        optimization techniques. The statistical distribution of parameters is
        calculated.

        Args:
            p (tuple): list of known parameters
            K (tuple): list of known partition coefficients
            N0 (np.ndarray): initial state of the conching system
            bounds (tuple): bounds in which to search for parameter values
            percent_dev (float): relative deviation from the mean for noisy
              data
            sigma_mul (float, optional): how many sigma-values from the mean is
              still considered to be part of the distribution. Defaults to 1.
            n_sets (int, optional): How many sets of noisy data to generate and
              do parameter estimation for. Defaults to 30.

        Returns:
            res (dict): Dictionary containing a list of estimated parameters,
              a figure of all parameter realizations overlaid on top and
              a list which parameter realizations are within sigma_mul.
        """
        noisy_data = self.create_noisy_datasets()
        noisy_fig = self.plot_noisy_data(noisy_data)

        param_estims, sols = self.estimate_and_solve_datasets(noisy_data)
        combined_fig = self.combine_plots(sols, noisy_fig)
        par_estim_fig = self.plot_histogram_of_params(param_estims)
        is_inside_bounds = tuple(
            map(
                lambda data: self.is_estimate_inside_bounds(
                    np.array(noisy_data).std(axis=0),
                    np.array(noisy_data).mean(axis=0),
                    data,
                    sigma_mul=self.kwargs["sigma_mul"],
                ),
                sols,
            )
        )
        self.result = {
            "params": param_estims,
            "solutions": sols,
            "fig": (combined_fig, par_estim_fig),
            "is_inside": is_inside_bounds,
        }
        return self.result

    def create_noisy_datasets(self) -> list:
        conche = self.meta_data.spawn_conche()
        pure_data = SimData.SimData(
            conche.run_sim(
                self.meta_data.sampling_times,
                self.kwargs["N0"],
                self.kwargs["mtc"],
                self.kwargs["partition_coeff"],
            ),
            self.meta_data.sampling_times,
            self.meta_data.phases,
            conche,
        )
        noisy_data = tuple(
            map(
                lambda _: pure_data.get_perturbed_data(
                    self.kwargs["percent_deviation"]
                ),
                range(0, self.kwargs["n_sets"]),
            )
        )
        return noisy_data

    def plot_noisy_data(self, noisy_data):
        data = np.array(noisy_data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        fig, ax = plt.subplots()
        for i_phase in range(0, mean.shape[0]):
            ax.errorbar(
                self.meta_data.sampling_times,
                mean[i_phase, :],
                yerr=std[i_phase, :],
                marker="o",
                linestyle="None",
            )
        ax.set_xlabel("Time in hours")
        ax.set_ylabel(r"Concentration in $\mu g/g$")
        return fig

    def plot_histogram_of_params(self, par_estims):
        fig, ax = plt.subplots()
        ax.set_ylabel("$n_{par}$")
        ax.set_xlabel("$par_{i}$")
        ax.hist(tuple(zip(*[par.x for par in par_estims])), density=True)
        fig.tight_layout()
        return fig

    def estimate_and_solve_datasets(self, noisy_data):
        param_estims = tuple(map(self.estimate_parameters, noisy_data))
        conche = self.meta_data.spawn_conche()
        param_realizations = tuple(
            map(
                lambda p: conche.run_sim(
                    self.meta_data.sampling_times,
                    self.kwargs["N0"],
                    p,
                    self.kwargs["partition_coeff"],
                ),
                [estim.x for estim in param_estims],
            )
        )
        return param_estims, param_realizations

    def estimate_parameters(self, idata):
        opti = self.meta_data.spawn_optimizer()
        opti.meta_data.timeseries = idata
        opt_out = opti.run_global_optimization(
            self.kwargs["bounds"], self.kwargs["partition_coeff"]
        )
        return opt_out

    def is_estimate_inside_bounds(self, std, mean, realization, sigma_mul=1):
        assert std.shape == realization.shape
        dc = np.abs(realization - mean)
        is_inside = np.all(dc <= std * sigma_mul)
        return is_inside


class UncertaintyAnalysis(Analysis):
    def solve_ensemble(self, ensemble_par: tuple):
        # Solve the parameter ensemble
        result = tuple(
            map(
                lambda mtc: self.meta_data.spawn_conche().run_sim(
                    self.meta_data.sampling_times,
                    self.kwargs["N0"],
                    mtc,
                    self.meta_data.K,
                ),
                ensemble_par,
            )
        )
        return [
            SimData.SimData(res, self.meta_data.sampling_times, self.meta_data.phases)
            for res in result
        ]

    def run_analysis(self):
        # Run the ensemble simulations
        ensemble_par = self.sample_parameter_space(
            self.kwargs["bounds"],
            self.kwargs["nsamples"],
            self.kwargs["sampling_method"],
        ).T
        ensemble_out = self.solve_ensemble(ensemble_par)
        ensemble_sol = np.array([sol.data for sol in ensemble_out])
        figs = self.plot_figures(ensemble_sol)
        self.result = {"fig": figs, "sols": ensemble_out, "pars": ensemble_par}
        return self.result

    def plot_figures(self, ensemble_sol):
        out_fig = []
        nphases = len(self.kwargs["N0"])
        ntimes = len(self.meta_data.sampling_times)
        if self.kwargs["plot_type"] == "scatter":
            for i_phase in range(0, nphases):
                curr_fig, curr_ax = plt.subplots()
                curr_ax.plot(
                    self.meta_data.sampling_times,
                    ensemble_sol[:, i_phase, :].T,
                    "o",
                    color="#0065BD",
                )
                curr_ax.set_title(f"Uncertainty Analysis Phase {i_phase}")
                curr_ax.set_xlabel("Time")
                curr_ax.set_ylabel(r"$c_{i}$")
                out_fig.append(curr_fig)
        elif self.kwargs["plot_type"] == "hist":
            for i_phase in range(0, nphases):
                curr_fig, curr_ax = plt.subplots(1, ntimes, sharey=True)
                curr_ax = [ax.set_xlabel("#") for ax in curr_ax]
                for i_time in range(0, ntimes):
                    curr_ax[i_time].hist(
                        ensemble_sol[:, i_phase, i_time],
                        orientation="horizontal",
                        bins=30,
                    )
                    curr_ax[i_time].set_title(
                        f"t={self.meta_data.sampling_times[i_time]};p={i_phase}"
                    )
                curr_ax[0].set_ylabel("$c_{i}$")
                curr_ax[0].set_xlabel("#")
                curr_fig.tight_layout()
                out_fig.append(curr_fig)
        return out_fig


class MainEffectAnalysis(Analysis):
    # Saltelli_2008

    def get_data_shapes(self, ensemble_data, ensemble_pars):
        "Return the shapes of ensemble data and parameters."
        n_phases = len(ensemble_data[0].phases)
        n_times = len(ensemble_data[0].t)
        n_pars = len(ensemble_pars[0])
        return n_times, n_phases, n_pars

    def create_subplot(
        self,
        lin_fit,
        pars: np.ndarray,
        data: np.ndarray,
        i_par: int,
        i_phase: int,
        i_time: int,
    ) -> plt.Figure:
        "Create scatterplots."
        fig, ax = plt.subplots()
        ax.scatter(pars, data)
        ax.plot(pars, lin_fit(pars), "-r")
        ax.set_title(
            f"Parameter #{i_par + 1}; Phase #{i_phase + 1}; sampling_time {i_time}"
        )
        ax.set_xlabel(f"$Parameter_{i_par + 1}$")
        ax.set_ylabel(f"$c_{i_phase+1, i_time}$")
        ax.legend(["Linear fit", "Ensemble Data"])
        fig.tight_layout()
        return fig

    def compute_main_effect(
        self,
        ensemble_data: np.ndarray,
        ensemble_pars: np.ndarray,
        i_par: int,
        i_phase: int,
        i_time: int,
    ):
        "Compute the main effects."
        pars = ensemble_pars[:, i_par]
        data = np.array([idata.data[i_phase, i_time] for idata in ensemble_data])
        lin_fit = np.polynomial.Polynomial.fit(pars, data, 1)
        return lin_fit, pars, data

    def get_exp_val_interval(
        self,
        ensemble_data: np.ndarray,
        ensemble_pars: np.ndarray,
        i_phase: int,
        i_time: int,
        n_slices: int = 10,
    ):
        "Get the mean data inside an interval of parameter values."
        _, _, n_pars = self.get_data_shapes(ensemble_data, ensemble_pars)
        par_stats = []
        figs = []
        for i_par in range(0, n_pars):
            pars = ensemble_pars[:, i_par]
            data = np.array([idata.data[i_phase, i_time] for idata in ensemble_data])
            slices = self.slice_parranges(pars, n_slices)
            selection = tuple(
                map(lambda slice: self.get_data_within_slice(pars, data, slice), slices)
            )
            # Calculate statistics
            _stats = [(np.mean(sel[0]), np.mean(sel[1])) for sel in selection]
            par_stats.append(_stats)
            fig = self.plot_exp_val_interval(_stats)
            fig.axes[0].set_title(
                f"Exp_Val_Interval for Phase #{i_phase+1}, Par {i_par}, at sampling_time {i_time}"
            )
            fig.tight_layout()
            figs.append(fig)
        return slices, par_stats, fig

    def slice_parranges(self, pars, n_slices):
        "Construct from - to par range."
        slice_borders = linspace(0, pars.max(), num=n_slices)
        slices = [
            (slice_borders[i_border], slice_borders[i_border + 1])
            for i_border in range(0, len(slice_borders) - 1)
        ]
        return slices

    def get_data_within_slice(self, pars, data, _slice):
        selector = np.ma.masked_inside(pars, _slice[0], _slice[1])
        return (pars[selector.mask], data[selector.mask])

    def compute_sensitivity_by_cond_variances(
        self, _stats, ensemble_data, i_phase, i_time
    ):
        expected_vals = [[stat[1] for stat in par] for par in _stats]
        data = np.array([idata.data[i_phase, i_time] for idata in ensemble_data])
        Si = np.var(expected_vals, axis=1) / np.var(data)
        return Si

    @staticmethod
    def plot_exp_val_interval(_stats):
        fig, ax = plt.subplots()
        x = [istat[0] for istat in _stats]
        y = [stat[1] for stat in _stats]
        ax.scatter(x, y)
        fig.tight_layout()
        return fig

    def get_all_effects(self, ensemble_data, ensemble_pars, i_time, i_phase):
        _, _, n_pars = self.get_data_shapes(ensemble_data, ensemble_pars)
        effects = [
            self.compute_main_effect(
                ensemble_data, ensemble_pars, i_par, i_phase, i_time
            )
            for i_par in range(0, n_pars)
        ]
        return effects

    def get_standardized_reg_coefs(self, effects):
        # Saltelli_2008
        n_pars = len(effects)
        standard_coeff_out = np.zeros((n_pars))
        for i_par in range(0, n_pars):
            lin_fit, data, pars = effects[i_par]
            reg_coeff = lin_fit.coef[1]
            std_data = np.std(data)
            std_pars = np.std(pars)
            standard_coeff = reg_coeff * (std_pars / std_data)
            standard_coeff_out[i_par] = standard_coeff
        return standard_coeff_out

    def eval_reg_coeffs(self, standard_reg_coefs, i_phase, i_time):
        labels = [f"Par #{i_par + 1}" for i_par in range(0, len(standard_reg_coefs))]
        fig, ax = plt.subplots()
        ax.set_prop_cycle(None)
        ax.plot(labels, standard_reg_coefs, "o")
        ax.set_title(f"Phase #{i_phase + 1} at sampling_time {i_time}")
        ax.set_ylabel("$dc_{i} / dp$")
        fig.tight_layout()
        return fig

    def create_scatterplots(self, ensemble_data, ensemble_pars, i_time, i_phase):
        _, _, n_pars = self.get_data_shapes(ensemble_data, ensemble_pars)
        out_figs = []
        for i_par in range(0, n_pars):
            lin_fit, pars, data = self.compute_main_effect(
                ensemble_data, ensemble_pars, i_par, i_phase, i_time
            )
            fig = self.create_subplot(lin_fit, pars, data, i_par, i_phase, i_time)
            out_figs.append(fig)
        return out_figs

    def run_analysis(self):
        """
        Compute the normalized main effects from the uncertainty analysis.
        Main effects are the computed by fitting scalar results from the
        ensemble data to respective parameter values and calculating a linear
        fit. The linear relationship between parameter and ensemble output is
        taken as the main effect. Plot this linear relationship.

        Returns:
            [type]: [description]
        """
        # Deconstruct the timeseries and create a fit for every point in time
        # Regression coefficients
        uncert_out = UncertaintyAnalysis(
            self.meta_data, self.kwargs, self.forward_sim
        ).run_analysis()
        sols = uncert_out["sols"]
        pars = uncert_out["pars"]
        self.result = {
            itime: self.get_effect_at_time(sols, pars, itime)
            for itime, _ in enumerate(sols[0].t)
        }
        self.result["uncert_out"] = uncert_out["fig"]
        return self.result

    def get_effect_at_time(self, sols, pars, itime):
        result = [
            self.get_effect_for_phase(sols, pars, itime, iphase)
            for iphase in range(0, len(sols[0].phases))
        ]
        return result

    def get_effect_for_phase(self, sols, pars, itime, iphase):
        effects = self.get_all_effects(sols, pars, itime, iphase)
        standard_reg_coeffs = self.get_standardized_reg_coefs(effects)
        reg_coeff_figs = self.eval_reg_coeffs(standard_reg_coeffs, iphase, itime)
        _, _stats, fig_int = self.get_exp_val_interval(
            sols, pars, iphase, itime, n_slices=20
        )
        first_order_sensitivity = self.compute_sensitivity_by_cond_variances(
            _stats, sols, iphase, itime
        )
        scatter_figs = self.create_scatterplots(sols, pars, itime, iphase)
        result = {
            "reg_coeffs": standard_reg_coeffs,
            "S1": first_order_sensitivity,
            "fig": (fig_int, reg_coeff_figs, scatter_figs),
        }
        return result


class SSRSurface(Analysis):
    """
    Computing the neighbouring sum of squared residuals due to changes in
    design parameters. Gives information about practical identifiability.

    Args:
        Analysis ([dict]): meta information for the conche
    """

    @staticmethod
    def residuals(solx, soly):
        res = solx - soly
        return np.sum(res)

    @staticmethod
    def plot_ssr_surface(p: tuple, ssr: tuple):
        fig, ax = plt.subplots()
        cont = ax.contourf(p[0], p[1], ssr, cmap="Greys")
        fig.colorbar(cont)
        return fig

    def run_analysis(self):
        """
        Run the analysis. Given a known parameter vector p and partition
        coefficient vector K, do a forward simulation. Run an ensemble
        simulation for parameters vals between bounds. Resolution determines
        the gridded parameter values to run.

        Args:
            p (tuple): known mass transport coefficients
            K (tuple): known partition coefficients
            t_sample (tuple): sampling times
            N0 (np.ndarray): initial state of conching system.
            bounds (tuple): tuple of tuples describing the bounds on parameter
              samples
            resolution (int): number of parameter perturbations to test.

        Returns:
            [type]: [description]
        """
        true_sol = self.meta_data.spawn_conche().run_sim(
            self.meta_data.sampling_times,
            self.meta_data.N0,
            self.kwargs["mtc"],
            self.kwargs["partition_coeff"],
        )
        par_samples = self.sample_parameter_space(
            self.kwargs["bounds"],
            self.kwargs["resolution"],
            self.kwargs["sample_method"],
        )
        par_grid = np.array(np.meshgrid(*par_samples))
        par_realizations = np.array(
            [
                [
                    self.setup_conche().run_sim(
                        self.meta_data.sampling_times,
                        self.meta_data.N0,
                        par_grid[:, ip, jp],
                        self.meta_data.K,
                    )
                    for ip in range(0, par_grid.shape[1])
                ]
                for jp in range(0, par_grid.shape[2])
            ]
        )
        ssr = par_realizations - true_sol
        ssr_plot = self.plot_ssr_surface(par_grid, ssr)
        ssr_plot.axes[0].scatter(p[0], p[1])
        return ssr, ssr_plot


class ProfileLikelihoodProper(Analysis):
    "Based on Wieland_2021"

    def run_forward_sim(self, t, N, mtc, K):
        return self.meta_data.spawn_conche().run_sim(t, N, mtc, K)

    def do_parameter_calibration(self, c, bounds, K):
        self.meta_data.timeseries = c
        opt = OptimizerClass.PLOptim(self.meta_data, self.meta_data.timeseries)
        est_out = opt.run_global_optimization(bounds, K)
        return est_out.x

    def run_pl_optimization(self, c, bounds, K, p0, resolution):
        self.meta_data.timeseries = c
        opt = OptimizerClass.PLOptim(self.meta_data, self.meta_data.timeseries)
        pl_out = opt.run_PL_optim(p0, bounds, K, resolution)
        return pl_out

    def create_fig(self, pl_out):
        pl_out = np.array(pl_out)
        figs = []
        for iset in range(0, pl_out.shape[0]):
            fig, ax = plt.subplots()
            ax.plot(pl_out[iset, :, iset + 2], pl_out[iset, :, 1])
            ax.vlines(self.kwargs["mtc"][iset], 0, max(pl_out[iset, :, 1]), colors="r")
            ax.set_xlabel("ParVal")
            ax.set_ylabel("SSR")
            ax.set_title(f"Fixed Par #{iset+1}")
            fig.tight_layout()
            figs.append(fig)
        return figs

    def run_analysis(self):
        sim_data = self.run_forward_sim(
            self.meta_data.sampling_times,
            self.kwargs["N0"],
            self.kwargs["mtc"],
            self.kwargs["partition_coeff"],
        )
        pl_out = self.run_pl_optimization(
            sim_data,
            self.kwargs["bounds"],
            self.kwargs["partition_coeff"],
            self.kwargs["mtc"],
            self.kwargs["resolution"],
        )
        figs = self.create_fig(pl_out)
        self.result = {"fval_and_pars": pl_out, "fig": figs}
        return self


class PofileLikelihoodProperExpData(ProfileLikelihoodProper):
    def run_pl_optimization(self, bounds, K, p0, resolution):
        opt = self.setup_optimizer()
        pl_out = opt.run_PL_optim(p0, bounds, K, resolution)
        return pl_out

    def run_analysis(self):
        self.kwargs["mtc"] = (
            self.setup_optimizer().run_global_optimization(self.kwargs["bounds"]).x
        )
        pl_out = self.run_pl_optimization(
            self.kwargs["bounds"],
            self.meta_data.K,
            self.kwargs["mtc"],
            self.kwargs["resolution"],
        )
        figs = self.create_fig(pl_out)
        self.result = {"fval_and_pars": pl_out, "fig": figs}
        return self


class GlobalSensitivityAnalysis(Analysis):
    def sample_and_solve(self):
        from SALib.sample import saltelli

        par_samples = saltelli.sample(self.kwargs["problem"], self.kwargs["nsamples"])
        sols = self.solve_ensemble(par_samples)
        return par_samples, sols

    def solve_ensemble(self, params):
        sols = np.array(
            [
                self.meta_data.spawn_conche().run_sim(
                    self.meta_data.sampling_times,
                    self.kwargs["N0"],
                    p,
                    self.meta_data.K,
                )
                for p in params
            ]
        )
        return sols

    def calculate_indices(self, sols, itime):
        from SALib.analyze import sobol

        Si_out = {}
        for i_phase, phase in enumerate(self.meta_data.phases):
            sol_subset = np.array([sol[i_phase, itime] for sol in sols])
            Si = sobol.analyze(self.kwargs["problem"], sol_subset)
            Si_out[phase] = Si
        return Si_out

    def run_analysis(self):
        par_samples, sols = self.sample_and_solve()
        Si = {
            f"{self.meta_data.sampling_times[i]:.3f}": self.calculate_indices(sols, i)
            for i in range(1, len(self.meta_data.sampling_times))
        }
        self.result = {
            "par_samples": par_samples,
            "sols": [sol for sol in sols],
            "Si": Si,
        }
        return self


class GlobalSensitivityAnalysisWithc0(GlobalSensitivityAnalysis):
    def solve_ensemble(self, params):
        sols = np.array(
            [
                self.meta_data.spawn_conche().run_sim(
                    self.meta_data.sampling_times,
                    p[len(self.meta_data.phases) :],
                    p[0 : len(self.meta_data.phases)],
                    self.meta_data.K,
                )
                for p in params
            ]
        )
        return sols


class AnalysisParser:
    def __init__(self, analysis_file: str):
        self.analysis_file = analysis_file
        self.content = self.parse_analysis()
        self.forward_sim = self.content["forward_sim"]
        self.meta_data = MetaData.MetaData(self.content["meta_info"])
        self.out_path = "/".join(analysis_file.split("/")[0:-1])
        self.to_run = self.content["Analysis"].keys()
        self.analyses = {}

    def parse_analysis(self):
        with open(self.analysis_file) as f:
            content = json.load(f)
        return content

    def save_analyses(self, out_path: str):
        for analysis in self.analyses.values():
            analysis.save(out_path)

    def run_analyses(self):
        if "GSA" in self.to_run:
            kwargs = self.content["Analysis"]["GSA"]
            self.analyses["GSA"] = self.run_GSA(
                self.meta_data, self.forward_sim, kwargs
            )
        if "PLP" in self.to_run:
            kwargs = self.content["Analysis"]["PLP"]
            self.analyses["PLP"] = self.run_PLP(
                self.meta_data, self.forward_sim, kwargs
            )
        return self.analyses

    def run_GSA(self, meta_data, forward_sim, kwargs):
        analysis = GlobalSensitivityAnalysis(meta_data, kwargs["problem"])
        analysis.run_analysis(
            forward_sim["partition_coeff"],
            forward_sim["N0"],
            kwargs["n_samples"],
        )
        return analysis

    def run_PLP(self, meta_data, forward_sim, kwargs):
        analysis = ProfileLikelihoodProper(meta_data)
        analysis.run_analysis(
            meta_data.sampling_times,
            forward_sim["N0"],
            forward_sim["mtc"],
            forward_sim["partition_coeff"],
            kwargs["bounds"],
            kwargs["resolution"],
        )
        return analysis


class DYNIA(Analysis):
    """Conducting DYNamic Identifiability Analysis, Wagener_2001c."""

    def run_analysis(self):
        targets = self.setup_optimization_targets()
        params = self.get_uniformly_distributed_parameters(
            targets, n=self.kwargs["nsets"]
        )
        ensembles = self.simulate_ensemble(targets, params)
        top_percent = self.compute_OF_for_window(ensembles).crop_underperformer()
        figs = [
            [self.plot(par_sets, key, iwindow) for key in par_sets[0].params.keys()]
            for iwindow in range(0, len(par_sets[0].nwins))
        ]
        self.result = {"fig": figs}
        return self

    def timeseries_windows(self, ts: Timeseries.aTimeseries, n=3):
        return [ts[s : s + n] for s in range(0, len(ts.sampling_times) - n + 1)]

    def get_uniformly_distributed_parameters(self, targets: TargetHandler, n=50):
        bounds = targets.bounds_2_elements
        samples = [
            np.random.uniform(low=bounds[0], high=bounds[1]) for _ in range(0, n)
        ]
        return samples

    def simulate_ensemble(self, targets: TargetHandler, par_samples: np.ndarray):
        """Simulate all uniformly drawn parameter samples."""
        result = []
        for sample in par_samples:
            params = targets.label_parameters(sample)
            sim_data = Timeseries.aTimeseries.from_sim(
                self.meta_data,
                self.setup_conche().run_sim(
                    self.meta_data.sampling_times,
                    params["c0"],
                    params["mtc"],
                    params["K"],
                ),
            )
            result.append((sim_data, params))
        return result

    def compute_OF_for_window(
        self, ensemble_results: list[tuple[Timeseries.aTimeseries, dict]]
    ) -> Sets:
        """Get the objective function evaluation over every window."""
        par_sets = []
        windows_exp_data = self.timeseries_windows(
            self.meta_data.timeseries, self.kwargs["window_size"]
        )
        for result in ensemble_results:
            result_windows = self.timeseries_windows(
                result[0], self.kwargs["window_size"]
            )
            window_pair = [
                self.WindowPair(exp, sim, result[1])
                for exp, sim in zip(windows_exp_data, result_windows)
            ]
            par_sets.append(self.aSet(window_pair, result[1]))
        return self.Sets(par_sets)

    def plot(self, costs: list[aSet], key: str, window: int, fig: plt.Figure = None):
        """Make scatterplots of costs vs parameter value."""
        if fig == None:
            par_items = len(costs[0].params[key])
            fig, ax = plt.subplots(1, par_items, sharey=True)
        subset = [
            (cost.windows[window], cost.windows[window].params[key]) for cost in costs
        ]
        for set in subset:
            for currpar, currax in zip(set[1], ax):
                currax.scatter(currpar, set[0])
                currax.set_xlabel(f"Par value {key}")
        ax[0].set_ylabel("SSR")
        ax[0].set_title(f"TimeWindow_{window}_Parameter_{key}")
        fig.tight_layout()
        return fig

    @dataclass(frozen=True, order=True, eq=True)
    class aSet:
        """Wrapper class for costs"""

        windows: list[DYNIA.WindowPair]
        params: dict

        @property
        def nwins(self):
            return len(self.windows)

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return self.windows[idx]
            elif isinstance(idx, slice):
                return self.windows[idx.start : idx.stop : idx.step]

    @dataclass(frozen=True)
    class Sets:
        """Wrapper for multiple aSets."""

        sets: list[DYNIA.aSet]

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return self.sets[idx]
            elif isinstance(idx, slice):
                return self.sets[idx.start : idx.stop : idx.step]
            else:
                raise TypeError

        def __len__(self):
            return len(self.sets)

        def get_window(self, idx: int):
            return [par_set[idx] for par_set in self.sets]

        def get_best_performing_for_window(
            self, iwin: int, cutoff: float = 0.1
        ) -> list[DYNIA.WindowPair]:
            """Get best performing sets for a specific window."""
            sorted_windows = sorted(self.get_window(iwin))
            return sorted_windows[0 : int(len(sorted_windows) * cutoff)]

        def crop_underperformer(self, cutoff: float = 0.1) -> DYNIA.Sets:
            """Get all best performing for every window-series."""
            windows = [
                self.get_best_performing_for_window(iwin, cutoff=cutoff)
                for iwin in range(0, self.sets[0].nwins)
            ]
            new_set = DYNIA.Sets(
                [
                    DYNIA.aSet(win, win[0].params)
                    for win in itertools.zip_longest(*windows)
                ]
            )
            return new_set

    @dataclass(frozen=False, order=True, eq=True)
    class WindowPair:
        """Wrapper for a single time window."""

        cost: float = field(init=False)
        w1: Timeseries.aTimeseries = field(repr=False)
        w2: Timeseries.aTimeseries = field(repr=False)
        params: dict

        def __post_init__(self):
            self.cost = (np.sum(np.sqrt((self.w1 - self.w2) ** 2))) / len(
                self.w2.sampling_times
            )


class AnalysisSaveObject:
    def __init__(self, data: dict, parent: Analysis):
        self.data = data
        self.parent = parent

    def save(self, base_path):
        self.save_arrays(base_path)
        # self.save_figures(base_path)
        self.save_fig_recursively(self.data, base_path)

    def save_arrays(self, base_path):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, mpl.figure.Figure):
                    return True
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, SimData.SimData):
                    return obj.data.tolist()
                else:
                    return "Could not read"
                return json.JSONEncoder.default(self, obj)

        with open(
            f"{self.construct_local_path(base_path)}/out.json", "w", encoding="utf-8"
        ) as file_out:
            json.dump(self.data, file_out, cls=NumpyEncoder, indent="\t")

    def save_figures(self, base_path):
        figs = self.data["fig"]
        if isinstance(figs, (list, tuple)):
            for ifig, fig in enumerate(figs):
                fig.savefig(f"{self.construct_local_path(base_path)}/{ifig}.svg")
        else:
            figs.savefig(f"{self.construct_local_path(base_path)}/0.svg")

    def save_fig_recursively(self, data, base_path):
        for item in data:
            if isinstance(data, dict):
                item = list(data.values())
            if isinstance(item, (tuple, list)):
                self.save_fig_recursively(item, base_path)
            elif isinstance(item, dict):
                self.save_fig_recursively(item.values(), base_path)
            elif isinstance(item, mpl.figure.Figure):
                item.savefig(
                    f"{self.construct_local_path(base_path)}/{item.axes[0].get_title()}.svg"
                )

    def construct_local_path(self, path):
        import os

        path_salt = self.parent.__class__.__name__
        resulting_path = os.path.join(path, path_salt)
        os.makedirs(resulting_path, exist_ok=True)
        return resulting_path


class AnalysisInfoObject:
    def __init__(self, path):
        self.data = self.load_json(path)

    def load_json(self, path):
        with open(path, "r") as f:
            result = json.load(f)
        return result

    @property
    def info(self):
        return self.meta_data, self.kwargs, self.forward_sim

    @property
    def meta_data(self):
        result = {
            key: self.unpack_meta_data(val)
            for key, val in self.data["meta_data"].items()
        }
        return result

    @property
    def kwargs(self):
        return self.data["kwargs"]

    @property
    def forward_sim(self):
        return self.data["forward_sim"]

    def unpack_meta_data(self, idata):
        if isinstance(idata, dict):
            return MetaData.MetaData(idata)
        elif isinstance(idata, str):
            with open(idata, "r") as source:
                return MetaData.aMetaData(json.load(source))
        else:
            raise TypeError


class Analyses(Enum):
    "Enum of analyses"
    LocalParameterEstimation = LocalParameterEstimation
    GlobalParameterEstimation = GlobalParameterEstimation
    CleanedGlobalParameterEstimation = CleanedGlobalParameterEstimation
    GaussianProcessRegression = GaussianProcessRegression
    MainEffectAnalysis = MainEffectAnalysis
    ProfileLikelihoodProper = ProfileLikelihoodProper
    GlobalSensitivityAnalysis = GlobalSensitivityAnalysis
    GlobalSensitivityAnalysisWithc0 = GlobalSensitivityAnalysisWithc0


if __name__ == "__main__":
    pass
