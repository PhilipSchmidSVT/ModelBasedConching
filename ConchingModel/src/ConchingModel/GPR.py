"Fit statistical models to time series data."
from dataclasses import dataclass, field

import ConchingModel.Data.Timeseries as Timeseries
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor


@dataclass
class GPRAnalysis:
    "Fit a Gaussian Process to a timeseries."
    ts: Timeseries.aTimeseries
    kernels: list = field(default_factory=list)

    def add_kernel(self, kernel):
        "Builder function. Add necessary kernels."
        self.kernels.append(kernel)
        return self

    def run_for_phase(self, phase: Timeseries.aPhase):
        "Fit GP to a single phase time series data."
        nsamples = len(phase.samples[0].concentration)
        sampling_times = (
            np.array(phase.sampling_times).reshape(-1, 1).repeat(nsamples, axis=1)
        )
        concentrations = [isample.concentration for isample in phase.samples]
        gpr = GaussianProcessRegressor(sum(self.kernels), n_restarts_optimizer=10)
        gpr.fit(sampling_times, concentrations)
        return gpr

    def run_analysis(self):
        "Fit the GP to all phases individually."
        gpr = list(map(self.run_for_phase, self.ts.phases))
        return gpr


@dataclass
class GPRPredictor:
    "Wrapper for easy predictions of mean and std at sampling times."
    gpr: GaussianProcessRegressor

    def predict_at(self, time: float) -> list[float, float]:
        "Predict the mean and std at time t for every phase."
        t_vec = np.array(time).reshape(1, -1).repeat(self.gpr.n_features_in_, axis=1)
        c_mean, c_std = self.gpr.predict(t_vec, return_std=True)
        return [c_mean.mean(), c_std.mean()]

    def get_probability(self, time: float, val: float):
        """
        Get the probability of a point assuming the point belongs to a normal distribution
        calculated from the Gaussin Process Regressor.
        """
        c_mean, c_std = self.predict_at(time)
        return scipy.stats.norm(c_mean, c_std).pdf(val)


class GPRVisualizer:
    "Visualize the results of a fit GP in context of a time series"

    def __init__(self, ts, gpr):
        self.ts = ts
        self.gpr = gpr

    def x(self, phase):
        "Converts single feature to nfeatures."
        return (
            np.array(self.ts.sampling_times)
            .reshape(-1, 1)
            .repeat(len(phase.samples[0].concentration), axis=1)
        )

    def plot(self, axes=None):
        if axes is None:
            fig = self.ts()
            axes = fig.get_axes()
        for ax in axes:
            for phase, gpr in zip(self.ts.phases, self.gpr):
                x = np.linspace(
                    min(self.ts.sampling_times), max(self.ts.sampling_times), 100
                )
                X = x.reshape(-1, 1).repeat(len(phase.samples[0].concentration), axis=1)
                y_mean, y_std = gpr.predict(X, return_std=True)
                ax.plot(
                    x,
                    np.mean(y_mean, axis=1),
                    "-",
                    color=phase.color,
                    label=f"GP-Prediction: {phase.label.value}",
                )
                ax.fill_between(
                    x,
                    np.mean(y_mean, axis=1) - y_std,
                    np.mean(y_mean, axis=1) + y_std,
                    color=phase.color,
                    alpha=0.3,
                    label=f"GP-Prediction Standard Dev: {phase.label.value}",
                )
        plt.legend()
        return axes[0].get_figure()
