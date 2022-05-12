"""
Class handling the visualization of timeseries components.
"""
import matplotlib.pyplot as plt
import numpy as np
from ConchingModel.Data.Timeseries import PhaseColors, aPhase, aSample, aTimeseries


class tsViz:
    "Handles visualization of timeseries"

    def __init__(self, ts: aTimeseries):
        self.ts = ts

    def __call__(self, ax=None, fmt: str="o", prefix: str="", suffix: str="") -> plt.Figure:
        "Visualize the whole timeseries."
        if ax is None:
            fig, ax = plt.subplots()

        for phase in self.ts.phases:
            fig, ax = self.viz_phase(phase, ax=ax, fmt=fmt, prefix=prefix, suffix=suffix)
        fig = self.annotate_plot(fig, ax)
        return fig

    def viz_phase(
        self, phase: aPhase, ax: plt.Axes = None, fmt: str = "o", prefix: str= "", suffix: str = ""
    ):
        "Visualize a single sample."
        if ax is None:
            _, ax = plt.subplots(squeeze=True)

        for sample in phase.samples:
            _, ax = self.viz_sample(sample, ax=ax, fmt=fmt)
        return ax.get_figure(), ax

    def viz_sample(
        self, sample: aSample, ax: plt.Axes = None, fmt: str = "o", prefix: str = "", suffix: str = ""
    ) -> tuple[plt.Figure, plt.Axes]:
        "Visualize a given sample."
        if ax is None:
            _, ax = plt.subplots(squeeze=True)

        if isinstance(sample.label, str):
            color = PhaseColors[sample.label].value
        else:
            color = PhaseColors[sample.label.value].value
        for ic in sample.concentration:
            ax.plot(sample.sampling_time, ic, fmt, color=color)
        return ax.get_figure(), ax

    def annotate_plot(self, fig: plt.Figure, ax: plt.Axes) -> plt.Figure:
        "Add annotations to the timeseries plot."
        ax.set_title("Timeseries")
        ax.set_ylabel(r"concentration / $\mu g / g$")
        ax.set_xlabel(r"Time / h")
        plt.legend([phase.label for phase in self.ts.phases])
        return fig


class tsVizErrorbar(tsViz):
    "Visualize samples by errorbar."

    def viz_phase(
        self, phase: aPhase, ax: plt.Axes = None, fmt: str = "o", prefix: str = "", suffix: str = ""
    ) -> plt.Figure:
        "Visualize a given sample."
        if ax is None:
            _, ax = plt.subplots(squeeze=True)

        std = [0 if isample is None else isample for isample in phase.c_std]
        ax.errorbar(
            phase.sampling_times,
            phase.c_mean,
            yerr=std,
            fmt=fmt,
            color=phase.color,
            label=prefix + phase.label_str + suffix,
        )
        return ax.get_figure(), ax

class tsVizTotalAmount(tsViz):

    def __call__(self, meta_data, ax=None, fmt: str = "o", prefix: str = "", suffix: str = "") -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(squeeze=True, tight_layout=True)
        ax2 = ax.twinx()
        total = np.multiply(np.array(self.ts.c_mean).T, np.multiply(list(meta_data.fractions.values()),  meta_data.mass)).T
        for phase, concentration in zip(self.ts.phases, total):
            ax2.errorbar(
                self.ts.sampling_times,
                concentration,
                yerr=0,
                fmt=fmt,
                color=phase.color,
                label= prefix + phase.label_str + suffix,
            )

        total_sum = np.sum(total, axis=0)
        ax2.errorbar(
            self.ts.sampling_times,
            total_sum,
            yerr=0,
            fmt = '-.',
            label = "Total amount"
        )
        ax2.set_ylabel(r"Phase bound mass / $\mu g$")
        return ax.get_figure()
