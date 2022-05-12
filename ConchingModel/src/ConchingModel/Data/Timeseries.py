" Class handling the timeseries data description for experiments. "
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from types import NoneType
from typing import Union

import ConchingModel.Data.MetaData as MetaData
import matplotlib.pyplot as plt
import numpy as np


class PhaseLabel(Enum):
    """Phase type definitions such as COCOABUTTER, COCOAPARTICLE, SUGARPARTICLE, PARTICLES"""

    COCOABUTTER = "cb"
    COCOAPARTICLE = "cp"
    SUGARPARTICLE = "sp"
    PARTICLES = "pp"
    TOTAL = "tt"


class PhaseColors(Enum):
    "Specific colors for plotting phases."

    cb = r"#FFC000"
    COCOABUTTER = r"#FFC000"
    cp = r"#AE5516"
    COCOAPARTICLE = r"#AE5516"
    sp = r"#000000"
    SUGARPARTICLE = r"#000000"
    pp = r"#572A8B"
    PARTICLES = r"#572A8B"
    TOTAL = r"#00FF00"


class SubstanceLabel(Enum):
    "Substance definitions."

    Essigsäure = "acetic_acid"
    Linalool = "linalool"
    Phenylethanol = "phenylethanol"
    TMP = "tmp"
    Benzaldehyd = "benzaldehyd"


@dataclass(frozen=True, order=True)
class aSample:
    """Class describing a single measurement of a Phase at time sampling_time"""

    label: PhaseLabel
    sampling_time: float
    concentration: list[float]

    @property
    def c_mean(self) -> float:
        "Means of concentrations."
        if np.isnan(np.mean(self.concentration)):
            return None
        else:
            return np.mean(self.concentration)

    @property
    def c_var(self) -> Union[float, NoneType]:
        "Variance of measurements."
        if np.var(self.concentration) == 0:
            result = None
        else:
            result = np.var(self.concentration)
        return result

    @property
    def no_outliers(self):
        """Remove outliers that are more than 2 standard deviations from the mean."""
        concentrations = [ic if ic > 0 else 0 for ic in self.concentration]
        if len(concentrations) == 1:
            return aSample(self.label, self.sampling_time, concentrations)
        if all([ic == 0 for ic in concentrations]):
            std = 1
        else:
            std = np.std(concentrations)
        distance = [
            (iconcentration - np.mean(concentrations)) / std
            for iconcentration in concentrations
        ]
        mask = [-1.5 < idist < 1.5 for idist in distance]
        non_outliers = [conc for conc, imask in zip(concentrations, mask) if imask]
        if len(non_outliers) == 0:
            raise ValueError
        return aSample(self.label, self.sampling_time, non_outliers)

    @property
    def c_var_for_residuals(self) -> float:
        "Variance of measurements."
        if np.var(self.concentration) == 0:
            result = 1.0
        else:
            result = np.var(self.concentration)
        return result

    @property
    def c_std(self) -> Union[float, NoneType]:
        "Std of measurements."
        if np.std(self.concentration) == 0:
            result = None
        else:
            result = np.std(self.concentration)
        return result

    @property
    def c_std_for_residuals(self) -> float:
        "Std of measurements."
        if np.std(self.concentration) == 0:
            result = 1.0
        else:
            result = np.std(self.concentration)
        return result

    def __mul__(self, other: float):
        return aSample(
            self.label,
            self.sampling_time,
            np.multiply(self.concentration, other).tolist(),
        )

    def concat(self, other: aSample):
        return aSample(
            self.label,
            self.sampling_time,
            self.concentration + other.concentration,
        )

    def __getitem__(self, val):
        if isinstance(val, int):
            result = self.concentration[val]
        elif isinstance(val, slice):
            result = self.concentration[val.start : val.stop : val.step]
        elif isinstance(val, str):
            result = getattr(self, val)
        else:
            raise TypeError
        return result

    def draw_sample(self, nsamples: int) -> aSample:
        "Sample from the distribution of the current sample and return a new sample."
        c_sample = np.random.normal(self.c_mean, self.c_std, nsamples).tolist()
        return aSample(self.label, self.sampling_time, c_sample)

    def add_measurement(self, sigma_rel: float, n=1) -> aSample:
        "Adds a measurement that is within sigma_rel from mean with a chance of 95%"
        new = np.random.normal(loc=self.c_mean, scale=sigma_rel / 1.96, size=n).tolist()
        if isinstance(self.concentration, float):
            result = [*new, self.concentration]
        elif isinstance(self.concentration, list):
            result = [*new, *self.concentration]
        return aSample(self.label, self.sampling_time, result)

    @staticmethod
    def from_dict(sample: dict) -> aSample:
        "Construct aSample from a dict. Decode JSON content."
        return aSample(
            PhaseLabel[sample["label"]],
            sample["sampling_time"],
            sample["concentration"],
        )


@dataclass(frozen=True)
class aPhase:
    """Class describing a single phase."""

    label: PhaseLabel
    sampling_times: tuple[float]
    samples: list[aSample]

    @property
    def c_mean(self) -> list[float]:
        "Mean of samples."
        return [sample.c_mean for sample in sorted(self.samples)]

    @property
    def c_var(self) -> list[float]:
        "Variance of samples."
        return [sample.c_var for sample in sorted(self.samples)]

    @property
    def c_var_for_residuals(self) -> list[float]:
        "Variance of samples."
        return [sample.c_var_for_residuals for sample in sorted(self.samples)]

    @property
    def c_std(self) -> list[float]:
        "Standard deviation of samples."
        return [sample.c_std for sample in sorted(self.samples)]

    @property
    def c_std_for_residuals(self) -> list[float]:
        "Standard deviation of samples."
        return [sample.c_std_for_residuals for sample in sorted(self.samples)]

    @property
    def no_outliers(self) -> aPhase:
        """Return phase with removed outliers."""
        return aPhase(
            self.label,
            self.sampling_times,
            [isample.no_outliers for isample in self.samples],
        )

    @property
    def color(self) -> str:
        "Return the hex plotting color"
        if isinstance(self.label, str):
            return PhaseColors[self.label].value
        else:
            return PhaseColors[self.label.value].value

    @property
    def label_str(self) -> str:
        "Return the string definition of the label"
        if isinstance(self.label, str):
            return self.label
        else:
            return self.label.value

    def plot(self, ax, fmt="o"):
        "Plot errorbars plot for phase data."
        std = self.c_std.copy()
        std = [0 if istd is None else istd for istd in std]
        ax.errorbar(
            self.sampling_times,
            self.c_mean,
            yerr=std,
            fmt=fmt,
            color=self.color,
            label=self.label.value
            if isinstance(self.label, PhaseLabel)
            else self.label,
        )

    def __getitem__(self, val):
        if isinstance(val, int):
            return aPhase(self.label, self.sampling_times[val], self.samples[val])
        elif isinstance(val, slice):
            return aPhase(
                self.label,
                self.sampling_times[val.start : val.stop : val.step],
                self.samples[val.start : val.stop : val.step],
            )
        elif isinstance(val, str):
            return getattr(self, val)
        else:
            raise TypeError

    def __mul__(self, other: float) -> aPhase:
        samples = [sample * other for sample in self.samples]
        return aPhase(self.label, self.sampling_times, samples)

    def total_to_particle(self, cb: aPhase, fractions: dict):
        "If a total concentration is given, calculate how much of that is due to particle fraction."
        assert self.label == PhaseLabel.TOTAL or self.label == "TOTAL"
        samples = []
        for stt, scb in zip(self.samples, cb.samples):
            new_conc = [
                (
                    (
                        stt.concentration[i]
                        - np.mean(scb.concentration) * fractions["cb"]
                    )
                    / (fractions["cp"] + fractions["sp"])
                )
                for i in range(0, len(stt.concentration))
            ]
            samples.append(aSample(PhaseLabel.PARTICLES, stt.sampling_time, new_conc))
        return aPhase(PhaseLabel.PARTICLES, self.sampling_times, samples)

    def draw_samples(self, nsamples: int) -> aPhase:
        "Resample the whole phase by drawing samples from the underlying dist."
        samples = [sample.draw_sample(nsamples) for sample in self.samples]
        return aPhase(self.label, self.sampling_times, samples)

    def add_measurement(self, sigma_rel: float, n: int = 1) -> aPhase:
        "Add measurements at every sampling time within a relative deviation."
        return aPhase(
            self.label,
            self.sampling_times,
            [sample.add_measurement(sigma_rel, n=n) for sample in self.samples],
        )

    @staticmethod
    def from_dict(phase: dict) -> aPhase:
        "Construct aPhase object from a dict. Decode JSON content."
        samples = [aSample.from_dict(isample) for isample in phase["samples"]]
        return aPhase(PhaseLabel[phase["label"]], phase["sampling_times"], samples)


@dataclass(frozen=True, order=True)
class aTimeseries:
    """Class describing a whole Timeseries observation."""

    phases: list[aPhase]
    sampling_times: tuple[float]

    @property
    def c_mean(self) -> list[float]:
        "Return the mean of the samples."
        return [phase.c_mean for phase in self.phases]

    @property
    def c_var(self) -> list[float]:
        "Return the variance of the samples."
        return [phase.c_var for phase in self.phases]

    @property
    def c_var_for_residuals(self) -> list[float]:
        "Return the variance of the samples."
        return [phase.c_var_for_residuals for phase in self.phases]

    @property
    def c_std(self) -> list[float]:
        "Return the standard deviation of the samples."
        return [phase.c_std for phase in self.phases]

    @property
    def c_std_mean(self):
        """Return mean of standard deviations across sampling times."""
        std = np.array(self.c_std, dtype=np.float64)
        mean = np.nan_to_num(np.nanmean(std, axis=1), nan=0.0)
        return mean

    @property
    def c_std_for_residuals(self) -> list[float]:
        "Return the standard deviation of the samples."
        return [phase.c_std_for_residuals for phase in self.phases]

    @property
    def label(self) -> list[PhaseLabel]:
        """Return label of phases."""
        return [phase.label for phase in self.phases]

    @property
    def no_outliers(self) -> aTimeseries:
        """Remove outliers out of a timeseries and get a new Timeseries."""
        return aTimeseries(
            [iphase.no_outliers for iphase in self.phases], self.sampling_times
        )

    def draw_samples(self, nsamples: int) -> aTimeseries:
        "Resample the whole timeseries by drawing from the dist."
        result = aTimeseries(
            [phase.draw_samples(nsamples) for phase in self.phases], self.sampling_times
        )
        return result

    def add_measurement(self, sigma_rel: float, n: int = 1) -> aTimeseries:
        "Add measurements for every phase and every sampling time inside a bound."
        return aTimeseries(
            [phase.add_measurement(sigma_rel, n=n) for phase in self.phases],
            self.sampling_times,
        )

    def __call__(self, ax=None, fmt="o") -> plt.Figure:
        "Create a visual representation of the concentration profile."
        if ax is None:
            _, ax = plt.subplots(tight_layout=True)
        for phase in self.phases:
            phase.plot(ax, fmt=fmt)
        ax.set_xlabel("Time in h")
        ax.set_ylabel(r"Concentration in $\mu g/g$")
        ax.set_title("Timeseries")
        plt.legend()
        return ax.get_figure()

    def __mul__(self, other: list[float]):
        phases = [phase * multiplier for phase, multiplier in zip(self.phases, other)]
        return aTimeseries(phases, self.sampling_times)

    def __sub__(self, other: aTimeseries):
        if not isinstance(other, aTimeseries):
            raise TypeError
        return np.array(self.c_mean) - np.array(other.c_mean)

    def __getitem__(self, val):
        if isinstance(val, int):
            result = aTimeseries(
                [phase.samples[val] for phase in self.phases], self.sampling_times[val]
            )
        if isinstance(val, slice):
            result = aTimeseries(
                [
                    aPhase(
                        phase.label,
                        phase.sampling_times[val.start : val.stop : val.step],
                        phase.samples[val.start : val.stop : val.step],
                    )
                    for phase in self.phases
                ],
                self.sampling_times[val.start : val.stop : val.step],
            )
        elif isinstance(val, str):
            result = getattr(self, val)
        else:
            raise TypeError
        return result

    def total_to_particles(self, fractions: dict):
        "Separate a total measurement (cb + pp) into pp."
        tt = [
            phase
            for phase in self.phases
            if phase.label == PhaseLabel.TOTAL or phase.label == "TOTAL"
        ][0]
        cb = [
            phase
            for phase in self.phases
            if phase.label == PhaseLabel.COCOABUTTER or phase.label == "COCOABUTTER"
        ][0]
        pp = tt.total_to_particle(cb, fractions)
        new_phases = [
            phase * (1 / (fractions["cp"] + fractions["sp"]))
            for phase in self.phases
            if phase.label != PhaseLabel.TOTAL and phase.label != "TOTAL"
        ]
        new_phases.append(pp)
        ts = aTimeseries(new_phases, self.sampling_times)
        return ts

    @staticmethod
    def from_dict(ts: dict) -> aTimeseries:
        "Construct a Timeseries object from a dict. Decode JSON content."
        phases = [aPhase.from_dict(iphase) for iphase in ts["phases"]]
        return aTimeseries(phases, ts["sampling_times"])

    @staticmethod
    def from_json(loc: str) -> aTimeseries:
        "Construct a standalone Timeseries object from a meta_data json file."
        with open(loc, "r", encoding="utf-8") as f:
            return aTimeseries.from_dict(json.load(f)["timeseries"])

    @staticmethod
    def from_sim(
        meta_data: MetaData.aMetaData, ts: Union[list, np.ndarray]
    ) -> aTimeseries:
        "Create Timeseries from simulation data. Factory method"
        phases = []
        for iphase, phaselabel in zip(ts, meta_data.phases):
            samples = [
                aSample(
                    phaselabel,
                    time,
                    [
                        iphase[itime].tolist(),
                    ],
                )
                for itime, time in enumerate(meta_data.sampling_times)
            ]
            phases.append(aPhase(phaselabel, meta_data.sampling_times, samples))
        return aTimeseries(phases, meta_data.sampling_times)

    def to_json(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self, f, cls=TimeseriesEncoder, indent=4)


class TimeseriesEncoder(json.JSONEncoder):
    "This Encoder extends the JSONEncoder to handle TimeSeries classes."

    def default(self, o):
        if isinstance(o, (aTimeseries, aPhase, aSample)):
            return o.__dict__
        elif isinstance(o, PhaseLabel):
            return o.name
        return super(TimeseriesEncoder, self).default(o)
