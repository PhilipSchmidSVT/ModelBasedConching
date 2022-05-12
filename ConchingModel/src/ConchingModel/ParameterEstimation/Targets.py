"Collection of parameter estimation targets."
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, chain, pairwise

import numpy as np
from ConchingModel.Data.MetaData import aMetaData
from scipy import optimize


@dataclass(frozen=True)
class OptimizationTarget(ABC):
    """Base class for the targets of optimization."""

    _bounds: list[float]

    @property
    def bounds(self) -> list[float]:
        """Returns the bounds of the specific optimization target."""
        return self._bounds

    @property
    def upper_bound(self) -> float:
        """Return the upper bound."""
        return self._bounds[-1]

    @property
    def lower_bound(self) -> float:
        """Return the lower bound."""
        return self._bounds[0]

    @property
    def mean_x0(self) -> float:
        """Return the mean of the bounds as starting vals for the optimization."""
        return np.mean(self.bounds, axis=1)

    @property
    def len(self) -> int:
        """Return how many items are included."""
        return len(self.bounds)


@dataclass(frozen=True)
class InitialState(OptimizationTarget):
    """Optimize for initial state."""

    @property
    def var_name(self):
        return "c0"

    @staticmethod
    def from_data(
        c0: list, c0_std: list, sigma: float = 2.0, additional_phase=None
    ) -> OptimizationTarget:
        """Create InitialState target from experimental bounds."""
        upper = np.array(np.array(c0) + sigma * np.array(c0_std))
        lower = np.array(np.array(c0) - sigma * np.mean(c0_std))
        lower[lower < 0] = 0
        bounds = list(zip(lower, upper))
        if not additional_phase is None:
            bounds.append(additional_phase)
        return InitialState(bounds)

    @staticmethod
    def new(
        meta_data: aMetaData, sigma: int = 2, additional_phase=None
    ) -> OptimizationTarget:
        "Create a boundary defined by the standard deviation of the underlying data."
        upper = np.add(meta_data.N0, sigma * np.array(meta_data.timeseries.c_std_mean))
        lower = np.subtract(
            meta_data.N0, sigma * np.array(meta_data.timeseries.c_std_mean)
        )
        lower[lower < 0] = 0
        bounds = list(zip(lower, upper))
        if not additional_phase is None:
            bounds.append(additional_phase)
        return InitialState(bounds)

    @staticmethod
    def direct(meta_data: aMetaData, additional_phase=None):
        upper = meta_data.N0
        lower = meta_data.N0
        bounds = list(zip(lower, upper))
        if not additional_phase is None:
            bounds.append(additional_phase)
        return InitialState(bounds)

    @staticmethod
    def set(bounds, additional_phase=None):
        if not additional_phase is None:
            bounds.append(additional_phase)
        return InitialState(bounds)


@dataclass(frozen=True)
class MassTransportCoefficients(OptimizationTarget):
    """Optimize for mass transport coefficients."""

    @property
    def var_name(self):
        return "mtc"


@dataclass(frozen=True)
class PartitionCoefficients(OptimizationTarget):
    """Optimize for partition coefficients."""

    @property
    def var_name(self):
        return "K"


class Targets(Enum):
    """Enumeration of optimization targets."""

    MASSTRANSPORTCOEFFICIENTS = MassTransportCoefficients
    PARTITIONCOEFFICIENTS = PartitionCoefficients
    INITIALSTATE = InitialState


@dataclass
class TargetHandler:
    """Handles a collection of targets for the optimization."""

    targets: list[OptimizationTarget]

    @property
    def bounds(self):
        return [target.bounds for target in self.targets]

    @property
    def bounds_obj(self):
        """Return all bounds."""
        return optimize.Bounds(
            self.bounds_2_elements[0], self.bounds_2_elements[1], keep_feasible=True
        )

    @property
    def bounds_2_elements(self):
        return list(zip(*[subitem for item in self.bounds for subitem in item]))

    @property
    def mean_x0(self):
        """Return all starting points as the mean of x0."""
        return list(chain.from_iterable([target.mean_x0 for target in self.targets]))

    @property
    def var_name(self):
        return {target.var_name for target in self.targets}

    def label_parameters(self, params: list[float]) -> dict:
        """Assign the params handed to residuals to a dict with their respective labels."""
        par_indices = list([0, *accumulate([target.len for target in self.targets])])
        start_and_end = list(pairwise(par_indices))
        sliced_params = [params[s:e] for s, e in start_and_end]
        result = {
            target.var_name: param for target, param in zip(self.targets, sliced_params)
        }
        return result
