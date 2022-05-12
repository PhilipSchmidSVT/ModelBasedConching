""" Class containing meta data for all experiments.
Program relies on data being wrapped into a MetaData object. """
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property, reduce

import ConchingModel.Data.Timeseries as Timeseries
from ConchingModel import ConcheClass
from ConchingModel.PhysicalProcessParameters import (
    PhysicalProcessParameters,
    SuspensionProcessParameters,
)
from ConchingModel.PropertyLib import part_coeffs_buhler_gpr


@dataclass
class aMetaData:
    """General class construct for data interfacing"""

    _meta_info: dict
    PPP: PhysicalProcessParameters = field(init=False)
    SPP: SuspensionProcessParameters = field(init=False)

    def __post_init__(self):
        self.PPP = PhysicalProcessParameters(self)
        self.SPP = SuspensionProcessParameters(self)

    # def __init__(self, meta_info, timeseries=None):
    # self._meta_info = meta_info
    # self.PPP = PhysicalProcessParameters.PhysicalProcessParameters(self)
    # self.SPP = PhysicalProcessParameters.SuspensionProcessParameters(self)
    # self._timeseries = timeseries
    # if self._timeseries is None:
    # self._timeseries = np.array([])

    @property
    def mass(self):
        "Return the mass of the whole system."
        if isinstance(self._meta_info["mass"], (int, float)):
            return self._meta_info["mass"]
        else:
            raise TypeError

    @property
    def timeseries(self) -> Timeseries.aTimeseries:
        "Return the timeseries data of the system."
        ts = Timeseries.aTimeseries.from_dict(self._meta_info["timeseries"])
        if Timeseries.PhaseLabel.TOTAL in ts.label:
            ts = ts.total_to_particles(self.fractions)
        return ts

    @timeseries.setter
    def timeseries(self, series: Timeseries.aTimeseries):
        if isinstance(series, (Timeseries.aTimeseries, dict)):
            self._meta_info["timeseries"] = series
        else:
            raise TypeError

    @property
    def N0(self) -> list:
        "Returns the initial concentration of the experimental data."
        result = [phase[0] for phase in self.timeseries.c_mean]
        if len(result) <= 2 and self.is_combined_particle_phase:
            particle_phase_mass = sum(self.phase_mass[1:])
            result.append(0)
            result[1] = result[1] * (particle_phase_mass / self.phase_mass[1])
        return result

    @property
    def substance(self) -> str:
        "Returns the substance linked with this experimental data."
        if not isinstance(self._meta_info["substance"], str):
            raise TypeError
        return self._meta_info["substance"]

    @property
    def sampling_times(self) -> list[float]:
        "Sampling times at which samples were taken."
        if not isinstance(self._meta_info["sampling_times"], (tuple, list)):
            raise TypeError
        return self._meta_info["sampling_times"]

    @sampling_times.setter
    def sampling_times(self, val: list):
        self._meta_info["sampling_times"] = val

    @property
    def fractions(self) -> dict:
        "Return the fractional distribution of phase masses."
        if not isinstance(self._meta_info["fractions"], dict):
            raise TypeError
        return self._meta_info["fractions"]

    @property
    def phases(self) -> list[str]:
        "Get the phaselabels of phases, whose fractions are not 0."
        phases = [
            key for key, val in self.fractions.items() if val is not None and val != 0
        ]
        if self.is_combined_particle_phase:
            phases = ["cb", "pp"]
        return phases

    @property
    def K(self) -> tuple[float]:
        """
        Return the partition coefficients of the given system
        from part_coeff_lib.
        """

        result = tuple(
            part_coeffs_buhler_gpr.part_coeffs[phase.value][self.substance]
            if isinstance(phase, Timeseries.PhaseLabel)
            else part_coeffs_buhler_gpr.part_coeffs[phase][self.substance]
            for phase in self.phases[1:]
        )
        return result

    @property
    def phasetypes(self) -> list:
        "A converter of phase labels to their phase types. Legacy"
        converter = {
            "cb": "continuous",
            "cp": "discontinuous",
            "sp": "discontinuous",
            "pp": "discontinuous",
            "air": "air",
        }
        phases = self.phases
        phasetypes = [converter.get(phase) for phase in phases]
        assert phasetypes[0] == "continuous"
        return phasetypes

    @property
    def temp(self):
        "Temperature of the suspension."
        return self.PPP.temp

    @cached_property
    def phase_mass(self) -> list[float]:
        "Get mass for all phases."
        phases = list(self.fractions.keys())
        result = [self.mass * self.fractions.get(phase) for phase in phases]
        return result

    @property
    def nonzero_mass(self) -> list:
        "Get mass of phase if fraction of phase is not 0."
        mass = list(self.phase_mass.values())
        nonzero_mass = [phase_mass for phase_mass in mass if phase_mass != 0]
        return nonzero_mass

    @property
    def is_combined_particle_phase(self) -> bool:
        """Check if the particle phase is observed
        as a combined sugar and cocoa particle phase"""
        if self._meta_info.get("is_combined_particle_phase") is None:
            result = False
        else:
            result = self._meta_info["is_combined_particle_phase"]
        return result

    @is_combined_particle_phase.setter
    def is_combined_particle_phase(self, val):
        self._meta_info["is_combined_particle_phase"] = val

    def spawn_conche(self):
        "Spawn a ConcheClass.ConcheClass from MetaData. Factory method"
        if not self.is_combined_particle_phase:
            conche = ConcheClass.ConcheClass(self)
        else:
            conche = ConcheClass.ConcheClassCombinedParticles(self)
        return conche

    def are_timeseries_equal_length(self):
        "Check if all phases have the same number of samples."
        len_of_timeseries = [len(phase.samples) for phase in self.timeseries.phases]
        return all(
            [
                reduce(lambda x, y: x == y, len_of_timeseries),
            ]
        )

    def from_mtc(self, mtc: list[float], N0=None) -> aMetaData:
        "Create timeseries from mass transport coefficients by forward sim."
        if N0 is None:
            sim_data = self.spawn_conche().run_sim(
                self.sampling_times, self.N0, mtc, self.K
            )
        else:
            sim_data = self.spawn_conche().run_sim(self.sampling_times, N0, mtc, self.K)

        ts = Timeseries.aTimeseries.from_sim(self, sim_data)
        meta_info = self._meta_info.copy()
        meta_info["timeseries"] = ts.__dict__
        return aMetaData(meta_info)

    def add_ts(self, ts) -> aMetaData:
        new_data = self._meta_info.copy()
        new_data["timeseries"] = ts
        return aMetaData(new_data)

    def to_json(self, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._meta_info, f, indent=4, cls=MetaDataEncoder)


class MetaDataEncoder(json.JSONEncoder):
    "This Encoder extends the JSONEncoder to handle TimeSeries classes,"

    def default(self, o):
        if isinstance(o, (aMetaData)):
            return o.__dict__
        elif isinstance(o, (PhysicalProcessParameters, SuspensionProcessParameters)):
            return "PPPorSPP"
        elif isinstance(o, Timeseries.PhaseLabel):
            return o.name
        if isinstance(
            o, (Timeseries.aTimeseries, Timeseries.aPhase, Timeseries.aSample)
        ):
            return o.__dict__
        return super(MetaDataEncoder, self).default(o)
