"""" Data wrapping post simulation data in an accessor. """

import json

import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from bunch import Bunch


@dataclass
class aSimData:
    "Wrapper for solve_ivp output data"
    data: Bunch
    sampling_times: list[float | int]

    @property
    def T(self):
        return self.data.y.T

    def __call__(self):
        fig, ax = plt.subplots()
        ax.plot(self.sampling_times, self.data.y.T, "-o")
        ax.set_xlabel("Time in h")
        ax.set_ylabel(r"Concentration in $\mu g\ per\ g$")
        return fig

    def __sub__(self, other):
        if type(other) is not type(self):
            raise TypeError
        else:
            diff = (self.data.y - other) ** 2
        return np.sum(diff).astype("float64")

    def write_json(self, out_name):
        assert not self.conche is None
        out_data = self.conche.meta_info._meta_info
        out_data["timeseries"] = self.data.tolist()
        with open(out_name, "w") as out_file:
            json.dump(out_data, out_file)
        return out_data
