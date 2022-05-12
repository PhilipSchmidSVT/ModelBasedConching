" Handles experiment setup and parameter estimation from metadata. "
import json
from typing import Union

import ConchingModel.Data.MetaData as MetaData
import ConchingModel.Data.Timeseries as Timeseries
import matplotlib.pyplot as plt
from ConchingModel import AnalysisLib, DataLoaders
from ConchingModel.PropertyLib import part_coeff_lib
from scipy.optimize.optimize import OptimizeResult


class Experiment:
    """Class containing and governing the metadata of a given experiment.
    Informs the Analysis data about its specific input
    """

    def __init__(
        self,
        meta_data: Union[str, MetaData.aMetaData, dict],
        partition_coeffs: list[float] = None,
    ) -> None:
        """Initializing the experiment
        Objects of this class describe a specific experiment. Its information
        is contained in the meta_file.
        The meta_file contains:
            mass[float] -> sample mass in g
            fractions[dict] -> recipe of experiment (e.g. 'cb':0.5, 'cp':0.5)
            sampling_times[list] -> At which point the experiment was sampled.
            temp[float] -> temperature at which data is sampled.
        Arguments:
            meta_file -> str or dict: path to json file or dict containing the
              information above.
            load_type -> str: Tells the data loader which specific loader to
              use. Possible choices are 'FromJson', 'FromRaw', 'FromPostCalc'
            file_locs -> file locations for data timeseries loading.
        Returns:
            None
        """

        self.meta_data = self.parse_meta_data(meta_data)
        self.partition_coeffs = partition_coeffs

    def parse_meta_data(
        self, meta_data: Union[dict, MetaData.aMetaData, str]
    ) -> MetaData.aMetaData:
        """
        Parse meta_info, which may be a string, dict or MetaData.

        Args:
            meta_info (str| dict | MetaData): MetaData describing the experiment.

        Returns:
            MetaData: meta_info wrapped in the MetaData wrapper.
        """
        meta_data = self.construct_metadata(meta_data)
        self.check_metadata_fields(meta_data)
        return meta_data

    def construct_metadata(self, meta_info) -> MetaData.aMetaData:
        """Turns meta_data,
        which could be string, dict or ConchingModel.Data.MetaData.MetaData
        into MetaData."""
        if isinstance(meta_info, str):
            meta_data = self.read_json(meta_info)
        elif isinstance(meta_info, dict):
            meta_data = MetaData.aMetaData(meta_info)
        elif isinstance(meta_info, MetaData.aMetaData):
            meta_data = meta_info
        else:
            raise TypeError("No common type for meta_data found")
        return meta_data

    def read_json(self, loc: str) -> MetaData.aMetaData:
        "Read the json if a string was passed as file loc."
        if loc.split(".")[-1] == "json":
            with open(loc, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = MetaData.aMetaData(data)
        else:
            raise ValueError("metadata file is not json format")
        return result

    def check_metadata_fields(self, meta_data: MetaData.aMetaData):
        "Check if meta_data contains all required fields for analysis."
        required_fields = ("mass", "fractions", "sampling_times", "temp", "substance")
        is_complete = all([hasattr(meta_data, field) for field in required_fields])
        if not is_complete:
            raise ValueError("Not all required fields in meta_info")

    def add_timeseries(
        self, data_loader: DataLoaders.BaseLoader
    ) -> Timeseries.aTimeseries:
        "Add timeseries data from a data_loader to the meta_data."
        timeseries = data_loader.get_concentration()
        return timeseries

    def pick_partition_coefficients(self) -> list[float]:
        "Pick the right partition coefficients from the part coeff lib."
        part_coeffs = [
            part_coeff_lib.part_coeffs[phase][self.meta_data.substance.lower()]
            for phase in self.meta_data.phases
            if phase in ("cp", "sp")
        ]
        return part_coeffs

    def estimate_kinetic_parameters(
        self, bounds: list[list[float]]
    ) -> tuple[OptimizeResult, plt.Figure]:
        "Wrapper for handling parameter estimation"
        # Setup
        partition_coeffs = self.pick_partition_coefficients()
        # run optimization
        result = self.meta_data.spawn_optimizer().run_global_optimization(
            bounds, partition_coeffs
        )
        fig = self.meta_data.spawn_optimizer().make_comparison_plot(
            self.meta_data.substance, result.x, self.meta_data.phasetypes
        )
        return result, fig

    def run_analysis_suite(
        self, analysis_types: list[AnalysisLib.Analysis]
    ) -> tuple[dict]:
        "Run all analyses included in Analysis Types."
        result = tuple(map(lambda analysis: analysis.run_analysis(), analysis_types))
        return result

    def run_and_save_analysis_suite(
        self, analysis_types: list[AnalysisLib.Analysis], out_path: str
    ) -> tuple[dict]:
        "Run and save at the same time"
        result = tuple(
            map(lambda analysis: analysis.run_and_save(out_path), analysis_types)
        )
        return result

    def save_analysis_suite_results(
        self, analysis_types: list[AnalysisLib.Analysis], out_path: str
    ) -> tuple[dict]:
        "Separate save after analyses were run"
        result = tuple(map(lambda analysis: analysis.save(out_path), analysis_types))
        return result


if __name__ == "__main__":
    pass
