""" Framework to load datasets from the analysis results """
import json

import ConchingModel.Data.MetaData as MetaData
import numpy as np
import pandas as pd


class BaseLoader:
    "This class is a parent to every subclass and provides interface function"

    def get_concentration(self) -> np.ndarray:
        return np.asarray(self.collect_concentration(self.timeseries))

    def get_phases(self) -> list:
        nonzero_phases = [
            phase
            for phase, fraction in self.data.fractions.items()
            if fraction is not None and fraction != 0
        ]
        return nonzero_phases

    def get_samplingtimes(self) -> list:
        return self.data.sampling_times

    def get_datatypes(self) -> list:
        converter = {
            "cb": "continuous",
            "cp": "discontinuous",
            "sp": "discontinuous",
            "pp": "dicsontinuous",
        }
        if all([phase in converter.keys() for phase in self.get_phases()]):
            phasetypes = [converter.get(phase) for phase in self.phases()]
        else:
            raise ValueError(f"Phase not in {converter.keys()}")
        return phasetypes

    def collect_concentration(self, data: pd.Series) -> list:
        return self.timeseries

    def to_json(
        self,
        out_path: str,
        substance: str,
        mass: float,
        fractions: dict,
        temp: float,
        sampling_times: list,
    ):
        "Write meta_info to json file"
        out_dict = {
            "mass": mass,
            "fractions": fractions,
            "substance": substance,
            "timeseries": self.get_concentration(),
            "temp": temp,
            "sampling_times": sampling_times,
        }
        pd.Series(out_dict).to_json(out_path, orient="index")
        pass


class FromJson(BaseLoader):
    """Read timeseries and meta_info from json"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_json(file_path)

    def read_json(self, file_path):
        with open(file_path) as f:
            result = MetaData.aMetaData(json.load(f))
        return result


class FromRaw(BaseLoader):
    """
    Loading data from raw analysis data xlsx.
    Basically only used in old Buhler data
    """

    def __init__(self, meta_data, *file_loc: str) -> None:
        """Initialize the object. Supply the location of the file containing
        phase specific analysis data."""

        self.timeseries = []
        self.data = meta_data
        for file_ in file_loc:
            self.analysis_df, self.calibration = self.get_analysis_data(file_)
            self.timeseries.append(self.calculate_concentration(self.data.substance))

    def get_analysis_data(self, file_loc: str) -> tuple:
        """Load data from an excel file
        Parameters:
            file_loc -> str: Path to the file containing single phase data
        Returns:
            analysis_df -> pd.DataFrame: Dataframe containing the raw
                data from excel file.
        """
        # This creates a dict with sheet names as keys
        analysis_df = pd.read_excel(file_loc, sheet_name=None, header=2, index_col=0)
        analysis_df = list(analysis_df.values())
        # First sheet is always calibration
        calibration = analysis_df.pop(0)
        # Clean data of any empty cols and rows
        analysis_df = [df.dropna(axis=0, how="all") for df in analysis_df]
        analysis_df = [df.dropna(axis=1, how="all") for df in analysis_df]
        calibration = self.clean_calibration(calibration)
        calibration = self.modify_calibration(calibration)
        # Change Linalool-xx to Linalool
        index_list = calibration.index.tolist()
        new_index_list = [
            index if not index.startswith("Linalool") else "Linalool"
            for index in index_list
        ]
        calibration = calibration.set_index(pd.Index(new_index_list), drop=True)
        calibration = calibration.dropna(axis=0, how="all")
        # Return analysis data as having substances as indices
        analysis_df = [df.T for df in analysis_df]
        return analysis_df, calibration

    def create_individual_calibration_df(self, df):
        """
        Pick the correct calibration values and form a specific
        calibration dataframe from them.
        """

        def pick_cal_row(substance):
            a_ratio = (
                df["Intensität (Analyt)"].loc[substance]
                / df["Intensität (STD)"].loc[substance]
            )
            subs_rows = self.calibration.loc[[substance]]
            if subs_rows.shape[0] == 1:
                return subs_rows
            else:
                row = subs_rows[
                    (subs_rows["a_min"] <= a_ratio) & (subs_rows["a_max"] >= a_ratio)
                ]
                return row

        substance_list = df.index.to_list()
        cal_rows = tuple(map(pick_cal_row, substance_list))
        cal_df = pd.concat(cal_rows, axis=0)
        return cal_df

    def calc_c(self, df: pd.DataFrame) -> pd.DataFrame:
        curr_cal = self.create_individual_calibration_df(df)
        conc = (
            ((df["Intensität (Analyt)"] / df["Intensität (STD)"]) - curr_cal["t"])
            / curr_cal["m"]
            * (df["m (STD, µg)"] / df["m (Probe, mg)"])
            * 1000
        )
        return conc

    def calculate_concentration(self, substance):
        """Calculate the concentration at every timestep for every
        Substance as mg/kg, aka parts per million
        # Calculate concentration from analysis data
        # c = ((I_Analyt/I_ISTD) - t)/m * (m_ISTD/EW) * 1000
        # c in [µg/kg]
        # m_ISTD in [µg]
        # EW in [g]
        """
        calculated_concentrations = list(
            map(lambda st: self.calc_c(st).loc[substance], self.analysis_df)
        )
        return calculated_concentrations

    def clean_calibration(self, raw_calibration: pd.DataFrame) -> pd.DataFrame:
        """Clean the calibration data dataframe"""
        header_row = 0
        correct_headers = raw_calibration.iloc[header_row].tolist()
        deprecated_calibration = raw_calibration.iloc[header_row + 1 :]
        deprecated_calibration.columns = correct_headers
        deprecated_calibration = deprecated_calibration.dropna(axis=0, how="any")
        calibration = deprecated_calibration
        return calibration

    def modify_calibration(self, calibration: pd.DataFrame) -> pd.DataFrame:
        """
        Calibration can be of different formats. Some calibration frames
        contain multiple calibration curves for the same substance. This
        function reconciles this and extends calibration data
        """
        # Check if calibration has multiple calibration curves for a single
        # Substance.
        if "verwendet für A (Analyt)/A(ISTD)" in calibration.columns:
            # If this row exists, it is a calibration from Franziska.
            # Treat differently from Helen's calibration.
            calibration = calibration.reset_index().fillna(method="ffill")
            calibration = calibration.set_index("index")
            a_ratio_list = calibration["verwendet für A (Analyt)/A(ISTD)"].to_list()
            a_ratio_min_max = [item.split("-") for item in a_ratio_list]
            a_min, a_max = zip(*a_ratio_min_max)
            a_min = tuple(map(lambda item: item.replace(",", "."), a_min))
            a_max = tuple(map(lambda item: item.replace(",", "."), a_max))
            a_min = tuple(map(float, a_min))
            a_max = tuple(map(float, a_max))
            calibration["a_min"] = a_min
            calibration["a_max"] = a_max
        return calibration


class FromArray(BaseLoader):
    """Set a given array to be the timeseries."""

    def __init__(self, timeseries):
        if type(timeseries) != np.ndarray:
            raise TypeError
        self.timeseries = timeseries


class FromPostCalc(BaseLoader):
    "Loading already calculated concentration data"
    substance_labels = ["essigsäure", "tmp", "benzaldehyd", "linalool", "phenylethanol"]

    def __init__(self, meta_data, *file_locs):
        self.data = meta_data
        self.timeseries = list(map(self.read_file, file_locs))

    def read_file(self, file_in) -> list:
        if file_in.split(".")[-1] == "xlsx":
            file_data = pd.read_excel(
                file_in, sheet_name=0, header=None, index_col=None, dtype=np.float64
            )
            index = pd.Index(self.substance_labels)
            file_data = file_data.set_index(index)
            data_out = file_data.loc[self.data.substance].to_list()
        elif file_in.split(".")[-1] == "csv":
            file_data = pd.read_csv(file_in, delimiter="\t")
            data_out = file_data.iloc[0, :].to_list()
        else:
            raise TypeError
        return data_out


class FromCSV(BaseLoader):
    def __init__(self, *file_locs):
        self.timeseries = [
            np.fromfile(file_loc, dtype=np.float64, sep="\t") for file_loc in file_locs
        ]


if __name__ == "__main__":
    pass
