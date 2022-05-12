""" Extract timeseries from buhler data from IVV analysis. """
import ConchingModel.Data.Timeseries as Timeseries
import pandas as pd

DATACOL = 7
LABEL_CONVERSION = {
    "Essigsäure": "acetic_acid",
    "Benzaldehyd": "benzaldehyd",
    "Linalool": "linalool",
    "TMP": "tmp",
    "Phenylethanol": "phenylethanol",
    "acetic_acid": "Essigsäure",
    "benzaldehyd": "Benzaldehyd",
    "linalool": "Linalool",
    "tmp": "TMP",
    "phenylethanol": "Phenylethanol",
}


def get_aTimeseries(
    phases: list[Timeseries.aPhase], meta_data: dict
) -> Timeseries.aTimeseries:
    "Create Timeseries data from Timeseries.aPhase list."
    return Timeseries.aTimeseries(phases, meta_data["sampling_times"])


def get_aPhase(
    meta_data: dict, file: str, label: Timeseries.PhaseLabel, sheets: list[int]
) -> Timeseries.aPhase:
    """
    Create Timeseries.aPhase from a buhler analysis sheet.

    Args:
        meta_data (dict): MetaData dict for the json containing sampling_times.
        substance (str): Substance string to extract from the excel file.
        file (str): location of the input file. May be phase specific.
        label (Timeseries.PhaseLabel): Enum describing the kind.
        sheets (list[int]): which sheets to index in the excel file.

    Returns:
        [Timeseries.aPhase]: All observations for a single phase during one experiment.
    """
    filtered_sheets = get_filtered_sheets(
        file, LABEL_CONVERSION[meta_data["substance"]], sheets
    )
    samples = [
        create_aSample(sheet, time, label)
        for sheet, time in zip(filtered_sheets, meta_data["sampling_times"])
    ]
    result = Timeseries.aPhase(label, meta_data["sampling_times"], samples)
    return result


def get_filtered_sheets(loc: str, substance: str, sheets: list[int]) -> list[pd.Series]:
    "Filter the sheets to contain only substance relevant data."
    file = pd.read_excel(loc, sheet_name=sheets, header=2, index_col=0)
    result = [sheet.loc[substance]["c (A) [µg/g]"] for sheet in file.values()]
    return result


def create_aSample(sheet: pd.Series, time: float, label: Timeseries.PhaseLabel):
    "Create Timeseries.aSample from sheet at time using label."
    result = Timeseries.aSample(label, time, sheet.to_list())
    return result
