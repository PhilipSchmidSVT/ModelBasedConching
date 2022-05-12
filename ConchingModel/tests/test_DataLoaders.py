" Test file for DataLoaders "
import os

import ConchingModel.Data.MetaData as MetaData
import pytest
from ConchingModel import DataLoaders
from ConchingModel.Experiment import Experiment

os.chdir(os.path.dirname(__file__))
# Globals


@pytest.fixture()
def FromJson():
    file_path = "data/FromJson/test.json"
    FromJson = DataLoaders.FromJson(file_path)
    return FromJson


# FromJson


def test_FromJson_load_json(FromJson):
    data = FromJson.data
    assert type(data) == MetaData.aMetaData


def test_FromJson_required_fields(FromJson):
    required_fields = ["mass", "temp", "fractions", "sampling_times"]
    data = FromJson.data
    assert all([hasattr(data, attr) for attr in required_fields])


def test_FromJson_get_samplingtimes(FromJson):
    sampling_times = FromJson.get_samplingtimes()
    assert sampling_times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_FromJson_get_nonzero_phases(FromJson):
    nonzero_phases = FromJson.get_phases()
    assert len(nonzero_phases) == 3


# FromRaw


@pytest.fixture()
def FromRaw():
    cb = "data/FromRaw/cb.xlsx"
    cp = "data/FromRaw/pp.xlsx"
    meta_info = {
        "mass": 200000,
        "sampling_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "fractions": {"cb": 0.29, "cp": 0.38, "sp": 0.33},
        "temp": 60,
        "substance": "Benzaldehyd",
    }
    meta_data = MetaData.aMetaData(meta_info)
    FromRaw = DataLoaders.FromRaw(meta_data, cb, cp)
    return FromRaw


def test_FromRaw_number_of_phases(FromRaw):
    expected_num_of_phases = 2
    assert len(FromRaw.get_concentration()) == expected_num_of_phases


def test_FromRaw_right_phases(FromRaw):
    expected_phases = ["cb", "cp", "sp"]
    phases = FromRaw.get_phases()
    assert all([phase in expected_phases for phase in phases])


# PostCalc


@pytest.fixture()
def FromPostCalc():
    cb = "data/FromPostCalc/cb.xlsx"
    cp = "data/FromPostCalc/cp.xlsx"
    meta_info = {
        "mass": 500,
        "sampling_times": [0.0, 0.5, 1.0, 1.5, 2.0],
        "fractions": {"cb": 0.38, "cp": 0.62},
        "temp": 60,
        "substance": "benzaldehyd",
    }
    meta_data = MetaData.aMetaData(meta_info)
    FromPostCalc = DataLoaders.FromPostCalc(meta_data, cb, cp)
    return FromPostCalc


def test_FromPostCalc_num_of_phases(FromPostCalc):
    phases_concentration = FromPostCalc.get_concentration()
    expected_num_of_phases = 2
    assert expected_num_of_phases == len(phases_concentration)


def test_FromPostCalc_right_phases(FromPostCalc):
    phases = FromPostCalc.get_phases()
    expected = ["cb", "cp"]
    assert all([phase in expected for phase in phases])
