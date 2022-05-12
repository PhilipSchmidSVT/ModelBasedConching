import os

import ConchingModel.Data.MetaData as MetaData
import pytest
from ConchingModel.Experiment import Experiment

os.chdir(os.path.dirname(__file__))

# Data Entry


@pytest.fixture()
def exp_json():
    file_path = "data/FromJson/test.json"
    exp_json = Experiment(file_path)
    return exp_json


def test_json_parse_data_gives_dict(exp_json):
    assert (
        type(exp_json.parse_meta_data("data/FromJson/test.json")) == MetaData.aMetaData
    )


# Raw data tests will probably not work because they have never been parsed to MetaData
# @pytest.fixture()
# def exp_raw():
# cb = "data/FromRaw/cb.xlsx"
# pp = "data/FromRaw/pp.xlsx"
# meta_data = MetaData.MetaData(
# {
# "mass": 200000,
# "fractions": {"cb": 0.33, "cp": 0.67, "sp": 0.0},
# "sampling_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
# "temp": 60,
# "substance": "Benzaldehyd",
# }
# )
# meta_data.timeseries = DataLoaders.FromRaw(meta_data, cb, pp).timeseries
# exp_raw = Experiment(meta_data)
# return exp_raw


# def test_raw_parse_data_gives_dict(exp_raw):
# meta_data = {
# "mass": 200000,
# "fractions": {"cb": 0.33, "cp": 0.67, "sp": 0.0},
# "sampling_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
# "temp": 60,
# "substance": "Benzaldehyd",
# }
# assert isinstance(exp_raw.parse_meta_data(meta_data), MetaData.MetaData)


# def test_raw_timeseries_is_loaded(exp_raw):
# assert isinstance(exp_raw.meta_data.timeseries, Timeseries.Timeseries)


# def test_raw_correct_masses(exp_raw):
# expected = {"cb": 0.33 * 200000, "cp": 0.67 * 200000, "sp": 0}
# calculated = exp_raw.meta_data.phase_mass
# np.testing.assert_array_almost_equal(list(expected.values()), calculated)


# def test_raw_get_only_nonzero_phases(exp_raw):
# expected = ["cb", "cp"]
# phases = exp_raw.meta_data.phases
# assert expected == phases


# def test_raw_get_only_nonzero_phasetypes(exp_raw):
# expected = ["continuous", "discontinuous"]
# phasetypes = exp_raw.meta_data.phasetypes
# assert expected == phasetypes
