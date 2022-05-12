"Tests for Timeseries class."
import pytest
from ConchingModel.Data.Timeseries import PhaseLabel, aSample


@pytest.fixture
def aSample_default():
    "Fixture for a default sample."
    concentration = [0.0, 1.0, 2.0]
    label = PhaseLabel.COCOABUTTER
    sampling_time = 0
    return aSample(label, sampling_time, concentration)


def test_aSample_mean(aSample_default):
    "Check if mean is reproduced."
    assert aSample_default.c_mean == 1.0


def test_aSample_std(aSample_default):
    "Check if std is reproduced."
    assert aSample_default.c_std - 0.816 < 0.001


def test_aSample_outliers():
    "Check if outliers are correctly removed."
    sample = aSample(PhaseLabel.COCOABUTTER, 0, [0, 1, 2, 500])
    assert len(sample.no_outliers.concentration) < len(sample.concentration)


def test_aSample_draw_measurement_number(aSample_default):
    "Check if the right number of measurements is gotten."
    giant_sample = aSample_default.add_measurement(0.1, 1000)
    assert len(giant_sample.concentration) == 1003


def test_aSample_draw_measurement_vals(aSample_default):
    "Check if the mean value does not move much given large number of samples."
    giant_sample = aSample_default.add_measurement(0.1, 10000)
    assert -0.01 < giant_sample.c_mean - 1.0 < 0.01
