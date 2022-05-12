import ConchingModel.Data.MetaData as MetaData
import pytest


@pytest.fixture
def meta_in_default():
    meta_info = {
        "mass": 200,
        "sampling_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "fractions": {
            "cb": 0.4,
            "sp": 0.3,
            "cp": 0.3,
        },
        "substance": "benzaldehyd",
        "temp": 60,
    }
    return meta_info


@pytest.fixture
def data_in_default(meta_in_default):
    data = MetaData.aMetaData(meta_in_default)
    return data


@pytest.mark.parametrize(
    "fractions, expected",
    [
        (
            {"cb": 0.4, "cp": 0.3, "sp": 0.3},
            ["continuous", "discontinuous", "discontinuous"],
        ),
        ({"cb": 0.4, "cp": 0.3, "sp": 0.0}, ["continuous", "discontinuous"]),
        ({"cb": 0.4, "cp": 0.0, "sp": 0.3}, ["continuous", "discontinuous"]),
    ],
)
def test_correct_phasetypes(fractions, expected, data_in_default):
    data_in_default._meta_info["fractions"] = fractions
    assert data_in_default.phasetypes == expected


@pytest.mark.parametrize(
    "fractions, expected",
    [
        ({"cb": 0.4, "cp": 0.3, "sp": 0.3}, ["cb", "cp", "sp"]),
        ({"cb": 0.4, "cp": 0.3, "sp": 0.0}, ["cb", "cp"]),
        ({"cb": 0.4, "cp": 0.0, "sp": 0.3}, ["cb", "sp"]),
    ],
)
def test_get_correct_phases(fractions, expected, data_in_default):
    data_in_default._meta_info["fractions"] = fractions
    assert data_in_default.phases == expected


def test_get_correct_masses(data_in_default):
    fractions = list(data_in_default.fractions.values())
    mass_fractions = list(
        map(lambda fraction: data_in_default.mass * fraction, fractions)
    )
    expected = [80, 60, 60]
    assert mass_fractions == expected
