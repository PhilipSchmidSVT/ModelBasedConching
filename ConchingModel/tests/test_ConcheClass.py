""" Testing environment for functions of the ConcheClass objects """
import ConchingModel.Data.MetaData as MetaData
import numpy as np
import pytest
from ConchingModel import ConcheClass

MTC = [0.5, 0.3]
PART_COEFF = {
    "Essigsäure": [1],
}
MASS = [200 * 0.4, 200 * 0.3, 200 * 0.3]


@pytest.fixture
def meta_info():
    meta_info = {
        "mass": 1,
        "sampling_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "fractions": {
            "cb": 0.33,
            "cp": 0.33,
            "sp": 0.33,
        },
        "substance": "benzaldehyd",
    }
    return MetaData.aMetaData(meta_info)


@pytest.fixture
def conche(meta_info):
    conche = ConcheClass.ConcheClass(meta_info)
    return conche


@pytest.fixture
def unobservable_conche(meta_info):
    conche = ConcheClass.ConcheClassCombinedParticles(meta_info)
    return conche


@pytest.mark.parametrize(
    "mtc, expected",
    [
        ((1, 1, 0), [0, 0, 0]),
        ((1, 1, 1), [-1, 0, 0]),
        ((0, 0, 0), [0, 0, 0]),
        ((0, 0, 1), [-1, 0, 0]),
    ],
)
def test_construct_dNdt_equal_starting_vals(mtc, expected, conche):
    N0 = [1.0, 1.0, 1.0]
    K = [1.0, 1.0]
    dNdt = conche.construct_dNdt(1, N0, mtc, K)
    np.testing.assert_almost_equal(dNdt, np.array(expected))


@pytest.mark.parametrize(
    "N0, expected",
    [
        ([0, 0, 0], [0, 0, 0]),
    ],
)
def test_construct_dNdt_varying_starting_vals(N0, expected, conche):
    K = [1.0, 1.0]
    mtc = [1, 1, 1]
    dNdt = conche.construct_dNdt(1, N0, mtc, K)
    np.testing.assert_almost_equal(dNdt, np.array(expected))


def test_forward_sim(conche):
    N0 = np.array([0.0, 1.0, 0.0])
    sampling_times = conche.meta_data.sampling_times
    mtc = [1.0, 0.0, 0.0]
    K = [1.0, 1.0]
    sim_result = conche.run_sim(sampling_times, N0, mtc, K)
    assert sim_result.shape == (len(N0), len(sampling_times))


def test_phase_reduction(unobservable_conche):
    meta_data = MetaData.aMetaData(
        {
            "sampling_times": (0, 1, 2, 3, 4, 5),
            "mass": 500,
            "fractions": {"cb": 0.3, "cp": 0.4, "sp": 0.3},
            "substance": "benzaldehyd",
            "temp": 60,
        }
    )
    conche = ConcheClass.ConcheClassCombinedParticles(meta_data)
    K = (0.2, 1.0)
    mtc = (2, 1.3, 0)
    c0 = np.array((0.033, 0.1, 0.033))
    result = conche.run_sim(unobservable_conche.meta_data.sampling_times, c0, mtc, K)
    np.testing.assert_almost_equal((0.033, 0.071), result[:, 0], decimal=3)


def test_Conche_CB_only():
    meta_data = MetaData.aMetaData(
        {
            "mass": 500,
            "fractions": {"COCOABUTTER": 1.0, "COCOAPARTICLE": 0, "SUGARPARTICLE": 0},
            "temp": 60,
            "sampling_times": [0.0, 1.0, 2.0],
            "substance": "benzaldehyd",
        }
    )
    conche = ConcheClass.ConcheClass(meta_data)
    dcdt = conche.construct_dNdt(
        0,
        np.array(
            [
                1,
            ]
        ),
        np.array(
            [
                0.3,
            ]
        ),
        [
            1,
        ],
    )
    assert len(dcdt) == 1


def test_meta_data_not_valid():
    meta_data = {"mass": 500, "substance": "benzaldehyd"}
    with pytest.raises(TypeError):
        ConcheClass.ConcheClass(meta_data)


def test_check_required_fields_succeeds():
    meta_data = MetaData.aMetaData({"mass": 500, "substance": "benzaldehyd"})
    conche = ConcheClass.ConcheClass(meta_data)
    assert conche.check_required_fields(meta_data)


def test_check_required_fields_fails():
    meta_data = MetaData.aMetaData(
        {
            "mass": 500,
        }
    )
    with pytest.raises(KeyError):
        ConcheClass.ConcheClass.check_required_fields(meta_data)
