import pytest
from ConchingModel.Data import MetaData, Timeseries
from ConchingModel.OptimizerClass import GlobalOptimizer, TargetHandler, Targets


@pytest.fixture
def data():
    "Define fixed meta data for tests."
    info = {
        "mass": 200,
        "sampling_times": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "fractions": {
            "COCOABUTTER": 0.4,
            "COCOAPARTICLE": 0.3,
            "SUGARPARTICLE": 0.3,
        },
        "substance": "benzaldehyd",
        "temp": 60,
    }
    meta_data = MetaData.aMetaData(info)
    mtc = [0.3, 0.1, 0.2]
    N0 = [0.0, 1.0, 0.0]
    ts = Timeseries.aTimeseries.from_sim(
        meta_data,
        meta_data.spawn_conche().run_sim(
            meta_data.sampling_times, N0, mtc, meta_data.K
        ),
    )
    meta_data.timeseries = ts.__dict__
    return meta_data


@pytest.fixture
def targets():
    mtc = Targets.MASSTRANSPORTCOEFFICIENTS.value([[0, 1], [0, 1], [0, 1]])
    K = Targets.PARTITIONCOEFFICIENTS.value([[1e-3, 1e3], [1e-3, 1e3]])
    c0 = Targets.INITIALSTATE.value([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    result = TargetHandler([mtc, K, c0])
    return result


@pytest.fixture
def optimizer(data, targets):
    optimizer = GlobalOptimizer(data, data.timeseries, targets)
    return optimizer


def test_residual_zero_at_true_vals(optimizer, data):
    residuals = optimizer.residuals(
        [0.3, 0.1, 0.2, *data.K, 0.0, 1.0, 0.0], data.sampling_times
    )
    assert -1e-6 < residuals < 1e-6


def test_residual_nonzero_at_false_vals(optimizer, data):
    residuals = optimizer.residuals(
        [0.8, 0.1, 0.2, *data.K, 0.0, 1.0, 0.0], data.sampling_times
    )
    assert abs(residuals) > 1e-6
