import ConchingModel.Data.MetaData as MetaData
import pytest
from ConchingModel import AnalysisLib, ConcheClass, OptimizerClass


@pytest.fixture
def meta_data():
    result = MetaData.aMetaData(
        {
            "mass": 200,
            "temp": 60,
            "fractions": {
                "COCOABUTTER": 0.3,
                "COCOAPARTICLE": 0.4,
                "SUGARPARTICLE": 0.3,
            },
            "sampling_times": [0, 1, 2, 3, 4, 5, 6],
            "timeseries": {"phases": {}, "sampling_times": []},
            "substance": "benzaldehyd",
        }
    )
    return result


@pytest.fixture
def analyzer_default(meta_data):
    kwargs = {"Analysis": {}}
    analyzer = AnalysisLib.Analysis(meta_data, kwargs)
    return analyzer


def test_analyzer_setup(analyzer_default):
    assert analyzer_default


def test_analyzer_spawn_conche(analyzer_default):
    assert isinstance(analyzer_default.setup_conche(), ConcheClass.ConcheClass)


def test_analyzer_spawn_optimzier(analyzer_default):
    with pytest.raises(KeyError):
        analyzer_default.setup_optimizer()


def test_analyzer_setup_targets_fails(analyzer_default):
    with pytest.raises(KeyError):
        analyzer_default.setup_optimization_targets()


def test_analyzer_setup_targets_works(analyzer_default):
    analyzer_default.kwargs["targets"] = ["MASSTRANSPORTCOEFFICIENTS"]
    analyzer_default.kwargs["bounds"] = [
        [0.0, 1.0],
    ]
    assert isinstance(
        analyzer_default.setup_optimization_targets(), OptimizerClass.TargetHandler
    )


def test_analyzer_setup_optimizer_works(analyzer_default):
    analyzer_default.kwargs["targets"] = ["MASSTRANSPORTCOEFFICIENTS"]
    analyzer_default.kwargs["bounds"] = [
        [0.0, 1.0],
    ]
    assert isinstance(
        analyzer_default.setup_optimizer(), OptimizerClass.GlobalOptimizer
    )
