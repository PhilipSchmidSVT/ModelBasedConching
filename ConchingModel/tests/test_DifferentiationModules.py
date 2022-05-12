import ConchingModel.Data.MetaData as MetaData
import numpy as np
import pytest
from ConchingModel import ConcheClass, DifferentiationModules


@pytest.fixture
def conche():
    meta_info = {
        "mass": 500,
        "fractions": {"cb": 0.4, "cp": 0.3, "sp": 0.3},
        "sampling_times": [0, 1, 2, 3, 4, 5],
        "substance": "benzaldehyd",
    }
    meta_data = MetaData.aMetaData(meta_info)
    conche = ConcheClass.ConcheClass(meta_data)
    return conche


@pytest.fixture
def c_step_diff(conche):
    c_step_diff = DifferentiationModules.Complex_Step_Differentiation(
        conche.run_sim, N=np.array([0.0, 1.0, 0.0], dtype=np.complex128), K=[0.02, 10]
    )
    return c_step_diff


def test_cstep_evaluate_function(c_step_diff):
    mock_params = [0.1, 0.1, 0.1]
    t = [1.0]
    res = c_step_diff.evaluate_function_at_time(mock_params, t)
    assert type(res) is np.ndarray


def test_cstep_perturb_element(c_step_diff):
    h = 1e-8
    mock_params = [0.1, 0.1, 0.1]
    perturbed_params = c_step_diff.perturb_param_set(mock_params, h)
    assert all(p_params != mock_params for p_params in perturbed_params)


def test_cstep_get_parameter_sensitivity_at(c_step_diff):
    t = [1.0]
    mock_params = [1e-3, 1e-3, 1e-3]
    h = 1e-8
    dFdp = c_step_diff.get_parameter_sensitivity_at(t, mock_params, h)
    assert type(dFdp) is np.ndarray


def test_cstep_get_parameter_sensitivity_at_times(c_step_diff):
    t = [1.0, 2.0, 3.0, 4.0, 5.0, 24.0]
    mock_params = [1e-6, 1e-6, 1e-4]
    h = 1e-16
    sensitivities = c_step_diff.get_parameter_sensitivity_at_times(t, mock_params, h)
    assert type(sensitivities) is np.ndarray


def test_cstep_change_in_sensitivity(c_step_diff):
    t = [1.0]
    h = 1e-16
    params = [[1e-6, 1e-6, 1e-3], [1, 1e-5, 1e-3]]
    sensitivity = [c_step_diff.get_parameter_sensitivity_at(t, p, h) for p in params]
    assert np.any(np.not_equal(sensitivity[0], sensitivity[1]))


def test_cstep_dimensions(c_step_diff):
    mock_params = [1e-6, 1e-6, 1e-3]
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = 1e-16
    sensitivity = c_step_diff.get_parameter_sensitivity_at_times(t, mock_params, h)
    assert sensitivity.shape == (5, 3, 3)


@pytest.fixture
def central_diff(conche):
    central_diff = DifferentiationModules.Centered_Difference(
        conche.run_sim, N=np.array([0.0, 1.0, 0.0]), K=[0.02, 10]
    )
    return central_diff


def test_central_diff_get_sensitivity_at(central_diff):
    mock_params = [1e-3, 1e-3, 1e-3]
    t = [1.0]
    h = 1e-8
    sens = central_diff.get_parameter_sensitivity_at(t, mock_params, h)
    assert type(sens) is np.ndarray


def test_central_diff_get_sensitivity_at_times(central_diff):
    mock_params = [1e-3, 1e-3, 1e-3]
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = 1e-8
    sens = central_diff.get_parameter_sensitivity_at_times(t, mock_params, h)
    assert type(sens) is np.ndarray
