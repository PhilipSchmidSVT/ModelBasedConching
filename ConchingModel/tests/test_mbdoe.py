""" Testing framework for the optimal experimental design module """
import ConchingModel.Data.MetaData as MetaData
import numpy as np
import pytest
from attr import dataclass
from ConchingModel import ConcheClass, DifferentiationModules, mbdoe


@pytest.fixture
def odoe3D():
    meta_info = {
        "mass": 500,
        "fractions": {"cb": 0.5, "cp": 0.3, "sp": 0.2},
        "sampling_times": [0, 1, 2, 3, 4, 5],
        "substance": "benzaldehyd",
    }
    meta_data = MetaData.aMetaData(meta_info)
    conche = ConcheClass.ConcheClass(meta_data)
    mock_params = [1e-6, 1e-5, 1e-3]
    N = np.array([0.3, 0.3, 0.01], dtype=np.complex128)
    phasetypes = ["continuous", "discontinuous", "discontinuous"]
    K = [0.01, 10]
    diff_mod = DifferentiationModules.Complex_Step_Differentiation(
        conche.run_sim, N=N, phasetypes=phasetypes, K=K
    )
    doe = mbdoe.SamplingTime(mock_params, diff_mod)
    return doe


@pytest.fixture
def odoe2D():
    meta_info = {
        "mass": 500,
        "fractions": {
            "cb": 0.5,
            "cp": 0.3,
            "sp": 0.2,
        },
        "sampling_times": [0, 1, 2, 3, 4, 5],
        "substance": "benzaldehyd",
    }
    meta_data = MetaData.aMetaData(meta_info)
    meta_data.is_combined_particle_phase = True
    conche = ConcheClass.ConcheClass(meta_data)
    mock_params = [1e-6, 1e-5, 1e-3]
    N = np.array([0.3, 0.3, 0.0], dtype=np.complex128)
    K = [0.01, 10]
    diff_mod = DifferentiationModules.Complex_Step_Differentiation(
        conche.run_sim, N=N, K=K
    )
    doe = mbdoe.SamplingTime(mock_params, diff_mod)
    return doe


def test_get_var_covar_mat_shape(odoe3D):
    t_s = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = 1e-16
    var_covar_mat = odoe3D.get_var_covar_mat(t_s, h)
    assert var_covar_mat.shape == (3, 3)


def test_get_var_covar_mat_shape2D(odoe2D):
    t_s = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = 1e-16
    var_covar_mat = odoe2D.get_var_covar_mat(t_s, h)
    assert var_covar_mat.shape == (3, 3)


def test_get_var_covar_mat_is_symetric(odoe3D):
    "Variance - Covariance Matrices are supposed to be symetric"
    t_s = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = 1e-16
    var_covar_mat = odoe3D.get_var_covar_mat(t_s, h)
    is_symetric = np.allclose(var_covar_mat, var_covar_mat.T, rtol=1e-5)
    assert is_symetric is True


# Does it even have to be positive semidefinite? Cant find source
# def test_get_var_covar_mat_is_pos_semidef(odoe3D):
# """ Variance - Covariance matrices are supposed to be positive semidef """
# t_s = [1., 2., 3., 4., 5.]
# h = 1e-16
# var_covar_mat = odoe3D.get_var_covar_mat(t_s, h)
# is_symetric = np.allclose(var_covar_mat, var_covar_mat.T, rtol=1e-5)
# eigvals = np.linalg.eigvals(var_covar_mat)
# has_positive_eigvals = all(eigvals >= 0)
# assert all((is_symetric, has_positive_eigvals)) is True


def test_cost_fun_t_input_dependency(odoe3D):
    T = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]]
    h = 1e-16
    score_1 = odoe3D.cost_fun_t(T[0], h, mbdoe.ODOE.get_d_optimality)
    score_2 = odoe3D.cost_fun_t(T[1], h, mbdoe.ODOE.get_d_optimality)
    assert score_1 != pytest.approx(score_2, 0.1)


def test_score_fun_a(odoe3D):
    T = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]]
    h = 1e-16
    score_1 = odoe3D.cost_fun_t(T[0], h, mbdoe.ODOE.get_a_optimality)
    score_2 = odoe3D.cost_fun_t(T[1], h, mbdoe.ODOE.get_d_optimality)
    assert score_1 != score_2


def test_score_fun_e(odoe3D):
    T = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]]
    h = 1e-16
    score_1 = odoe3D.cost_fun_t(T[0], h, mbdoe.ODOE.get_e_optimality)
    score_2 = odoe3D.cost_fun_t(T[1], h, mbdoe.ODOE.get_d_optimality)
    assert score_1 != score_2


# def test_get_optimal_sampling_times(odoe3D):
# t_init = np.array([1., 2., 3., 4., 5.])
# h = 1e-16
# t_opt = odoe3D.optimize_samplingtimes(mbdoe.ODOE.get_d_optimality,
# h,
# t_init)
# assert type(t_opt) is not None


# def test_get_optimal_sampling_times_2D(odoe2D):
# t_init = np.array([1., 2., 3., 4., 5.])
# h = 1e-16
# t_opt = odoe2D.optimize_samplingtimes(mbdoe.ODOE.get_d_optimality,
# h,
# t_init)
# assert type(t_opt) is not None
