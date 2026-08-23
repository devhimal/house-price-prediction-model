import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.ode_solver import ODESolver


@pytest.fixture
def solver():
    return ODESolver()


class TestRungeKutta4:
    def test_rk4_exponential_growth(self, solver):
        f = lambda t, y: y
        result = solver.runge_kutta_4(f, (0.0, 1.0), 1.0, n_steps=100)
        y_final = result["y_values"][-1]
        assert np.isclose(y_final, np.exp(1.0), atol=1e-4)

    def test_rk4_constant_slope(self, solver):
        f = lambda t, y: 2.0
        result = solver.runge_kutta_4(f, (0.0, 3.0), 0.0, n_steps=50)
        y_final = result["y_values"][-1]
        assert np.isclose(y_final, 6.0, atol=1e-6)

    def test_rk4_sine_solution(self, solver):
        f = lambda t, y: np.cos(t)
        result = solver.runge_kutta_4(f, (0.0, np.pi / 2), 0.0, n_steps=200)
        y_final = result["y_values"][-1]
        assert np.isclose(y_final, 1.0, atol=1e-4)

    def test_rk4_decay(self, solver):
        f = lambda t, y: -2 * y
        result = solver.runge_kutta_4(f, (0.0, 1.0), 1.0, n_steps=200)
        y_final = result["y_values"][-1]
        exact = np.exp(-2.0)
        assert np.isclose(y_final, exact, atol=1e-4)

    def test_rk4_returns_metadata(self, solver):
        f = lambda t, y: y
        result = solver.runge_kutta_4(f, (0.0, 1.0), 1.0, n_steps=100)
        assert result["method"] == "Runge-Kutta 4"
        assert result["n_steps"] == 100
        assert len(result["t_values"]) == 101
        assert len(result["y_values"]) == 101

    def test_rk4_linear_ode(self, solver):
        f = lambda t, y: 1.0
        result = solver.runge_kutta_4(f, (0.0, 5.0), 0.0, n_steps=100)
        exact = 5.0
        assert np.isclose(result["y_values"][-1], exact, atol=1e-6)

    def test_rk4_initial_value(self, solver):
        f = lambda t, y: y
        result = solver.runge_kutta_4(f, (0.0, 1.0), 3.0, n_steps=100)
        assert np.isclose(result["y_values"][0], 3.0, atol=1e-10)


class TestAdamsBashforth4:
    def test_ab4_exponential_growth(self, solver):
        f = lambda t, y: y
        result = solver.adams_bashforth_4(f, (0.0, 1.0), 1.0, n_steps=500)
        y_final = result["y_values"][-1]
        assert np.isclose(y_final, np.exp(1.0), atol=1e-2)

    def test_ab4_constant_slope(self, solver):
        f = lambda t, y: 2.0
        result = solver.adams_bashforth_4(f, (0.0, 3.0), 0.0, n_steps=100)
        y_final = result["y_values"][-1]
        assert np.isclose(y_final, 6.0, atol=1e-4)

    def test_ab4_decay(self, solver):
        f = lambda t, y: -2 * y
        result = solver.adams_bashforth_4(f, (0.0, 1.0), 1.0, n_steps=500)
        y_final = result["y_values"][-1]
        exact = np.exp(-2.0)
        assert np.isclose(y_final, exact, atol=1e-2)

    def test_ab4_returns_metadata(self, solver):
        f = lambda t, y: y
        result = solver.adams_bashforth_4(f, (0.0, 1.0), 1.0, n_steps=100)
        assert result["method"] == "Adams-Bashforth 4"
        assert len(result["t_values"]) == 101

    def test_ab4_initial_value(self, solver):
        f = lambda t, y: 0.0
        result = solver.adams_bashforth_4(f, (0.0, 1.0), 5.0, n_steps=100)
        assert np.isclose(result["y_values"][0], 5.0, atol=1e-10)


class TestCompareMethods:
    def test_compare_returns_both(self, solver):
        f = lambda t, y: y
        result = solver.compare_methods(f, (0.0, 1.0), 1.0, n_steps=100)
        assert "rk4" in result
        assert "adams_bashforth_4" in result

    def test_compare_with_exact(self, solver):
        f = lambda t, y: y
        exact = lambda t: np.exp(t)
        result = solver.compare_methods(f, (0.0, 1.0), 1.0, n_steps=200,
                                        exact_solution=exact)
        assert "rk4_max_error" in result
        assert "ab4_max_error" in result
        assert result["rk4_max_error"] < 0.01

    def test_compare_rk4_more_accurate(self, solver):
        f = lambda t, y: y
        exact = lambda t: np.exp(t)
        result = solver.compare_methods(f, (0.0, 1.0), 1.0, n_steps=100,
                                        exact_solution=exact)
        assert result["rk4_max_error"] < result["ab4_max_error"]
