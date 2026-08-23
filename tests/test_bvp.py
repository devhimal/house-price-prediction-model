import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.bvp_solver import BVPSolver


@pytest.fixture
def solver():
    return BVPSolver()


class TestFiniteDifference:
    def test_fd_constant_coefficients(self, solver):
        p = lambda x: 0.0
        q = lambda x: 0.0
        r = lambda x: -2.0
        result = solver.finite_difference(p, q, r, a=0.0, b=1.0,
                                          alpha=0.0, beta=0.0, n=50)
        x_vals = result["x_values"]
        y_vals = result["y_values"]
        expected = x_vals * (1.0 - x_vals)
        assert np.isclose(y_vals, expected, atol=1e-2).all()

    def test_fd_boundary_conditions(self, solver):
        p = lambda x: 0.0
        q = lambda x: 0.0
        r = lambda x: 0.0
        result = solver.finite_difference(p, q, r, a=0.0, b=1.0,
                                          alpha=1.0, beta=2.0, n=20)
        assert np.isclose(result["y_values"][0], 1.0, atol=1e-10)
        assert np.isclose(result["y_values"][-1], 2.0, atol=1e-10)

    def test_fd_returns_metadata(self, solver):
        p = lambda x: 0.0
        q = lambda x: 0.0
        r = lambda x: 1.0
        result = solver.finite_difference(p, q, r, a=0.0, b=1.0,
                                          alpha=0.0, beta=0.0, n=10)
        assert result["method"] == "finite_difference"
        assert result["n_points"] == 11
        assert len(result["x_values"]) == 11
        assert len(result["y_values"]) == 11

    def test_fd_simple_harmonic(self, solver):
        p = lambda x: 0.0
        q = lambda x: 1.0
        r = lambda x: 0.0
        result = solver.finite_difference(p, q, r, a=0.0, b=np.pi / 2,
                                          alpha=0.0, beta=1.0, n=100)
        x_vals = result["x_values"]
        expected = np.sin(x_vals)
        assert np.isclose(result["y_values"], expected, atol=0.05).all()

    def test_fd_with_linear_source(self, solver):
        p = lambda x: 0.0
        q = lambda x: 0.0
        r = lambda x: 6 * x
        result = solver.finite_difference(p, q, r, a=0.0, b=1.0,
                                          alpha=0.0, beta=1.0, n=100)
        x_vals = result["x_values"]
        expected = x_vals**3
        assert np.isclose(result["y_values"], expected, atol=0.05).all()


class TestShootingMethod:
    def test_shooting_linear_ode(self, solver):
        f = lambda x, y, dy: 0.0
        result = solver.shooting_method(f, a=0.0, b=1.0,
                                        alpha=0.0, beta=1.0, n_steps=200)
        x_vals = result["x_values"]
        expected = x_vals
        assert np.isclose(result["y_values"], expected, atol=0.05).all()

    def test_shooting_boundary_conditions(self, solver):
        f = lambda x, y, dy: -y
        result = solver.shooting_method(f, a=0.0, b=np.pi,
                                        alpha=0.0, beta=0.0, n_steps=200)
        assert np.isclose(result["y_values"][0], 0.0, atol=1e-6)
        assert np.isclose(result["y_values"][-1], 0.0, atol=0.1)

    def test_shooting_returns_metadata(self, solver):
        f = lambda x, y, dy: 0.0
        result = solver.shooting_method(f, a=0.0, b=1.0,
                                        alpha=0.0, beta=1.0, n_steps=100)
        assert result["method"] == "shooting_method"
        assert "initial_slope" in result
        assert len(result["x_values"]) == 101

    def test_shooting_sine_solution(self, solver):
        f = lambda x, y, dy: -y
        result = solver.shooting_method(f, a=0.0, b=np.pi / 2,
                                        alpha=0.0, beta=1.0, n_steps=400)
        x_vals = result["x_values"]
        expected = np.sin(x_vals)
        assert np.isclose(result["y_values"], expected, atol=0.05).all()

    def test_shooting_initial_slope_linear(self, solver):
        f = lambda x, y, dy: 0.0
        result = solver.shooting_method(f, a=0.0, b=1.0,
                                        alpha=0.0, beta=2.0, n_steps=100)
        assert result["initial_slope"] is not None
        assert len(result["y_values"]) == 101
