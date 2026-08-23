import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.interpolation.interpolation_engine import InterpolationEngine


@pytest.fixture
def engine():
    return InterpolationEngine()


class TestLagrange:
    def test_lagrange_linear(self, engine):
        x_data = np.array([0.0, 1.0])
        y_data = np.array([1.0, 3.0])
        assert np.isclose(engine.lagrange(x_data, y_data, 0.5), 2.0, atol=1e-10)

    def test_lagrange_exact_points(self, engine):
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 4.0, 9.0])
        for xi, yi in zip(x_data, y_data):
            assert np.isclose(engine.lagrange(x_data, y_data, xi), yi, atol=1e-10)

    def test_lagrange_quadratic(self, engine):
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([0.0, 1.0, 4.0])
        assert np.isclose(engine.lagrange(x_data, y_data, 1.5), 2.25, atol=1e-10)

    def test_lagrange_cubic(self, engine):
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([0.0, 1.0, 8.0, 27.0])
        assert np.isclose(engine.lagrange(x_data, y_data, 1.5), 3.375, atol=1e-10)

    def test_lagrange_two_points(self, engine):
        x_data = np.array([0.0, 2.0])
        y_data = np.array([5.0, 11.0])
        assert np.isclose(engine.lagrange(x_data, y_data, 1.0), 8.0, atol=1e-10)


class TestNewtonDividedDifference:
    def test_newton_linear(self, engine):
        x_data = np.array([0.0, 1.0])
        y_data = np.array([1.0, 3.0])
        result = engine.newton_divided_difference(x_data, y_data, 0.5)
        assert np.isclose(result["value"], 2.0, atol=1e-10)

    def test_newton_exact_points(self, engine):
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 4.0, 9.0])
        for xi, yi in zip(x_data, y_data):
            result = engine.newton_divided_difference(x_data, y_data, xi)
            assert np.isclose(result["value"], yi, atol=1e-10)

    def test_newton_returns_coefficients(self, engine):
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([0.0, 1.0, 4.0])
        result = engine.newton_divided_difference(x_data, y_data, 1.0)
        assert "coefficients" in result
        assert len(result["coefficients"]) == len(x_data)

    def test_newton_divided_diff_table(self, engine):
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 3.0, 7.0])
        result = engine.newton_divided_difference(x_data, y_data, 1.5)
        assert "divided_diff_table" in result
        table = result["divided_diff_table"]
        assert table.shape == (3, 3)

    def test_newton_matches_lagrange(self, engine):
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([2.0, 1.0, 4.0, 3.0])
        x_eval = 1.7
        lagrange_val = engine.lagrange(x_data, y_data, x_eval)
        newton_val = engine.newton_divided_difference(x_data, y_data, x_eval)["value"]
        assert np.isclose(lagrange_val, newton_val, atol=1e-10)


class TestCubicSpline:
    def test_cubic_spline_exact_points(self, engine):
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([0.0, 1.0, 4.0, 9.0])
        for xi, yi in zip(x_data, y_data):
            result = engine.cubic_spline(x_data, y_data, xi)
            assert np.isclose(result["value"], yi, atol=1e-6)

    def test_cubic_spline_two_points(self, engine):
        x_data = np.array([0.0, 2.0])
        y_data = np.array([1.0, 5.0])
        result = engine.cubic_spline(x_data, y_data, 1.0)
        assert np.isclose(result["value"], 3.0, atol=1e-6)

    def test_cubic_spline_returns_coefficients(self, engine):
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([0.0, 1.0, 4.0, 9.0])
        result = engine.cubic_spline(x_data, y_data, 1.5)
        assert "spline_coefficients" in result
        assert len(result["spline_coefficients"]) == 3

    def test_cubic_spline_smoothness(self, engine):
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        results = []
        for x in np.linspace(0, 4, 20):
            result = engine.cubic_spline(x_data, y_data, x)
            results.append(result["value"])
        values = np.array(results)
        diffs = np.diff(values)
        assert np.all(np.abs(diffs) < 2.0)

    def test_cubic_spline_endpoints(self, engine):
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([5.0, 3.0, 7.0, 4.0])
        r_start = engine.cubic_spline(x_data, y_data, 0.0)
        r_end = engine.cubic_spline(x_data, y_data, 3.0)
        assert np.isclose(r_start["value"], 5.0, atol=1e-6)
        assert np.isclose(r_end["value"], 4.0, atol=1e-6)
