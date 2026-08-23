import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.differentiation import NumericalDifferentiation


@pytest.fixture
def diff():
    return NumericalDifferentiation(h=1e-5)


class TestForwardDifference:
    def test_forward_basic_linear(self, diff):
        f = lambda x: 3 * x + 2
        assert np.isclose(diff.forward_difference(f, 1.0), 3.0, atol=1e-4)

    def test_forward_quadratic(self, diff):
        f = lambda x: x**2
        assert np.isclose(diff.forward_difference(f, 3.0), 6.0, atol=1e-4)

    def test_forward_sine(self, diff):
        f = lambda x: np.sin(x)
        assert np.isclose(diff.forward_difference(f, 0.0), 1.0, atol=1e-4)

    def test_forward_custom_h(self, diff):
        f = lambda x: x**2
        result = diff.forward_difference(f, 1.0, h=1e-3)
        assert np.isclose(result, 2.0, atol=1e-2)


class TestBackwardDifference:
    def test_backward_basic_linear(self, diff):
        f = lambda x: 3 * x + 2
        assert np.isclose(diff.backward_difference(f, 1.0), 3.0, atol=1e-4)

    def test_backward_quadratic(self, diff):
        f = lambda x: x**2
        assert np.isclose(diff.backward_difference(f, 3.0), 6.0, atol=1e-4)

    def test_backward_sine(self, diff):
        f = lambda x: np.sin(x)
        assert np.isclose(diff.backward_difference(f, 0.0), 1.0, atol=1e-4)

    def test_backward_custom_h(self, diff):
        f = lambda x: x**3
        result = diff.backward_difference(f, 2.0, h=1e-3)
        assert np.isclose(result, 12.0, atol=1e-2)


class TestCentralDifference:
    def test_central_basic_linear(self, diff):
        f = lambda x: 3 * x + 2
        assert np.isclose(diff.central_difference(f, 1.0), 3.0, atol=1e-6)

    def test_central_quadratic(self, diff):
        f = lambda x: x**2
        assert np.isclose(diff.central_difference(f, 3.0), 6.0, atol=1e-4)

    def test_central_sine(self, diff):
        f = lambda x: np.sin(x)
        assert np.isclose(diff.central_difference(f, 0.0), 1.0, atol=1e-6)

    def test_central_exponential(self, diff):
        f = lambda x: np.exp(x)
        x = 1.0
        assert np.isclose(diff.central_difference(f, x), np.exp(x), atol=1e-4)

    def test_central_more_accurate(self, diff):
        f = lambda x: x**3
        x = 2.0
        exact = 12.0
        fwd_err = abs(diff.forward_difference(f, x) - exact)
        cen_err = abs(diff.central_difference(f, x) - exact)
        assert cen_err < fwd_err


class TestSecondDerivative:
    def test_second_derivative_quadratic(self, diff):
        f = lambda x: x**2
        assert np.isclose(diff.second_derivative(f, 3.0), 2.0, atol=1e-4)

    def test_second_derivative_cubic(self, diff):
        f = lambda x: x**3
        assert np.isclose(diff.second_derivative(f, 2.0), 12.0, atol=1e-3)

    def test_second_derivative_sine(self, diff):
        f = lambda x: np.sin(x)
        assert np.isclose(diff.second_derivative(f, np.pi / 4), -np.sin(np.pi / 4), atol=1e-4)

    def test_second_derivative_constant(self, diff):
        f = lambda x: 5.0
        assert np.isclose(diff.second_derivative(f, 1.0), 0.0, atol=1e-6)


class TestCompareMethods:
    def test_compare_returns_all(self, diff):
        f = lambda x: x**2
        df_exact = lambda x: 2 * x
        result = diff.compare_methods(f, df_exact, 3.0)
        assert "forward" in result
        assert "backward" in result
        assert "central" in result
        assert "exact" in result
        assert "error_forward" in result
        assert "error_backward" in result
        assert "error_central" in result

    def test_compare_central_most_accurate(self, diff):
        f = lambda x: np.sin(x)
        df_exact = lambda x: np.cos(x)
        result = diff.compare_methods(f, df_exact, 1.0)
        assert result["error_central"] < result["error_forward"]
        assert result["error_central"] < result["error_backward"]

    def test_compare_exact_values(self, diff):
        f = lambda x: x**3
        df_exact = lambda x: 3 * x**2
        result = diff.compare_methods(f, df_exact, 2.0)
        assert np.isclose(result["exact"], 12.0, atol=1e-10)


class TestPriceSensitivity:
    def test_price_sensitivity_positive(self, diff):
        price_fn = lambda x: 200 * x + 50
        result = diff.price_sensitivity(price_fn, 10.0, "sqft")
        assert result["sensitivity"] > 0
        assert result["feature"] == "sqft"

    def test_price_sensitivity_negative(self, diff):
        price_fn = lambda x: -100 * x + 500
        result = diff.price_sensitivity(price_fn, 3.0, "age")
        assert result["sensitivity"] < 0

    def test_price_sensitivity_zero(self, diff):
        price_fn = lambda x: 100.0
        result = diff.price_sensitivity(price_fn, 5.0, "noise")
        assert np.isclose(result["sensitivity"], 0.0, atol=1e-4)
