import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.integration import NumericalIntegration


@pytest.fixture
def integ():
    return NumericalIntegration()


class TestTrapezoidal:
    def test_trapezoidal_linear(self, integ):
        f = lambda x: 2 * x + 1
        result = integ.trapezoidal(f, 0.0, 3.0, n=100)
        exact = 12.0
        assert np.isclose(result["result"], exact, atol=1e-4)

    def test_trapezoidal_constant(self, integ):
        f = lambda x: 5.0
        result = integ.trapezoidal(f, 0.0, 2.0, n=10)
        assert np.isclose(result["result"], 10.0, atol=1e-10)

    def test_trapezoidal_quadratic(self, integ):
        f = lambda x: x**2
        result = integ.trapezoidal(f, 0.0, 1.0, n=1000)
        exact = 1.0 / 3.0
        assert np.isclose(result["result"], exact, atol=1e-3)

    def test_trapezoidal_returns_metadata(self, integ):
        f = lambda x: x
        result = integ.trapezoidal(f, 0.0, 1.0, n=50)
        assert result["method"] == "Trapezoidal"
        assert result["n_intervals"] == 50


class TestSimpsonOneThird:
    def test_simpson_quadratic(self, integ):
        f = lambda x: x**2
        result = integ.simpson_one_third(f, 0.0, 1.0, n=100)
        exact = 1.0 / 3.0
        assert np.isclose(result["result"], exact, atol=1e-6)

    def test_simpson_cubic(self, integ):
        f = lambda x: x**3
        result = integ.simpson_one_third(f, 0.0, 1.0, n=100)
        exact = 1.0 / 4.0
        assert np.isclose(result["result"], exact, atol=1e-6)

    def test_simpson_sine(self, integ):
        f = lambda x: np.sin(x)
        result = integ.simpson_one_third(f, 0.0, np.pi, n=100)
        exact = 2.0
        assert np.isclose(result["result"], exact, atol=1e-4)

    def test_simpson_adjusts_odd_n(self, integ):
        f = lambda x: x**2
        result = integ.simpson_one_third(f, 0.0, 1.0, n=101)
        assert result["n_intervals"] % 2 == 0


class TestSimpsonThreeEighth:
    def test_simpson38_cubic(self, integ):
        f = lambda x: x**3
        result = integ.simpson_three_eighth(f, 0.0, 1.0, n=100)
        exact = 1.0 / 4.0
        assert np.isclose(result["result"], exact, atol=1e-5)

    def test_simpson38_quadratic(self, integ):
        f = lambda x: x**2
        result = integ.simpson_three_eighth(f, 0.0, 1.0, n=100)
        exact = 1.0 / 3.0
        assert np.isclose(result["result"], exact, atol=1e-5)

    def test_simpson38_adjusts_n(self, integ):
        f = lambda x: x**2
        result = integ.simpson_three_eighth(f, 0.0, 1.0, n=10)
        assert result["n_intervals"] % 3 == 0


class TestGaussianQuadrature:
    def test_gauss2_linear(self, integ):
        f = lambda x: 2 * x + 1
        result = integ.gaussian_quadrature_2point(f, 0.0, 3.0)
        assert np.isclose(result["result"], 12.0, atol=1e-10)

    def test_gauss2_quadratic(self, integ):
        f = lambda x: x**2
        result = integ.gaussian_quadrature_2point(f, 0.0, 1.0)
        assert np.isclose(result["result"], 1.0 / 3.0, atol=1e-2)

    def test_gauss3_cubic(self, integ):
        f = lambda x: x**3
        result = integ.gaussian_quadrature_3point(f, 0.0, 1.0)
        exact = 1.0 / 4.0
        assert np.isclose(result["result"], exact, atol=1e-4)

    def test_gauss3_quadratic(self, integ):
        f = lambda x: x**2
        result = integ.gaussian_quadrature_3point(f, 0.0, 1.0)
        exact = 1.0 / 3.0
        assert np.isclose(result["result"], exact, atol=1e-6)


class TestCompareMethods:
    def test_compare_returns_all_methods(self, integ):
        f = lambda x: x**2
        result = integ.compare_methods(f, 0.0, 1.0, exact=1.0 / 3.0)
        assert "Trapezoidal" in result
        assert "Simpson's 1/3" in result
        assert "Simpson's 3/8" in result
        assert "Gaussian 2-point" in result
        assert "Gaussian 3-point" in result
        assert "errors" in result

    def test_compare_simpson_better_than_trapezoid(self, integ):
        f = lambda x: np.sin(x)
        exact = 2.0
        result = integ.compare_methods(f, 0.0, np.pi, exact=exact)
        assert result["errors"]["Simpson's 1/3"] < result["errors"]["Trapezoidal"]

    def test_compare_gaussian_accurate(self, integ):
        f = lambda x: np.exp(x)
        exact = np.exp(1) - 1
        result = integ.compare_methods(f, 0.0, 1.0, exact=exact)
        for method in ["Trapezoidal", "Simpson's 1/3", "Gaussian 3-point"]:
            assert result["errors"][method] < 0.1
