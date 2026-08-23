import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.root_finder import RootFinder


@pytest.fixture
def rf():
    return RootFinder(tolerance=1e-8, max_iterations=1000)


class TestBisection:
    def test_bisection_basic(self, rf):
        f = lambda x: x**2 - 4
        result = rf.bisection(f, 0, 5)
        assert result["success"]
        assert np.isclose(result["root"], 2.0, atol=1e-6)

    def test_bisection_negative_root(self, rf):
        f = lambda x: x**2 - 4
        result = rf.bisection(f, -5, 0)
        assert result["success"]
        assert np.isclose(result["root"], -2.0, atol=1e-6)

    def test_bisection_no_sign_change(self, rf):
        f = lambda x: x**2 + 1
        result = rf.bisection(f, 1, 3)
        assert not result["success"]
        assert "opposite signs" in result["message"]

    def test_bisection_already_root(self, rf):
        f = lambda x: x - 2
        result = rf.bisection(f, 2, 5)
        assert result["success"]
        assert result["root"] == 2.0

    def test_bisection_cubic(self, rf):
        f = lambda x: x**3 - x - 2
        result = rf.bisection(f, 1, 2)
        assert result["success"]
        expected_root = 1.5213797068045677
        assert np.isclose(result["root"], expected_root, atol=1e-5)

    def test_bisection_iterations_count(self, rf):
        f = lambda x: x**2 - 2
        result = rf.bisection(f, 0, 2)
        assert result["success"]
        assert result["iterations"] > 0
        assert len(result["convergence_history"]) == result["iterations"]


class TestNewtonRaphson:
    def test_newton_basic(self, rf):
        f = lambda x: x**2 - 4
        df = lambda x: 2 * x
        result = rf.newton_raphson(f, df, x0=3.0)
        assert result["success"]
        assert np.isclose(result["root"], 2.0, atol=1e-8)

    def test_newton_negative_root(self, rf):
        f = lambda x: x**2 - 4
        df = lambda x: 2 * x
        result = rf.newton_raphson(f, df, x0=-3.0)
        assert result["success"]
        assert np.isclose(result["root"], -2.0, atol=1e-8)

    def test_newton_convergence_fast(self, rf):
        f = lambda x: x**3 - 6*x**2 + 11*x - 6
        df = lambda x: 3*x**2 - 12*x + 11
        result = rf.newton_raphson(f, df, x0=0.5)
        assert result["success"]
        assert np.isclose(result["root"], 1.0, atol=1e-6)

    def test_newton_zero_derivative(self, rf):
        f = lambda x: x**2
        df = lambda x: 2 * x
        result = rf.newton_raphson(f, df, x0=0.0)
        assert not result["success"]
        assert "Derivative is zero" in result["message"]

    def test_newton_transcendental(self, rf):
        f = lambda x: np.cos(x) - x
        df = lambda x: -np.sin(x) - 1
        result = rf.newton_raphson(f, df, x0=0.5)
        assert result["success"]
        expected = 0.7390851332151607
        assert np.isclose(result["root"], expected, atol=1e-6)


class TestSecant:
    def test_secant_basic(self, rf):
        f = lambda x: x**2 - 4
        result = rf.secant(f, x0=0.0, x1=3.0)
        assert result["success"]
        assert np.isclose(result["root"], 2.0, atol=1e-6)

    def test_secant_negative_root(self, rf):
        f = lambda x: x**2 - 4
        result = rf.secant(f, x0=-3.0, x1=0.0)
        assert result["success"]
        assert np.isclose(result["root"], -2.0, atol=1e-6)

    def test_secant_already_root_at_x0(self, rf):
        f = lambda x: x - 2
        result = rf.secant(f, x0=2.0, x1=5.0)
        assert result["success"]
        assert result["root"] == 2.0

    def test_secant_already_root_at_x1(self, rf):
        f = lambda x: x - 2
        result = rf.secant(f, x0=0.0, x1=2.0)
        assert result["success"]
        assert result["root"] == 2.0

    def test_secant_cubic(self, rf):
        f = lambda x: x**3 - x - 2
        result = rf.secant(f, x0=1.0, x1=2.0)
        assert result["success"]
        expected = 1.5213797068045677
        assert np.isclose(result["root"], expected, atol=1e-5)


class TestSimultaneousNewton:
    def test_simultaneous_basic(self, rf):
        f1 = lambda x, y: x**2 + y**2 - 4
        f2 = lambda x, y: x - y
        jacobian = lambda x, y: [[2*x, 2*y], [1.0, -1.0]]
        result = rf.simultaneous_newton(
            [f1, f2], jacobian, [1.0, 1.0]
        )
        assert result["success"]
        sol = result["solution"]
        assert np.isclose(sol[0], np.sqrt(2), atol=1e-5)
        assert np.isclose(sol[1], np.sqrt(2), atol=1e-5)

    def test_simultaneous_nonlinear_system(self, rf):
        f1 = lambda x, y: x**2 + y - 11
        f2 = lambda x, y: x + y**2 - 7
        jacobian = lambda x, y: [[2*x, 1.0], [1.0, 2*y]]
        result = rf.simultaneous_newton(
            [f1, f2], jacobian, [1.0, 1.0]
        )
        assert result["success"]
        sol = result["solution"]
        assert np.isclose(f1(sol[0], sol[1]), 0, atol=1e-6)
        assert np.isclose(f2(sol[0], sol[1]), 0, atol=1e-6)

    def test_simultaneous_singular_jacobian(self, rf):
        f1 = lambda x, y: x + y
        f2 = lambda x, y: x + y
        jacobian = lambda x, y: [[1.0, 1.0], [1.0, 1.0]]
        result = rf.simultaneous_newton(
            [f1, f2], jacobian, [1.0, 1.0]
        )
        assert not result["success"]

    def test_simultaneous_three_variables(self, rf):
        f1 = lambda x, y, z: x + y + z - 6
        f2 = lambda x, y, z: 2*x - y + z - 3
        f3 = lambda x, y, z: x + 2*y - z - 1
        jacobian = lambda x, y, z: [
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 1.0],
            [1.0, 2.0, -1.0],
        ]
        result = rf.simultaneous_newton(
            [f1, f2, f3], jacobian, [1.0, 1.0, 1.0]
        )
        assert result["success"]
        sol = result["solution"]
        expected = np.linalg.solve(
            [[1, 1, 1], [2, -1, 1], [1, 2, -1]],
            [6, 3, 1]
        )
        assert np.isclose(sol, expected, atol=1e-5).all()
