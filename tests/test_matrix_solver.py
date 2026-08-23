import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.matrix_solver import MatrixSolver


@pytest.fixture
def solver():
    return MatrixSolver(tolerance=1e-10)


class TestGaussJordan:
    def test_gauss_jordan_basic_2x2(self, solver):
        A = np.array([[2.0, 1.0], [5.0, 7.0]])
        b = np.array([11.0, 13.0])
        result = solver.gauss_jordan(A, b)
        assert result["success"]
        expected = np.linalg.solve(A, b)
        assert np.isclose(result["solution"], expected, atol=1e-8).all()

    def test_gauss_jordan_basic_3x3(self, solver):
        A = np.array([[1.0, 1.0, 1.0],
                       [2.0, 3.0, 1.0],
                       [3.0, 5.0, 6.0]])
        b = np.array([6.0, 14.0, 32.0])
        result = solver.gauss_jordan(A, b)
        assert result["success"]
        expected = np.linalg.solve(A, b)
        assert np.isclose(result["solution"], expected, atol=1e-8).all()

    def test_gauss_jordan_singular_matrix(self, solver):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([3.0, 6.0])
        result = solver.gauss_jordan(A, b)
        assert not result["success"]

    def test_gauss_jordan_non_square_matrix(self, solver):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([1.0, 2.0])
        result = solver.gauss_jordan(A, b)
        assert not result["success"]

    def test_gauss_jordan_identity_system(self, solver):
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        result = solver.gauss_jordan(A, b)
        assert result["success"]
        assert np.isclose(result["solution"], b, atol=1e-10).all()


class TestLUDecomposition:
    def test_lu_basic_2x2(self, solver):
        A = np.array([[4.0, 3.0], [6.0, 3.0]])
        b = np.array([10.0, 12.0])
        result = solver.lu_decomposition(A, b)
        assert result["success"]
        assert np.isclose(result["solution"], [1.0, 2.0], atol=1e-8).all()

    def test_lu_basic_3x3(self, solver):
        A = np.array([[2.0, 1.0, -1.0],
                       [-3.0, -1.0, 2.0],
                       [-2.0, 1.0, 2.0]])
        b = np.array([8.0, -11.0, -3.0])
        result = solver.lu_decomposition(A, b)
        assert result["success"]
        expected = np.array([2.0, 3.0, -1.0])
        assert np.isclose(result["solution"], expected, atol=1e-8).all()

    def test_lu_decomposition_factors(self, solver):
        A = np.array([[4.0, 3.0], [6.0, 3.0]])
        b = np.array([10.0, 12.0])
        result = solver.lu_decomposition(A, b)
        assert result["success"]
        L = result["L"]
        U = result["U"]
        reconstructed = L @ U
        assert np.isclose(reconstructed, A, atol=1e-8).all()

    def test_lu_singular_matrix(self, solver):
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = np.array([2.0, 2.0])
        result = solver.lu_decomposition(A, b)
        assert not result["success"]

    def test_lu_identity_system(self, solver):
        A = np.eye(4)
        b = np.array([4.0, 3.0, 2.0, 1.0])
        result = solver.lu_decomposition(A, b)
        assert result["success"]
        assert np.isclose(result["solution"], b, atol=1e-10).all()


class TestCholesky:
    def test_cholesky_basic_2x2(self, solver):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([8.0, 8.0])
        result = solver.cholesky(A, b)
        assert result["success"]
        expected = np.linalg.solve(A, b)
        assert np.isclose(result["solution"], expected, atol=1e-8).all()

    def test_cholesky_basic_3x3(self, solver):
        A = np.array([[4.0, 12.0, -16.0],
                       [12.0, 37.0, -43.0],
                       [-16.0, -43.0, 98.0]])
        b = np.array([1.0, 2.0, 3.0])
        result = solver.cholesky(A, b)
        assert result["success"]
        expected = np.linalg.solve(A, b)
        assert np.isclose(result["solution"], expected, atol=1e-8).all()

    def test_cholesky_not_symmetric(self, solver):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        result = solver.cholesky(A, b)
        assert not result["success"]
        assert "not symmetric" in result["message"]

    def test_cholesky_not_positive_definite(self, solver):
        A = np.array([[1.0, 2.0], [2.0, 1.0]])
        b = np.array([3.0, 3.0])
        result = solver.cholesky(A, b)
        assert not result["success"]

    def test_cholesky_identity_system(self, solver):
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        result = solver.cholesky(A, b)
        assert result["success"]
        assert np.isclose(result["solution"], b, atol=1e-8).all()


class TestCompareMethods:
    def test_compare_methods_all_agree(self, solver):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([10.0, 8.0])
        result = solver.compare_methods(A, b)
        gauss_sol = result["gauss_jordan"]["solution"]
        lu_sol = result["lu_decomposition"]["solution"]
        cholesky_sol = result["cholesky"]["solution"]
        assert gauss_sol is not None
        assert lu_sol is not None
        assert cholesky_sol is not None
        assert np.isclose(gauss_sol, lu_sol, atol=1e-8).all()
        assert np.isclose(lu_sol, cholesky_sol, atol=1e-8).all()

    def test_compare_methods_returns_comparison(self, solver):
        A = np.array([[2.0, 1.0], [5.0, 7.0]])
        b = np.array([11.0, 13.0])
        result = solver.compare_methods(A, b)
        assert "comparison" in result
        assert len(result["comparison"]) == 3

    def test_compare_methods_singluar_matrix(self, solver):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([3.0, 6.0])
        result = solver.compare_methods(A, b)
        for method in ["gauss_jordan", "lu_decomposition", "cholesky"]:
            assert not result[method]["success"]
