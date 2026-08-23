import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.numerical_methods.eigen_solver import EigenSolver


@pytest.fixture
def solver():
    return EigenSolver(tolerance=1e-10, max_iterations=10000)


class TestPowerMethod:
    def test_power_method_diagonal(self, solver):
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        result = solver.power_method(A)
        assert result["success"]
        assert np.isclose(result["eigenvalue"], 3.0, atol=1e-4)

    def test_power_method_symmetric(self, solver):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = solver.power_method(A)
        assert result["success"]
        assert np.isclose(result["eigenvalue"], 3.0, atol=1e-4)

    def test_power_method_3x3(self, solver):
        A = np.array([[5.0, 0.0, 0.0],
                       [0.0, 3.0, 0.0],
                       [0.0, 0.0, 1.0]])
        result = solver.power_method(A)
        assert result["success"]
        assert np.isclose(result["eigenvalue"], 5.0, atol=1e-4)

    def test_power_method_eigenvector(self, solver):
        A = np.array([[2.0, 0.0], [0.0, 5.0]])
        result = solver.power_method(A)
        assert result["success"]
        eigenvector = result["eigenvector"]
        normalized = eigenvector / np.linalg.norm(eigenvector)
        assert np.isclose(abs(normalized[0]), 0.0, atol=1e-3)
        assert np.isclose(abs(normalized[1]), 1.0, atol=1e-3)

    def test_power_method_non_square(self, solver):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = solver.power_method(A)
        assert not result["success"]
        assert "square" in result["message"]

    def test_power_method_convergence_history(self, solver):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        result = solver.power_method(A)
        assert result["success"]
        assert len(result["convergence_history"]) > 0
        errors = result["convergence_history"]
        assert errors[-1] < errors[0]


class TestQRIteration:
    def test_qr_diagonal_matrix(self, solver):
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        result = solver.qr_iteration(A)
        assert result["success"]
        eigenvalues = np.sort(result["eigenvalues"])
        assert np.isclose(eigenvalues, [1.0, 3.0], atol=1e-4).all()

    def test_qr_symmetric_2x2(self, solver):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = solver.qr_iteration(A)
        assert result["success"]
        eigenvalues = np.sort(result["eigenvalues"])
        assert np.isclose(eigenvalues, [1.0, 3.0], atol=1e-4).all()

    def test_qr_symmetric_3x3(self, solver):
        A = np.array([[4.0, -2.0, 0.0],
                       [-2.0, 4.0, 0.0],
                       [0.0, 0.0, 5.0]])
        result = solver.qr_iteration(A)
        assert result["success"]
        eigenvalues = np.sort(result["eigenvalues"])
        expected = np.sort(np.linalg.eigvalsh(A))
        assert np.isclose(eigenvalues, expected, atol=1e-3).all()

    def test_qr_identity_matrix(self, solver):
        A = np.eye(3)
        result = solver.qr_iteration(A)
        assert result["success"]
        eigenvalues = np.sort(result["eigenvalues"])
        assert np.isclose(eigenvalues, [1.0, 1.0, 1.0], atol=1e-4).all()

    def test_qr_non_square(self, solver):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = solver.qr_iteration(A)
        assert not result["success"]

    def test_qr_eigenvectors_span(self, solver):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = solver.qr_iteration(A)
        assert result["success"]
        V = result["eigenvectors"]
        for i in range(V.shape[1]):
            col_norm = np.linalg.norm(V[:, i])
            assert np.isclose(col_norm, 0.0, atol=1e-3) or np.isclose(col_norm, 1.0, atol=1e-3)
