import numpy as np
import time
from typing import Dict, Any, Tuple, Optional


class EigenSolver:
    """A collection of manual eigenvalue/eigenvector solving methods without using numpy.linalg.eig."""

    def __init__(self, tolerance: float = 1e-10, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _check_symmetric(self, A: np.ndarray) -> bool:
        """Check if matrix is symmetric (within tolerance).

        Args:
            A: Square matrix to check.

        Returns:
            True if A is approximately symmetric, False otherwise.
        """
        return np.allclose(A, A.T, atol=self.tolerance)

    def _qr_factorization(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Manual QR factorization using modified Gram-Schmidt process.

        Decomposes A = Q @ R where Q is orthogonal and R is upper triangular.

        Args:
            A: Matrix of shape (n, m) with n >= m.

        Returns:
            Tuple of (Q, R) where Q is (n, m) orthonormal and R is (m, m) upper triangular.
        """
        n, m = A.shape
        Q = np.zeros((n, m), dtype=np.float64)
        R = np.zeros((m, m), dtype=np.float64)

        for j in range(m):
            v = A[:, j].copy().astype(np.float64)
            for i in range(j):
                R[i, j] = float(np.dot(Q[:, i], v))
                v = v - R[i, j] * Q[:, i]
            R[j, j] = float(np.linalg.norm(v))
            if R[j, j] > self.tolerance:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = 0.0

        return Q, R

    def power_method(self, A: np.ndarray) -> Dict[str, Any]:
        """Find dominant eigenvalue and eigenvector using Power Method.

        Algorithm:
            1. Start with random vector v
            2. Repeat: w = A @ v, lambda = ||w||, v = w / lambda
            3. Until |lambda_new - lambda_old| < tolerance

        Args:
            A: Square matrix to find the dominant eigenvalue of.

        Returns:
            Dict with keys: eigenvalue, eigenvector, iterations, convergence_history, success, message.
        """
        start_time = time.time()

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {
                "eigenvalue": None,
                "eigenvector": None,
                "iterations": 0,
                "convergence_history": [],
                "success": False,
                "message": "Matrix must be square",
                "execution_time": 0.0,
            }

        n = A.shape[0]
        A_f = A.astype(np.float64)
        v = np.random.rand(n).astype(np.float64)
        v = v / np.linalg.norm(v)

        convergence_history = []
        eigenvalue_old = 0.0

        for i in range(1, self.max_iterations + 1):
            w = A_f @ v
            eigenvalue = float(np.linalg.norm(w))

            if eigenvalue < self.tolerance:
                return {
                    "eigenvalue": 0.0,
                    "eigenvector": v,
                    "iterations": i,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "Dominant eigenvalue is zero",
                    "execution_time": time.time() - start_time,
                }

            v = w / eigenvalue
            error = abs(eigenvalue - eigenvalue_old)
            convergence_history.append(error)

            if error < self.tolerance:
                return {
                    "eigenvalue": float(eigenvalue),
                    "eigenvector": v,
                    "iterations": i,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "Power method converged",
                    "execution_time": time.time() - start_time,
                }

            eigenvalue_old = eigenvalue

        return {
            "eigenvalue": float(eigenvalue),
            "eigenvector": v,
            "iterations": self.max_iterations,
            "convergence_history": convergence_history,
            "success": False,
            "message": "Maximum iterations reached without convergence",
            "execution_time": time.time() - start_time,
        }

    def qr_iteration(self, A: np.ndarray) -> Dict[str, Any]:
        """Find eigenvalues using QR iteration with shift.

        Algorithm:
            1. A_0 = A
            2. Repeat: Q, R = QR_factorization(A_k), A_{k+1} = R @ Q
            3. Eigenvalues converge to diagonal of A_k

        Uses Wilkinson shift for faster convergence on real eigenvalues.

        Args:
            A: Square matrix to find eigenvalues of.

        Returns:
            Dict with keys: eigenvalues, eigenvectors, iterations, convergence_history, success, message.
        """
        start_time = time.time()

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {
                "eigenvalues": None,
                "eigenvectors": None,
                "iterations": 0,
                "convergence_history": [],
                "success": False,
                "message": "Matrix must be square",
                "execution_time": 0.0,
            }

        n = A.shape[0]
        A_k = A.astype(np.float64).copy()
        V = np.eye(n, dtype=np.float64)
        convergence_history = []

        for iteration in range(1, self.max_iterations + 1):
            # Compute Wilkinson shift from bottom-right 2x2 block
            shift = 0.0
            if n >= 2:
                a = A_k[n - 2, n - 2]
                b = A_k[n - 2, n - 1]
                c = A_k[n - 1, n - 2]
                d = A_k[n - 1, n - 1]
                trace = a + d
                det = a * d - b * c
                disc = trace * trace - 4.0 * det
                if disc >= 0:
                    sqrt_disc = np.sqrt(disc)
                    lam1 = (trace + sqrt_disc) / 2.0
                    lam2 = (trace - sqrt_disc) / 2.0
                    if abs(lam1 - d) < abs(lam2 - d):
                        shift = lam1
                    else:
                        shift = lam2
                else:
                    shift = d  # Use diagonal element for complex eigenvalues

            A_shifted = A_k - shift * np.eye(n, dtype=np.float64)
            Q, R = self._qr_factorization(A_shifted)
            A_k = R @ Q + shift * np.eye(n, dtype=np.float64)
            V = V @ Q

            # Measure convergence: off-diagonal norm
            off_diag = 0.0
            for i in range(1, n):
                for j in range(i):
                    off_diag += A_k[i, j] ** 2
            off_diag = np.sqrt(off_diag)
            convergence_history.append(off_diag)

            if off_diag < self.tolerance:
                eigenvalues = np.array([A_k[i, i] for i in range(n)], dtype=np.float64)
                return {
                    "eigenvalues": eigenvalues,
                    "eigenvectors": V,
                    "iterations": iteration,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "QR iteration converged",
                    "execution_time": time.time() - start_time,
                }

        eigenvalues = np.array([A_k[i, i] for i in range(n)], dtype=np.float64)
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": V,
            "iterations": self.max_iterations,
            "convergence_history": convergence_history,
            "success": False,
            "message": "Maximum iterations reached without convergence",
            "execution_time": time.time() - start_time,
        }
