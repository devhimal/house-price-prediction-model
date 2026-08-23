import numpy as np
import time
from typing import Dict, Any, Tuple, Optional


class MatrixSolver:
    """A collection of manual matrix solving methods without using numpy.linalg."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def _validate_inputs(self, A: np.ndarray, b: np.ndarray) -> Optional[str]:
        """Validate matrix dimensions and compatibility."""
        if A.ndim != 2:
            return "Matrix A must be 2-dimensional"
        if A.shape[0] != A.shape[1]:
            return "Matrix A must be square"
        if b.ndim == 1:
            if b.shape[0] != A.shape[0]:
                return "Vector b must have same number of rows as A"
        elif b.ndim == 2:
            if b.shape[0] != A.shape[0]:
                return "Vector b must have same number of rows as A"
            if b.shape[1] != 1:
                return "Vector b must be a column vector (n x 1)"
        else:
            return "Vector b must be 1-dimensional or a column vector (n x 1)"
        if A.dtype not in [np.float64, np.float32]:
            return "Matrix A must contain numeric (float) values"
        return None

    def _is_singular(self, A: np.ndarray) -> bool:
        """Check if a matrix is singular using its determinant estimate."""
        n = A.shape[0]
        det_sign = 1.0
        pivot_product = 1.0
        for i in range(n):
            max_val = 0.0
            max_idx = i
            for j in range(i, n):
                if abs(A[j, i]) > max_val:
                    max_val = abs(A[j, i])
                    max_idx = j
            if max_val < self.tolerance:
                return True
            if max_idx != i:
                A[[i, max_idx]] = A[[max_idx, i]]
                det_sign *= -1.0
            pivot_product *= A[i, i]
            for j in range(i + 1, n):
                if abs(A[i, i]) > self.tolerance:
                    factor = A[j, i] / A[i, i]
                    for k in range(i + 1, n):
                        A[j, k] -= factor * A[i, k]
        return False

    def gauss_jordan(self, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Solve Ax = b using Gauss-Jordan elimination with partial pivoting.

        Steps:
        1. Form augmented matrix [A|b]
        2. Forward elimination with partial pivoting
        3. Back substitution
        """
        start_time = time.time()

        error = self._validate_inputs(A, b)
        if error is not None:
            return {
                "solution": None,
                "iterations": 0,
                "execution_time": 0.0,
                "residual": None,
                "success": False,
                "message": error,
            }

        n = A.shape[0]
        Ab = np.hstack([A.astype(np.float64).copy(), b.astype(np.float64).reshape(n, 1)])

        iterations = 0
        for col in range(n):
            max_row = col
            max_val = abs(Ab[col, col])
            for row in range(col + 1, n):
                if abs(Ab[row, col]) > max_val:
                    max_val = abs(Ab[row, col])
                    max_row = row
            if max_val < self.tolerance:
                return {
                    "solution": None,
                    "iterations": iterations,
                    "execution_time": time.time() - start_time,
                    "residual": None,
                    "success": False,
                    "message": f"Matrix is singular or near-singular at column {col} (pivot = {max_val:.2e})",
                }
            if max_row != col:
                Ab[[col, max_row]] = Ab[[max_row, col]]

            pivot = Ab[col, col]
            Ab[col] = Ab[col] / pivot

            for row in range(n):
                if row != col:
                    factor = Ab[row, col]
                    Ab[row] = Ab[row] - factor * Ab[col]
            iterations += 1

        solution = Ab[:, n]
        residual = np.linalg.norm(A @ solution - b.astype(np.float64))
        success = residual < self.tolerance or np.allclose(A @ solution, b.astype(np.float64), atol=self.tolerance)

        return {
            "solution": solution,
            "iterations": iterations,
            "execution_time": time.time() - start_time,
            "residual": float(residual),
            "success": success,
            "message": "Gauss-Jordan elimination completed successfully" if success else f"Solution may be inaccurate (residual = {residual:.2e})",
        }

    def lu_decomposition(self, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Solve Ax = b using LU decomposition (Doolittle's method).

        Steps:
        1. Decompose A = LU where L is lower triangular with 1s on diagonal, U is upper triangular
        2. Solve Ly = b (forward substitution)
        3. Solve Ux = y (back substitution)
        """
        start_time = time.time()

        error = self._validate_inputs(A, b)
        if error is not None:
            return {
                "solution": None,
                "L": None,
                "U": None,
                "execution_time": 0.0,
                "residual": None,
                "success": False,
                "message": error,
            }

        n = A.shape[0]
        L = np.eye(n, dtype=np.float64)
        U = A.astype(np.float64).copy()

        for i in range(n):
            if abs(U[i, i]) < self.tolerance:
                return {
                    "solution": None,
                    "L": None,
                    "U": None,
                    "execution_time": time.time() - start_time,
                    "residual": None,
                    "success": False,
                    "message": f"Zero pivot encountered at step {i} (value = {U[i, i]:.2e}). Matrix may be singular.",
                }

            for j in range(i + 1, n):
                L[j, i] = U[j, i] / U[i, i]
                for k in range(i, n):
                    U[j, k] -= L[j, i] * U[i, k]

        y = np.zeros(n, dtype=np.float64)
        b_float = b.astype(np.float64).ravel()
        for i in range(n):
            y[i] = b_float[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]

        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < self.tolerance:
                return {
                    "solution": None,
                    "L": L,
                    "U": U,
                    "execution_time": time.time() - start_time,
                    "residual": None,
                    "success": False,
                    "message": f"Zero pivot during back substitution at row {i}.",
                }
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= U[i, j] * x[j]
            x[i] /= U[i, i]

        residual = np.linalg.norm(A @ x - b_float)
        success = residual < self.tolerance or np.allclose(A @ x, b_float, atol=self.tolerance)

        return {
            "solution": x,
            "L": L,
            "U": U,
            "execution_time": time.time() - start_time,
            "residual": float(residual),
            "success": success,
            "message": "LU decomposition completed successfully" if success else f"Solution may be inaccurate (residual = {residual:.2e})",
        }

    def cholesky(self, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Solve Ax = b using Cholesky decomposition (A must be symmetric positive definite).

        Steps:
        1. Check A is symmetric and positive definite
        2. Decompose A = LL^T
        3. Solve Ly = b (forward substitution)
        4. Solve L^Tx = y (back substitution)
        """
        start_time = time.time()

        error = self._validate_inputs(A, b)
        if error is not None:
            return {
                "solution": None,
                "L": None,
                "execution_time": 0.0,
                "residual": None,
                "success": False,
                "message": error,
            }

        n = A.shape[0]
        A_f = A.astype(np.float64).copy()

        if not np.allclose(A_f, A_f.T, atol=self.tolerance * 100):
            max_diff = float(np.max(np.abs(A_f - A_f.T)))
            return {
                "solution": None,
                "L": None,
                "execution_time": time.time() - start_time,
                "residual": None,
                "success": False,
                "message": f"Matrix is not symmetric (max asymmetry = {max_diff:.2e}).",
            }

        L = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i):
                s = 0.0
                for k in range(j):
                    s += L[i, k] * L[j, k]
                L[i, j] = (A_f[i, j] - s) / L[j, j]

            s = 0.0
            for k in range(i):
                s += L[i, k] * L[i, k]
            diag_val = A_f[i, i] - s
            if diag_val <= 0:
                return {
                    "solution": None,
                    "L": None,
                    "execution_time": time.time() - start_time,
                    "residual": None,
                    "success": False,
                    "message": f"Matrix is not positive definite at step {i} (diagonal value = {diag_val:.2e}).",
                }
            L[i, i] = np.sqrt(diag_val)

        y = np.zeros(n, dtype=np.float64)
        b_float = b.astype(np.float64).ravel()
        for i in range(n):
            y[i] = b_float[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
            y[i] /= L[i, i]

        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= L[j, i] * x[j]
            x[i] /= L[i, i]

        residual = np.linalg.norm(A @ x - b_float)
        success = residual < self.tolerance or np.allclose(A @ x, b_float, atol=self.tolerance)

        return {
            "solution": x,
            "L": L,
            "execution_time": time.time() - start_time,
            "residual": float(residual),
            "success": success,
            "message": "Cholesky decomposition completed successfully" if success else f"Solution may be inaccurate (residual = {residual:.2e})",
        }

    def compare_methods(self, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Run all three methods and compare results.

        Returns dict with results for each method and comparison table.
        """
        results = {}

        gauss_result = self.gauss_jordan(A, b)
        results["gauss_jordan"] = gauss_result

        lu_result = self.lu_decomposition(A, b)
        results["lu_decomposition"] = lu_result

        cholesky_result = self.cholesky(A, b)
        results["cholesky"] = cholesky_result

        methods = ["gauss_jordan", "lu_decomposition", "cholesky"]
        comparison_rows = []
        for method in methods:
            r = results[method]
            sol = r.get("solution")
            res = r.get("residual")
            exec_t = r.get("execution_time", 0.0)
            succ = r.get("success", False)
            msg = r.get("message", "")
            comparison_rows.append({
                "method": method,
                "success": succ,
                "residual": res,
                "execution_time": exec_t,
                "message": msg,
                "solution": sol.tolist() if sol is not None else None,
            })

        reference = None
        for method in ["gauss_jordan", "lu_decomposition", "cholesky"]:
            if results[method]["success"] and results[method]["solution"] is not None:
                reference = results[method]["solution"]
                break

        if reference is not None:
            for row in comparison_rows:
                if row["solution"] is not None:
                    row["solution_diff_from_reference"] = float(np.linalg.norm(np.array(row["solution"]) - reference))
                else:
                    row["solution_diff_from_reference"] = None
        else:
            for row in comparison_rows:
                row["solution_diff_from_reference"] = None

        results["comparison"] = comparison_rows
        results["reference_solution"] = reference.tolist() if reference is not None else None

        return results
