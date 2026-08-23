import numpy as np
from typing import Dict, Any, Optional
from src.numerical_methods.matrix_solver import MatrixSolver


class RegressionEngine:
    """Regression engine that communicates with MatrixSolver to fit linear and polynomial models."""

    def __init__(self):
        self.coefficients: Optional[Dict[str, float]] = None
        self.solver = MatrixSolver()
        self.is_fitted: bool = False
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None

    def fit_linear(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
        solver: str = "gauss_jordan",
    ) -> Dict[str, Any]:
        """Fit multiple linear regression: Price = b0 + b1*X1 + b2*X2 + ...

        Steps:
        1. Add column of 1s to X for intercept
        2. Compute A = X^T @ X
        3. Compute b = X^T @ y
        4. Solve A @ beta = b using selected solver method
        5. Store coefficients

        Returns dict with: coefficients dict (feature -> value), solver_used, residual, success
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]

        self.feature_names = ["intercept"] + list(feature_names)

        ones = np.ones((n_samples, 1), dtype=np.float64)
        X_design = np.hstack([ones, X])

        A = X_design.T @ X_design
        b = X_design.T @ y

        solve_method = getattr(self.solver, solver)
        result = solve_method(A, b)

        beta = result["solution"]

        if beta is None:
            return {
                "coefficients": None,
                "solver_used": solver,
                "residual": None,
                "success": False,
                "message": result.get("message", "Solver failed"),
            }

        coeffs = {}
        for i, name in enumerate(self.feature_names):
            coeffs[name] = float(beta[i])

        self.coefficients = coeffs
        self.is_fitted = True
        self.X_train = X.copy()
        self.y_train = y.copy()

        predicted = X_design @ beta
        residual_val = np.linalg.norm(y - predicted)

        return {
            "coefficients": coeffs,
            "solver_used": solver,
            "residual": float(residual_val),
            "success": result.get("success", False),
            "message": result.get("message", ""),
        }

    def fit_polynomial(
        self,
        X: np.ndarray,
        y: np.ndarray,
        degree: int = 2,
        feature_index: int = 0,
        solver: str = "gauss_jordan",
    ) -> Dict[str, Any]:
        """Fit polynomial regression using a single feature.

        Price = b0 + b1*A + b2*A^2 + ... + bd*A^d

        Returns dict with: coefficients, degree, success
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        a = X[:, feature_index]

        poly_features = np.column_stack([a**d for d in range(degree + 1)])

        feature_names = [f"X{feature_index}^{d}" for d in range(degree + 1)]

        result = self.fit_linear(poly_features, y, feature_names=feature_names, solver=solver)

        return {
            "coefficients": result.get("coefficients"),
            "degree": degree,
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "residual": result.get("residual"),
            "solver_used": solver,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using fitted model."""
        if not self.is_fitted or self.coefficients is None:
            raise RuntimeError("Model has not been fitted yet. Call fit_linear or fit_polynomial first.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        ones = np.ones((n_samples, 1), dtype=np.float64)
        X_design = np.hstack([ones, X])

        n_coeffs = len(self.coefficients)
        beta = np.array([self.coefficients[name] for name in list(self.coefficients.keys())], dtype=np.float64)

        return X_design @ beta

    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R^2 = 1 - SS_res / SS_tot"""
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1.0 - ss_res / ss_tot)

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE = sqrt(mean((y_true - y_pred)^2))"""
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAE = mean(|y_true - y_pred|)"""
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        return float(np.mean(np.abs(y_true - y_pred)))

    def calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Return residuals = y_true - y_pred"""
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        return y_true - y_pred

    def get_model_summary(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Return comprehensive model evaluation metrics."""
        residuals = self.calculate_residuals(y_true, y_pred)

        return {
            "r2": self.calculate_r2(y_true, y_pred),
            "rmse": self.calculate_rmse(y_true, y_pred),
            "mae": self.calculate_mae(y_true, y_pred),
            "residuals": residuals,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "n_samples": int(len(y_true)),
        }
