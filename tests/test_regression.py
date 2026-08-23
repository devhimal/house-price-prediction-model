import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.regression.regression_engine import RegressionEngine


@pytest.fixture
def engine():
    return RegressionEngine()


class TestFitLinear:
    def test_fit_linear_simple(self, engine):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
        result = engine.fit_linear(X, y)
        assert result["success"]
        assert np.isclose(result["coefficients"]["intercept"], 1.0, atol=1e-3)
        assert np.isclose(result["coefficients"]["X0"], 2.0, atol=1e-3)

    def test_fit_linear_multiple_features(self, engine):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0], [5.0, 4.0]])
        y = np.array([5.0, 4.0, 9.0, 8.0, 13.0])
        result = engine.fit_linear(X, y, feature_names=["area", "rooms"])
        assert result["success"]
        assert "intercept" in result["coefficients"]
        assert "area" in result["coefficients"]
        assert "rooms" in result["coefficients"]

    def test_fit_linear_perfect_fit(self, engine):
        X = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        result = engine.fit_linear(X, y)
        assert result["success"]
        assert np.isclose(result["residual"], 0.0, atol=1e-10)

    def test_fit_linear_with_noise(self, engine):
        np.random.seed(42)
        X = np.linspace(0, 10, 50)
        y = 2 * X + 3 + np.random.normal(0, 0.1, 50)
        result = engine.fit_linear(X, y)
        assert result["success"]
        assert np.isclose(result["coefficients"]["X0"], 2.0, atol=0.2)
        assert np.isclose(result["coefficients"]["intercept"], 3.0, atol=0.2)


class TestFitPolynomial:
    def test_fit_polynomial_returns_structure(self, engine):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = X**2
        result = engine.fit_polynomial(X, y, degree=2)
        assert "coefficients" in result
        assert "degree" in result
        assert "success" in result
        assert "message" in result
        assert "residual" in result
        assert "solver_used" in result
        assert result["degree"] == 2

    def test_fit_polynomial_detects_singular(self, engine):
        X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = X**2
        result = engine.fit_polynomial(X, y, degree=2)
        assert not result["success"]
        assert "singular" in result["message"].lower()

    def test_fit_polynomial_different_degrees(self, engine):
        X = np.array([0.0, 1.0, 2.0])
        for degree in [1, 2, 3]:
            result = engine.fit_polynomial(X, X**2, degree=degree)
            assert result["degree"] == degree
            assert "success" in result


class TestPredict:
    def test_predict_after_fit(self, engine):
        X_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
        engine.fit_linear(X_train, y_train)
        X_new = np.array([6.0, 7.0])
        predictions = engine.predict(X_new)
        assert np.isclose(predictions, [13.0, 15.0], atol=1e-3).all()

    def test_predict_before_fit_raises(self, engine):
        X = np.array([1.0, 2.0])
        with pytest.raises(RuntimeError):
            engine.predict(X)


class TestMetrics:
    def test_r2_perfect_fit(self, engine):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(engine.calculate_r2(y_true, y_pred), 1.0, atol=1e-10)

    def test_r2_bad_fit(self, engine):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        r2 = engine.calculate_r2(y_true, y_pred)
        assert r2 < 0

    def test_rmse_perfect(self, engine):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert np.isclose(engine.calculate_rmse(y_true, y_pred), 0.0, atol=1e-10)

    def test_mae_known(self, engine):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert np.isclose(engine.calculate_mae(y_true, y_pred), 0.5, atol=1e-10)

    def test_residuals(self, engine):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([9.0, 21.0, 28.0])
        residuals = engine.calculate_residuals(y_true, y_pred)
        assert np.isclose(residuals, [1.0, -1.0, 2.0], atol=1e-10).all()

    def test_model_summary(self, engine):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        summary = engine.get_model_summary(y_true, y_pred)
        assert "r2" in summary
        assert "rmse" in summary
        assert "mae" in summary
        assert summary["n_samples"] == 5
