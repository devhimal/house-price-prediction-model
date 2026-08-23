import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.error_analysis.error_analyzer import ErrorAnalyzer


@pytest.fixture
def analyzer():
    return ErrorAnalyzer()


class TestAbsoluteError:
    def test_absolute_error_basic(self, analyzer):
        assert np.isclose(analyzer.absolute_error(10.0, 10.5), 0.5, atol=1e-10)

    def test_absolute_error_zero(self, analyzer):
        assert np.isclose(analyzer.absolute_error(5.0, 5.0), 0.0, atol=1e-10)

    def test_absolute_error_negative(self, analyzer):
        assert np.isclose(analyzer.absolute_error(-3.0, -1.0), 2.0, atol=1e-10)

    def test_absolute_error_symmetric(self, analyzer):
        assert np.isclose(analyzer.absolute_error(5.0, 3.0),
                          analyzer.absolute_error(3.0, 5.0), atol=1e-10)


class TestRelativeError:
    def test_relative_error_basic(self, analyzer):
        assert np.isclose(analyzer.relative_error(100.0, 105.0), 0.05, atol=1e-10)

    def test_relative_error_zero_true(self, analyzer):
        result = analyzer.relative_error(0.0, 5.0)
        assert result == float('inf')

    def test_relative_error_both_zero(self, analyzer):
        assert np.isclose(analyzer.relative_error(0.0, 0.0), 0.0, atol=1e-10)

    def test_relative_error_exact(self, analyzer):
        assert np.isclose(analyzer.relative_error(42.0, 42.0), 0.0, atol=1e-10)


class TestPercentageError:
    def test_percentage_error_basic(self, analyzer):
        assert np.isclose(analyzer.percentage_error(100.0, 105.0), 5.0, atol=1e-10)

    def test_percentage_error_zero(self, analyzer):
        assert np.isclose(analyzer.percentage_error(50.0, 50.0), 0.0, atol=1e-10)

    def test_percentage_error_large(self, analyzer):
        assert np.isclose(analyzer.percentage_error(10.0, 20.0), 100.0, atol=1e-10)


class TestMeanAbsoluteError:
    def test_mae_perfect(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert np.isclose(analyzer.mean_absolute_error(y_true, y_pred), 0.0, atol=1e-10)

    def test_mae_known(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 5.0])
        # |1-2| + |2-3| + |3-5| = 1+1+2 = 4, MAE = 4/3
        assert np.isclose(analyzer.mean_absolute_error(y_true, y_pred), 4.0 / 3.0, atol=1e-10)

    def test_mae_symmetric(self, analyzer):
        y_true = np.array([1.0, 5.0])
        y_pred = np.array([3.0, 3.0])
        assert np.isclose(analyzer.mean_absolute_error(y_true, y_pred), 2.0, atol=1e-10)


class TestRootMeanSquaredError:
    def test_rmse_perfect(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert np.isclose(analyzer.root_mean_squared_error(y_true, y_pred), 0.0, atol=1e-10)

    def test_rmse_known(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3.0)
        assert np.isclose(analyzer.root_mean_squared_error(y_true, y_pred), expected, atol=1e-10)

    def test_rmse_greater_than_mae(self, analyzer):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 5.0])
        rmse = analyzer.root_mean_squared_error(y_true, y_pred)
        mae = analyzer.mean_absolute_error(y_true, y_pred)
        assert rmse >= mae


class TestRSquared:
    def test_r2_perfect(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(analyzer.r_squared(y_true, y_pred), 1.0, atol=1e-10)

    def test_r2_mean_predictor(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, np.mean(y_true))
        assert np.isclose(analyzer.r_squared(y_true, y_pred), 0.0, atol=1e-10)

    def test_r2_negative(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert analyzer.r_squared(y_true, y_pred) < 0


class TestMeanPercentageError:
    def test_mpe_perfect(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert np.isclose(analyzer.mean_percentage_error(y_true, y_pred), 0.0, atol=1e-10)

    def test_mpe_known(self, analyzer):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        expected = np.mean([10.0, 10.0])
        assert np.isclose(analyzer.mean_percentage_error(y_true, y_pred), expected, atol=1e-10)

    def test_mpe_zero_true_values(self, analyzer):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        result = analyzer.mean_percentage_error(y_true, y_pred)
        assert result == 0.0


class TestPredictionErrorAnalysis:
    def test_prediction_analysis_structure(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1])
        result = analyzer.prediction_error_analysis(y_true, y_pred)
        assert "individual_errors" in result
        assert "abs_individual_errors" in result
        assert "summary" in result

    def test_prediction_analysis_summary_keys(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = analyzer.prediction_error_analysis(y_true, y_pred)
        summary = result["summary"]
        assert np.isclose(summary["mae"], 0.0, atol=1e-10)
        assert np.isclose(summary["rmse"], 0.0, atol=1e-10)
        assert np.isclose(summary["r_squared"], 1.0, atol=1e-10)
        assert summary["n_samples"] == 3

    def test_prediction_analysis_individual_errors(self, analyzer):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([9.0, 22.0, 27.0])
        result = analyzer.prediction_error_analysis(y_true, y_pred)
        assert np.isclose(result["individual_errors"], [1.0, -2.0, 3.0], atol=1e-10).all()


class TestResidualAnalysis:
    def test_residual_analysis_zero_residuals(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = analyzer.residual_analysis(y_true, y_pred)
        assert np.isclose(result["stats"]["mean"], 0.0, atol=1e-10)
        assert np.isclose(result["stats"]["std"], 0.0, atol=1e-10)

    def test_residual_analysis_structure(self, analyzer):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        result = analyzer.residual_analysis(y_true, y_pred)
        assert "stats" in result
        assert "skewness" in result
        assert "excess_kurtosis" in result
        assert "is_symmetric" in result
        assert "is_mesokurtic" in result

    def test_residual_analysis_two_points(self, analyzer):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 1.5])
        result = analyzer.residual_analysis(y_true, y_pred)
        assert "stats" in result
        assert result["skewness"] == 0.0
        assert result["excess_kurtosis"] == 0.0


class TestErrorPropagation:
    def test_propagation_addition_basic(self, analyzer):
        result = analyzer.error_propagation_addition([1.0, 1.0])
        assert np.isclose(result, np.sqrt(2.0), atol=1e-10)

    def test_propagation_addition_empty(self, analyzer):
        assert np.isclose(analyzer.error_propagation_addition([]), 0.0, atol=1e-10)

    def test_propagation_addition_single(self, analyzer):
        assert np.isclose(analyzer.error_propagation_addition([3.0]), 3.0, atol=1e-10)

    def test_propagation_multiplication_basic(self, analyzer):
        result = analyzer.error_propagation_multiplication([10.0, 10.0], [1.0, 1.0])
        expected = 100.0 * np.sqrt(0.01 + 0.01)
        assert np.isclose(result, expected, atol=1e-10)

    def test_propagation_multiplication_mismatch(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.error_propagation_multiplication([1.0], [1.0, 2.0])

    def test_propagation_multiplication_zero_value(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.error_propagation_multiplication([0.0, 1.0], [1.0, 1.0])
