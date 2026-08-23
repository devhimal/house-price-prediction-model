#!/usr/bin/env python3
"""Example: Regression pipeline on house price data."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from src.data.data_processor import DataProcessor
from src.regression.regression_engine import RegressionEngine


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "house_prices.csv")

    print("=" * 60)
    print("  Regression Pipeline: House Price Prediction")
    print("=" * 60)
    print()

    # --- Load and preprocess data ---
    dp = DataProcessor()
    result = dp.preprocess(data_path)

    X = result["X"]
    y = result["y"]
    feature_names = result["feature_names"]
    summary = result["summary"]

    print("Dataset Summary:")
    print(f"  Records:  {summary['records']}")
    print(f"  Features: {summary['features']} ({', '.join(feature_names)})")
    print(f"  Target:   {summary['target']}")
    print()

    # --- Train/test split ---
    np.random.seed(42)
    X_train, X_test, y_train, y_test = dp.train_test_split(X, y, test_size=0.2)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print()

    # --- Fit linear regression with Gauss-Jordan ---
    engine = RegressionEngine()
    result = engine.fit_linear(X_train, y_train, feature_names=feature_names, solver="gauss_jordan")

    print("--- Linear Regression (Gauss-Jordan Solver) ---")
    print(f"  Success: {result['success']}")
    print(f"  Residual norm: {result['residual']:.2f}")
    print()
    print("  Coefficients:")
    for name, value in result["coefficients"].items():
        print(f"    {name:20s} = {value:>14,.4f}")
    print()

    # --- Predictions and metrics ---
    y_pred = engine.predict(X_test)
    metrics = engine.get_model_summary(y_test, y_pred)

    print("  Model Performance on Test Set:")
    print(f"    R² Score: {metrics['r2']:.6f}")
    print(f"    RMSE:     {metrics['rmse']:>14,.2f}")
    print(f"    MAE:      {metrics['mae']:>14,.2f}")
    print()

    # --- Compare solvers ---
    print("--- Solver Comparison ---")
    for solver_name in ["gauss_jordan", "lu_decomposition", "cholesky"]:
        eng = RegressionEngine()
        res = eng.fit_linear(X_train, y_train, feature_names=feature_names, solver=solver_name)
        if res["success"]:
            yp = eng.predict(X_test)
            r2 = eng.calculate_r2(y_test, yp)
            rmse = eng.calculate_rmse(y_test, yp)
            print(f"  {solver_name:20s}  R²={r2:.6f}  RMSE={rmse:>14,.2f}")
        else:
            print(f"  {solver_name:20s}  FAILED: {res.get('message', '')}")

    print()

    # --- Example prediction ---
    sample_house = X_test[0:1]
    predicted_price = engine.predict(sample_house)[0]
    actual_price = y_test[0]
    print("--- Example Prediction ---")
    print(f"  Features: {dict(zip(feature_names, sample_house[0]))}")
    print(f"  Actual price:    ${actual_price:>14,.2f}")
    print(f"  Predicted price: ${predicted_price:>14,.2f}")
    print(f"  Error:           ${abs(actual_price - predicted_price):>14,.2f}")


if __name__ == "__main__":
    main()
