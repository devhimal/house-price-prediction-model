#!/usr/bin/env python3
"""Example: Root finding for target house price analysis."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from src.data.data_processor import DataProcessor
from src.regression.regression_engine import RegressionEngine
from src.numerical_methods.root_finder import RootFinder


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "house_prices.csv")

    print("=" * 60)
    print("  Root Finding: Required Area for Target Price")
    print("=" * 60)
    print()

    # --- Load data and fit a quick model ---
    dp = DataProcessor()
    result = dp.preprocess(data_path)
    X, y, feature_names = result["X"], result["y"], result["feature_names"]

    engine = RegressionEngine()
    fit_result = engine.fit_linear(X, y, feature_names=feature_names, solver="gauss_jordan")
    coeff_vals = list(fit_result["coefficients"].values())

    print("Fitted model coefficients:")
    for name, val in fit_result["coefficients"].items():
        print(f"  {name:20s} = {val:>14,.4f}")
    print()

    # --- Define the root-finding problem ---
    target_price = 20_000_000.0

    # Fix all features except "area" at their median values
    fixed_values = {}
    for i, name in enumerate(feature_names):
        fixed_values[name] = float(np.median(X[:, i]))

    intercept = coeff_vals[0]
    area_idx = feature_names.index("area")
    area_coeff = coeff_vals[area_idx + 1]

    fixed_sum = sum(
        coeff_vals[i + 1] * fixed_values[name]
        for i, name in enumerate(feature_names)
        if name != "area"
    )

    def price_minus_target(area_val):
        return intercept + area_coeff * area_val + fixed_sum - target_price

    print(f"Target price: ${target_price:,.2f}")
    print(f"Fixed features: {', '.join(f'{k}={v:.1f}' for k, v in fixed_values.items() if k != 'area')}")
    print(f"Model: Price = {intercept:,.0f} + {area_coeff:,.0f} * Area + {fixed_sum:,.0f}")
    print()

    # --- Bisection ---
    rf = RootFinder()

    result = rf.bisection(price_minus_target, a=500, b=10000)
    print("--- Bisection Method ---")
    print(f"  Root (required area): {result['root']:,.2f} sq ft")
    print(f"  Iterations:  {result['iterations']}")
    print(f"  Final error: {result['error']:.6e}")
    print(f"  Success:     {result['success']}")
    print()

    # --- Newton-Raphson ---
    df_price = lambda a: area_coeff  # derivative of linear model w.r.t. area
    result = rf.newton_raphson(price_minus_target, df_price, x0=2000.0)
    print("--- Newton-Raphson Method ---")
    print(f"  Root (required area): {result['root']:,.2f} sq ft")
    print(f"  Iterations:  {result['iterations']}")
    print(f"  Final error: {result['error']:.6e}")
    print(f"  Success:     {result['success']}")
    print()

    # --- Secant ---
    result = rf.secant(price_minus_target, x0=1000, x1=5000)
    print("--- Secant Method ---")
    print(f"  Root (required area): {result['root']:,.2f} sq ft")
    print(f"  Iterations:  {result['iterations']}")
    print(f"  Final error: {result['error']:.6e}")
    print(f"  Success:     {result['success']}")
    print()

    # --- Verify ---
    required_area = result["root"]
    predicted_price = intercept + area_coeff * required_area + fixed_sum
    print("--- Verification ---")
    print(f"  Area:    {required_area:,.2f} sq ft")
    print(f"  Price:   ${predicted_price:,.2f}")
    print(f"  Target:  ${target_price:,.2f}")
    print(f"  Match:   {abs(predicted_price - target_price) < 1.0}")


if __name__ == "__main__":
    main()
