#!/usr/bin/env python3
"""Example: Interpolation methods on house price data."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from src.data.data_processor import DataProcessor
from src.interpolation.interpolation_engine import InterpolationEngine


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "house_prices.csv")

    print("=" * 60)
    print("  Interpolation: House Price Estimation")
    print("=" * 60)
    print()

    # --- Load data ---
    dp = DataProcessor()
    result = dp.preprocess(data_path)
    X, y, feature_names = result["X"], result["y"], result["feature_names"]

    # --- Prepare data: sort by area and pick a subset ---
    area_idx = feature_names.index("area")
    areas = X[:, area_idx]
    sort_idx = np.argsort(areas)
    areas_sorted = areas[sort_idx]
    prices_sorted = y[sort_idx]

    n_points = 10
    step = max(1, len(areas_sorted) // n_points)
    idx = np.arange(0, len(areas_sorted), step)[:n_points]
    x_data = areas_sorted[idx]
    y_data = prices_sorted[idx]

    print(f"Using {len(x_data)} data points for interpolation:")
    for xi, yi in zip(x_data, y_data):
        print(f"  Area: {xi:>7,.0f} sq ft  ->  Price: ${yi:>14,.2f}")
    print()

    # --- Query points ---
    x_query = np.linspace(x_data.min(), x_data.max(), 20)
    interp = InterpolationEngine()

    # --- Lagrange ---
    print("--- Lagrange Interpolation ---")
    lagrange_values = [interp.lagrange(x_data, y_data, x) for x in x_query]
    for x, val in zip(x_query[::4], lagrange_values[::4]):
        print(f"  Area {x:>7,.0f} -> ${val:>14,.2f}")
    print()

    # --- Newton Divided Difference ---
    print("--- Newton Divided Difference ---")
    newton_result = interp.newton_divided_difference(x_data, y_data, x_query[0])
    print(f"  Polynomial coefficients: {[f'{c:.4e}' for c in newton_result['coefficients']]}")
    newton_values = [interp.newton_divided_difference(x_data, y_data, x)["value"] for x in x_query]
    for x, val in zip(x_query[::4], newton_values[::4]):
        print(f"  Area {x:>7,.0f} -> ${val:>14,.2f}")
    print()

    # --- Cubic Spline ---
    print("--- Cubic Spline Interpolation ---")
    spline_result = interp.cubic_spline(x_data, y_data, x_query[0])
    print(f"  Number of spline intervals: {len(spline_result['spline_coefficients'])}")
    spline_values = [interp.cubic_spline(x_data, y_data, x)["value"] for x in x_query]
    for x, val in zip(x_query[::4], spline_values[::4]):
        print(f"  Area {x:>7,.0f} -> ${val:>14,.2f}")
    print()

    # --- Least Squares Fit ---
    print("--- Least Squares Fit (degree 2) ---")
    lsq_result = interp.least_squares_fit(x_data, y_data, degree=2, x=x_query[0])
    print(f"  Coefficients: {[f'{c:.4e}' for c in lsq_result['coefficients']]}")
    lsq_values = [interp.least_squares_fit(x_data, y_data, degree=2, x=x)["value"] for x in x_query]
    for x, val in zip(x_query[::4], lsq_values[::4]):
        print(f"  Area {x:>7,.0f} -> ${val:>14,.2f}")
    print()

    # --- Compare at a single point ---
    test_area = float(np.median(x_data))
    print(f"--- Comparison at Area = {test_area:,.0f} sq ft ---")
    lagrange_val = interp.lagrange(x_data, y_data, test_area)
    newton_val = interp.newton_divided_difference(x_data, y_data, test_area)["value"]
    spline_val = interp.cubic_spline(x_data, y_data, test_area)["value"]
    lsq_val = interp.least_squares_fit(x_data, y_data, degree=2, x=test_area)["value"]

    print(f"  Lagrange:              ${lagrange_val:>14,.2f}")
    print(f"  Newton Div. Diff.:     ${newton_val:>14,.2f}")
    print(f"  Cubic Spline:          ${spline_val:>14,.2f}")
    print(f"  Least Squares (deg 2): ${lsq_val:>14,.2f}")

    vals = [lagrange_val, newton_val, spline_val, lsq_val]
    print(f"  Spread (max-min):      ${max(vals) - min(vals):>14,.2f}")


if __name__ == "__main__":
    main()
