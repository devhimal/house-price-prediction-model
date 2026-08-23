#!/usr/bin/env python3
"""Example: Solving a 3x3 linear system using MatrixSolver."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from src.numerical_methods.matrix_solver import MatrixSolver


def main():
    solver = MatrixSolver()

    A = np.array([
        [2.0,  1.0, -1.0],
        [-3.0, -1.0,  2.0],
        [-2.0,  1.0,  2.0],
    ])
    b = np.array([8.0, -11.0, -3.0])

    print("=" * 60)
    print("  3x3 Linear System: Ax = b")
    print("=" * 60)
    print()
    print("Matrix A:")
    for row in A:
        print("  ", row)
    print()
    print("Vector b:", b)
    print()

    expected = np.linalg.solve(A, b)
    print("Reference solution (numpy):", expected)
    print()

    # --- Gauss-Jordan ---
    result = solver.gauss_jordan(A, b)
    print("--- Gauss-Jordan Elimination ---")
    print(f"  Success:    {result['success']}")
    print(f"  Solution:   {result['solution']}")
    print(f"  Residual:   {result['residual']:.2e}")
    print(f"  Time:       {result['execution_time']:.6f}s")
    print()

    # --- LU Decomposition ---
    result = solver.lu_decomposition(A, b)
    print("--- LU Decomposition (Doolittle) ---")
    print(f"  Success:    {result['success']}")
    print(f"  Solution:   {result['solution']}")
    print(f"  Residual:   {result['residual']:.2e}")
    print(f"  Time:       {result['execution_time']:.6f}s")
    print()
    print("  L matrix:")
    for row in result["L"]:
        print("    ", [f"{v:8.4f}" for v in row])
    print("  U matrix:")
    for row in result["U"]:
        print("    ", [f"{v:8.4f}" for v in row])
    print()

    # --- Cholesky ---
    # Cholesky requires symmetric positive-definite matrix
    A_spd = np.array([
        [4.0,  12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ])
    b_chol = np.array([1.0, 2.0, 3.0])

    print("--- Cholesky Decomposition (A = LL^T) ---")
    print("  Note: Requires symmetric positive-definite matrix")
    print(f"  Matrix A_spd:")
    for row in A_spd:
        print("    ", row)
    print(f"  Vector b: {b_chol}")

    result = solver.cholesky(A_spd, b_chol)
    print(f"  Success:    {result['success']}")
    print(f"  Solution:   {result['solution']}")
    print(f"  Residual:   {result['residual']:.2e}")
    print()

    # --- Method Comparison ---
    print("--- Comparing All Methods (original 3x3 system) ---")
    comparison = solver.compare_methods(A, b)
    print(f"  Reference solution: {comparison['reference_solution']}")
    print()
    for row in comparison["comparison"]:
        res_str = f"{row['residual']:.2e}" if row["residual"] is not None else "N/A"
        print(f"  {row['method']:20s}  success={row['success']}  residual={res_str}  time={row['execution_time']:.6f}s")


if __name__ == "__main__":
    main()
