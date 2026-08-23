# Testing Documentation

## Overview

The project uses **pytest** for automated testing. All tests are located in the `tests/` directory.

---

## Running Tests

### Run All Tests

```bash
pytest
```

### Run a Specific Test File

```bash
pytest tests/test_matrix_solver.py
pytest tests/test_regression.py
pytest tests/test_root_finder.py
pytest tests/test_interpolation.py
pytest tests/test_eigen_solver.py
pytest tests/test_differentiation.py
pytest tests/test_integration.py
pytest tests/test_ode.py
pytest tests/test_bvp.py
pytest tests/test_error_analysis.py
```

### Run a Specific Test Class or Method

```bash
pytest tests/test_matrix_solver.py::TestGaussJordan
pytest tests/test_matrix_solver.py::TestGaussJordan::test_gauss_jordan_basic_3x3
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Short Tracebacks

```bash
pytest -tb=short
```

---

## Test Files and Coverage

### `test_matrix_solver.py`

Tests the `MatrixSolver` class from `src/numerical_methods/matrix_solver.py`.

| Test Class | What It Covers |
|---|---|
| `TestGaussJordan` | 2×2 and 3×3 systems, singular matrices, non-square matrices, identity systems |
| `TestLUDecomposition` | 2×2 and 3×3 systems, LU factor verification (L·U = A), singular matrices, identity systems |
| `TestCholesky` | 2×2 and 3×3 symmetric positive-definite systems, non-symmetric rejection, non-positive-definite rejection, identity systems |
| `TestCompareMethods` | Agreement between all three solvers, comparison output structure, singular matrix handling |

### `test_regression.py`

Tests the `RegressionEngine` class from `src/regression/regression_engine.py`.

| Test Class | What It Covers |
|---|---|
| `TestFitLinear` | Single-feature and multi-feature regression, coefficient verification against known solutions |
| `TestFitPolynomial` | Polynomial regression with varying degrees |
| `TestPredict` | Prediction accuracy after fitting |
| `TestMetrics` | R², RMSE, MAE calculations against known values |

### `test_root_finder.py`

Tests the `RootFinder` class from `src/numerical_methods/root_finder.py`.

| Test Class | What It Covers |
|---|---|
| `TestBisection` | Known root finding, sign-change validation, convergence verification |
| `TestNewtonRaphson` | Known root finding, derivative-zero handling |
| `TestSecant` | Known root finding, convergence verification |
| `TestSimultaneousNewton` | Systems of nonlinear equations |

### `test_interpolation.py`

Tests the `InterpolationEngine` class from `src/interpolation/interpolation_engine.py`.

| Test Class | What It Covers |
|---|---|
| `TestLagrange` | Exact interpolation of known polynomials |
| `TestNewtonDividedDifference` | Coefficient table verification, value accuracy |
| `TestCubicSpline` | Smoothness at knots, boundary conditions |
| `TestLeastSquaresFit` | Regression vs interpolation distinction |

### `test_eigen_solver.py`

Tests the `EigenSolver` class from `src/numerical_methods/eigen_solver.py`.

| Test Class | What It Covers |
|---|---|
| `TestPowerMethod` | Dominant eigenvalue identification for simple matrices |
| `TestQRIteration` | All eigenvalues for symmetric and non-symmetric matrices |
| `TestQRFactorization` | Q orthogonality (Q^T Q ≈ I), R upper triangular, A ≈ QR reconstruction |

### `test_differentiation.py`

Tests the `NumericalDifferentiation` class from `src/numerical_methods/differentiation.py`.

| Test Class | What It Covers |
|---|---|
| `TestForwardDifference` | Accuracy on known functions |
| `TestBackwardDifference` | Accuracy on known functions |
| `TestCentralDifference` | Superior accuracy over forward/backward |
| `TestSecondDerivative` | Accuracy of second derivative approximation |
| `TestCompareMethods` | Method comparison output structure |

### `test_integration.py`

Tests the `NumericalIntegration` class from `src/numerical_methods/integration.py`.

| Test Class | What It Covers |
|---|---|
| `TestTrapezoidal` | Accuracy on polynomial and trigonometric functions |
| `TestSimpsonOneThird` | Higher accuracy verification |
| `TestSimpsonThreeEighth` | Accuracy verification |
| `TestGaussianQuadrature` | 2-point and 3-point accuracy |
| `TestCompareMethods` | Cross-method comparison |

### `test_ode.py`

Tests the `ODESolver` class from `src/numerical_methods/ode_solver.py`.

| Test Class | What It Covers |
|---|---|
| `TestRungeKutta4` | Exponential growth accuracy, linear ODE accuracy |
| `TestAdamsBashforth4` | Accuracy comparison with RK4 |
| `TestCompareMethods` | Error metrics between methods |

### `test_bvp.py`

Tests the `BVPSolver` class from `src/numerical_methods/bvp_solver.py`.

| Test Class | What It Covers |
|---|---|
| `TestFiniteDifference` | Known analytical solution matching, boundary condition satisfaction |
| `TestShootingMethod` | Known analytical solution matching, convergence of initial slope |

### `test_error_analysis.py`

Tests the `ErrorAnalyzer` class from `src/error_analysis/error_analyzer.py`.

| Test Class | What It Covers |
|---|---|
| `TestBasicMetrics` | Absolute, relative, percentage error |
| `TestAggregateMetrics` | MAE, RMSE, R² against known values |
| `TestResidualAnalysis` | Residual statistics, skewness, kurtosis |
| `TestErrorPropagation` | Addition and multiplication propagation formulas |

---

## Test Conventions

- All tests use `numpy` assertions (`np.isclose`, `np.allclose`) for floating-point comparisons
- Each test file adds the project root to `sys.path` for imports
- Tests are organized in classes matching the class/methods under test
- Expected values are either hand-computed, from `numpy.linalg` reference solutions, or from known analytical results
- Fixture-based setup is used where a shared solver instance is needed
