# System Architecture

## Overview

This project implements numerical methods for house price prediction using a three-tier architecture. All algorithms are implemented from scratch using only NumPy for array operations — no `numpy.linalg` solvers are used for core computations.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEMONSTRATION TIER                           │
│                                                                     │
│  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────────┐   │
│  │  ODE Solver  │  │  Multi-step ODE │  │    BVP Solver         │   │
│  │  (RK4)       │  │  (Adams-Bash)   │  │  (Finite Difference   │   │
│  │              │  │                 │  │   & Shooting Method)  │   │
│  └──────┬───────┘  └───────┬─────────┘  └───────────┬───────────┘   │
│         │                  │                         │               │
│         └──────────────────┼─────────────────────────┘               │
│                            │                                         │
├────────────────────────────┼─────────────────────────────────────────┤
│                      ANALYSIS TIER                                   │
│                            │                                         │
│  ┌────────────┐ ┌─────────┴──┐ ┌───────────┐ ┌─────────┐ ┌──────┐  │
│  │ Root       │ │Interpolation│ │ Eigenvalue│ │ Diff.   │ │ Integ│  │
│  │ Finding    │ │  Engine    │ │  Solver   │ │         │ │      │  │
│  │            │ │            │ │           │ │         │ │      │  │
│  │•Bisection  │ │•Lagrange   │ │•Power Mthd│ │•Forward │ │•Trap.│  │
│  │•Newton-R   │ │•Newton Div │ │•QR Iter.  │ │•Backward│ │•Simp.│  │
│  │•Secant     │ │•Cubic Spline│ │           │ │•Central │ │•Gauss│  │
│  └─────┬──────┘ │•LSQ Fit   │ └─────┬─────┘ └────┬────┘ └──┬───┘  │
│        │        └─────┬──────┘       │            │         │       │
│        └──────────────┼──────────────┘            │         │       │
│                       │                           │         │       │
├───────────────────────┼───────────────────────────┼─────────┼───────┤
│                   CORE TIER                       │         │       │
│                       │                           │         │       │
│  ┌────────────────────▼───────────────────────────▼─────────▼───┐   │
│  │              Regression Engine                                │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │         Normal Equations: X^T X β = X^T y           │    │   │
│  │  └──────────────────────┬───────────────────────────────┘    │   │
│  │                         │                                    │   │
│  │  ┌──────────────────────▼───────────────────────────────┐    │   │
│  │  │              Matrix Solver                            │    │   │
│  │  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐   │    │   │
│  │  │  │ Gauss-   │  │ LU Decomp.   │  │ Cholesky     │   │    │   │
│  │  │  │ Jordan   │  │ (Doolittle)  │  │ (A = LL^T)   │   │    │   │
│  │  │  └──────────┘  └──────────────┘  └──────────────┘   │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  │                         │                                    │   │
│  │  ┌──────────────────────▼───────────────────────────────┐    │   │
│  │  │              Error Analysis                           │    │   │
│  │  │  MAE, RMSE, R², Residual Analysis, Error Propagation│    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    SUPPORTING MODULES                                │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ Data Processor  │  │ Visualization  │  │  Streamlit App       │  │
│  │ (Load/Clean/   │  │ Plotter        │  │  (app.py)            │  │
│  │  Preprocess)   │  │                │  │                      │  │
│  └────────────────┘  └────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tier Descriptions

### CORE Tier

The foundation layer providing linear algebra solvers and regression capabilities.

#### Matrix Solver (`src/numerical_methods/matrix_solver.py`)

Implements three methods for solving linear systems **Ax = b** without `numpy.linalg`:

| Method | Algorithm | Use Case |
|--------|-----------|----------|
| `gauss_jordan` | Gauss-Jordan elimination with partial pivoting | General square systems |
| `lu_decomposition` | Doolittle's method (A = LU) | Systems with multiple RHS vectors |
| `cholesky` | Cholesky decomposition (A = LL^T) | Symmetric positive-definite matrices |

Each method returns a result dict containing: `solution`, `residual`, `execution_time`, `success`, and `message`.

#### Regression Engine (`src/regression/regression_engine.py`)

Fits linear and polynomial regression models by solving the normal equations:

```
X^T X β = X^T y
```

- `fit_linear(X, y)`: Multiple linear regression with intercept
- `fit_polynomial(X, y, degree)`: Polynomial regression on a single feature
- `predict(X)`: Generate predictions from fitted model
- Model evaluation: `calculate_r2()`, `calculate_rmse()`, `calculate_mae()`, `get_model_summary()`

#### Error Analysis (`src/error_analysis/error_analyzer.py`)

Comprehensive error metrics and diagnostics:

- Point metrics: `absolute_error()`, `relative_error()`, `percentage_error()`
- Aggregate metrics: `mean_absolute_error()`, `root_mean_squared_error()`, `r_squared()`, `mean_percentage_error()`
- Diagnostics: `prediction_error_analysis()`, `residual_analysis()`
- Error propagation: `error_propagation_addition()`, `error_propagation_multiplication()`

---

### ANALYSIS TIER

Numerical analysis algorithms built on top of the core solvers.

#### Root Finding (`src/numerical_methods/root_finder.py`)

Finds roots of f(x) = 0 using three iterative methods:

| Method | Convergence | Requires Derivative | Notes |
|--------|-------------|---------------------|-------|
| `bisection` | Linear (O(1) bits/iter) | No | Guaranteed convergence if sign change exists |
| `newton_raphson` | Quadratic | Yes (f') | Fast but may diverge |
| `secant` | Superlinear (~1.618) | No | Approximates derivative from secant line |

Also includes `simultaneous_newton()` for systems of nonlinear equations.

#### Interpolation (`src/interpolation/interpolation_engine.py`)

Estimates values between known data points:

| Method | Description |
|--------|-------------|
| `lagrange` | Lagrange polynomial interpolation via basis polynomials |
| `newton_divided_difference` | Newton form using divided difference table (nested multiplication) |
| `cubic_spline` | Natural cubic spline with tridiagonal system solved via Thomas algorithm |
| `least_squares_fit` | Least squares polynomial fit (regression, not interpolation) |

#### Eigenvalue Solver (`src/numerical_methods/eigen_solver.py`)

Computes eigenvalues and eigenvectors:

| Method | Description |
|--------|-------------|
| `power_method` | Finds the dominant eigenvalue and corresponding eigenvector iteratively |
| `qr_iteration` | QR iteration with Wilkinson shift to find all eigenvalues |

Uses a manual QR factorization via modified Gram-Schmidt.

#### Numerical Differentiation (`src/numerical_methods/differentiation.py`)

Approximates derivatives using finite differences:

| Method | Formula | Accuracy |
|--------|---------|----------|
| `forward_difference` | f'(x) ≈ (f(x+h) - f(x)) / h | O(h) |
| `backward_difference` | f'(x) ≈ (f(x) - f(x-h)) / h | O(h) |
| `central_difference` | f'(x) ≈ (f(x+h) - f(x-h)) / (2h) | O(h²) |
| `second_derivative` | f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h² | O(h²) |

Includes `price_sensitivity()` for analyzing how house prices respond to feature changes.

#### Numerical Integration (`src/numerical_methods/integration.py`)

Computes definite integrals:

| Method | Formula | Notes |
|--------|---------|-------|
| `trapezoidal` | Area of trapezoids | Simple, O(h²) error |
| `simpson_one_third` | Parabolic arcs (n must be even) | O(h⁴) error |
| `simpson_three_eighth` | Cubic arcs (n must be multiple of 3) | O(h⁴) error |
| `gaussian_quadrature_2point` | 2-point Gauss-Legendre | Exact for polynomials up to degree 3 |
| `gaussian_quadrature_3point` | 3-point Gauss-Legendre | Exact for polynomials up to degree 5 |

---

### DEMONSTRATION TIER

Applied numerical methods demonstrating ODE and BVP solving.

#### ODE Solver (`src/numerical_methods/ode_solver.py`)

Solves ordinary differential equations dy/dt = f(t, y):

| Method | Type | Order | Description |
|--------|------|-------|-------------|
| `runge_kutta_4` | Single-step | 4th order | Classic RK4 with weighted slope averages |
| `adams_bashforth_4` | Multi-step | 4th order | Uses 4 previous function evaluations; bootstraps first 3 steps with RK4 |

#### BVP Solver (`src/numerical_methods/bvp_solver.py`)

Solves boundary value problems y'' + p(x)y' + q(x)y = r(x):

| Method | Description |
|--------|-------------|
| `finite_difference` | Discretizes derivatives, builds tridiagonal system, solves via Thomas algorithm |
| `shooting_method` | Converts BVP to IVP, adjusts initial slope via secant method until boundary condition is met |

---

## Data Flow

### Regression Pipeline

```
CSV File
   │
   ▼
DataProcessor.load_csv() ───► DataProcessor.handle_missing_values()
   │                                  │
   │                                  ▼
   │                          DataProcessor.remove_duplicates()
   │                                  │
   │                                  ▼
   │                          X (features), y (target)
   │                                  │
   ▼                                  ▼
RegressionEngine.fit_linear(X, y)
   │
   ├──► Build design matrix [1 | X]
   │
   ├──► Compute A = X^T X,  b = X^T y
   │
   ├──► MatrixSolver.gauss_jordan(A, b)  ─── or lu_decomposition / cholesky
   │         │
   │         ▼
   │    solution β (coefficients)
   │
   ├──► RegressionEngine.predict(X_new)
   │         │
   │         ▼
   │    ŷ = [1 | X_new] β
   │
   └──► ErrorAnalyzer.prediction_error_analysis(y_true, y_pred)
              │
              ▼
         MAE, RMSE, R², residuals
```

### Root Finding Pipeline (Target Price Analysis)

```
Regression Model (coefficients β)
   │
   ▼
Define: f(Area) = β₀ + β₁·Area + Σ(βᵢ·fixedᵢ) - TargetPrice
   │
   ▼
RootFinder.bisection(f, a, b)  ─── or newton_raphson / secant
   │
   ▼
Required Area for target price
```

### Interpolation Pipeline

```
House Data (area, price)
   │
   ▼
Sort by area, select subset of points
   │
   ├──► InterpolationEngine.lagrange(x_data, y_data, x_query)
   ├──► InterpolationEngine.newton_divided_difference(x_data, y_data, x_query)
   ├──► InterpolationEngine.cubic_spline(x_data, y_data, x_query)
   └──► InterpolationEngine.least_squares_fit(x_data, y_data, degree, x_query)
           │
           ▼
     Estimated price at x_query
```

### ODE / BVP Demonstration Pipeline

```
Price Growth Model:  dP/dt = r · P
   │
   ├──► ODESolver.runge_kutta_4(f, t_span, P₀, n_steps)
   ├──► ODESolver.adams_bashforth_4(f, t_span, P₀, n_steps)
   └──► ODESolver.compare_methods(f, t_span, P₀, n_steps, P_exact)
           │
           ▼
     Price trajectory over time

BVP:  y'' + p(x)y' + q(x)y = r(x),  y(a)=α, y(b)=β
   │
   ├──► BVPSolver.finite_difference(p, q, r, a, b, α, β, n)
   └──► BVPSolver.shooting_method(f, a, b, α, β, n_steps)
           │
           ▼
     Solution y(x) on [a, b]
```

---

## Directory Structure

```
new-one/
├── app.py                          # Streamlit web application
├── data/
│   └── house_prices.csv            # Dataset (121 records, 8 features)
├── docs/
│   ├── architecture.md             # This document
│   ├── numerical_methods.md        # Mathematical background
│   └── testing.md                  # Testing documentation
├── examples/
│   ├── matrix_example.py           # Matrix solver demonstration
│   ├── regression_example.py       # Regression pipeline demo
│   ├── root_example.py             # Root finding demo
│   └── interpolation_example.py    # Interpolation demo
├── src/
│   ├── data/
│   │   └── data_processor.py       # Data loading and preprocessing
│   ├── error_analysis/
│   │   └── error_analyzer.py       # Error metrics and diagnostics
│   ├── interpolation/
│   │   └── interpolation_engine.py # Lagrange, Newton, Cubic Spline
│   ├── numerical_methods/
│   │   ├── matrix_solver.py        # Gauss-Jordan, LU, Cholesky
│   │   ├── root_finder.py          # Bisection, Newton-Raphson, Secant
│   │   ├── eigen_solver.py         # Power Method, QR Iteration
│   │   ├── differentiation.py      # Finite difference methods
│   │   ├── integration.py          # Simpson, Trapezoidal, Gauss Quad
│   │   ├── ode_solver.py           # RK4, Adams-Bashforth
│   │   └── bvp_solver.py           # Finite Difference, Shooting
│   ├── regression/
│   │   └── regression_engine.py    # Linear/Polynomial regression
│   └── visualization/
│       └── plotter.py              # Matplotlib chart generators
└── tests/
    ├── test_matrix_solver.py
    ├── test_regression.py
    ├── test_root_finder.py
    ├── test_interpolation.py
    ├── test_eigen_solver.py
    ├── test_differentiation.py
    ├── test_integration.py
    ├── test_ode.py
    ├── test_bvp.py
    └── test_error_analysis.py
```
