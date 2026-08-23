# Numerical Methods-Based House Price Prediction and Analysis System

<p align="center">
  <em>A comprehensive Streamlit application demonstrating the practical application of numerical methods in real estate price prediction and analysis</em>
</p>

<p align="center">
  <a href="#problem-statement">Problem Statement</a> •
  <a href="#objectives">Objectives</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#testing">Testing</a> •
  <a href="#license">License</a>
</p>

---

## Problem Statement

Predicting house prices is one of the most valuable applications of data analysis in real estate. Accurate price estimation benefits buyers, sellers, investors, and policymakers by enabling informed decision-making. Traditional pricing methods rely heavily on manual appraisal, which can be subjective, inconsistent, and time-consuming.

Numerical methods provide a robust mathematical foundation for house price prediction by transforming raw property data into actionable insights. Techniques such as least squares regression enable systematic model fitting, matrix decomposition methods (Gauss-Jordan, LU, Cholesky) solve the underlying normal equations efficiently, and root-finding algorithms facilitate target-price analysis. Interpolation methods allow estimation of prices for properties not directly observed, while eigenvalue analysis reveals the underlying structure and importance of features. Numerical differentiation quantifies price sensitivity to individual property attributes, and integration methods support aggregate analysis. Ordinary differential equation solvers model price growth over time, and boundary value problem methods address spatial price distribution.

This project systematically implements these numerical methods within a unified Streamlit application, demonstrating how computational mathematics directly supports real estate analytics.

---

## Objectives

1. **Predict house prices** using the least squares regression method as the primary prediction engine.
2. **Implement multiple numerical methods** spanning linear algebra, root finding, interpolation, differentiation, integration, eigenvalue analysis, ODEs, and BVPs.
3. **Analyze prediction errors** systematically, computing residuals, absolute errors, relative errors, mean squared error, and R² scores.
4. **Compare numerical methods** by solving the same problem (e.g., normal equations) using different techniques and benchmarking their performance.
5. **Visualize results** through interactive Streamlit charts, tables, and mathematical explanations that connect theory to practice.
6. **Bridge theory and application** by mapping every Week 1–8 course topic to a concrete software module.

---

## Course Learning Outcomes

The following table maps each weekly course topic to its corresponding implementation in this project:

| Week | Numerical Method | Project Application |
|------|-----------------|---------------------|
| 1 | Error Analysis | Prediction and residual errors |
| 2 | Gauss-Jordan Elimination | Regression normal equations |
| 2 | LU Decomposition | Regression normal equations |
| 2 | Cholesky Decomposition | Regression normal equations |
| 3 | Power Method | Feature covariance analysis |
| 3 | QR Iteration | Feature eigenvalue analysis |
| 4 | Bisection Method | Target-price analysis |
| 4 | Newton-Raphson Method | Target-price analysis |
| 4 | Secant Method | Target-price analysis |
| 4 | Simultaneous Newton Method | Nonlinear property model |
| 5 | Finite Difference Method | Price sensitivity |
| 5 | Newton-Cotes Integration | Numerical integration |
| 5 | Gaussian Quadrature | Numerical integration |
| 6 | Lagrange Interpolation | House-price interpolation |
| 6 | Newton Divided Difference | House-price interpolation |
| 6 | Cubic Spline | Smooth price interpolation |
| 6 | Least Squares Regression | Main price prediction engine |
| 7 | Runge-Kutta 4th Order (RK4) | Price-growth IVP |
| 7 | Multi-step Methods | Price-growth IVP |
| 8 | Finite Difference BVP | Property-value BVP demonstration |
| 8 | Shooting Method | Property-value BVP demonstration |

---

## System Architecture

The project is organized into three architectural layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CORE ENGINE                              │
│                                                                 │
│   Least Squares ──► Matrix Solver ──► Error Analysis            │
│                      ├── Gauss-Jordan                           │
│                      ├── LU Decomposition                       │
│                      └── Cholesky Decomposition                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ANALYSIS MODULES                          │
│                                                                 │
│   Root Finding          Interpolation         Eigenvalue        │
│   ├── Bisection         ├── Lagrange          ├── Power Method  │
│   ├── Newton-Raphson    ├── Newton Div. Diff  └── QR Iteration  │
│   ├── Secant            └── Cubic Spline                         │
│   └── Simultaneous Newton                                          │
│                                                                 │
│   Differentiation       Integration           Regression        │
│   └── Finite Difference ├── Newton-Cotes      └── Least Squares │
│                         └── Gauss Quadrature                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DEMONSTRATION LAYER                         │
│                                                                 │
│   ODE Solving             Boundary Value Problems               │
│   ├── Runge-Kutta 4       ├── Finite Difference BVP             │
│   └── Multi-step Methods  └── Shooting Method                   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

```
house-price-numerical-methods/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── data/
│   └── synthetic_houses.csv        # Generated synthetic dataset
├── core/
│   ├── __init__.py
│   ├── least_squares.py            # Least squares regression engine
│   ├── matrix_solver.py            # Gauss-Jordan, LU, Cholesky solvers
│   └── error_analysis.py           # Residual and error metrics
├── analysis/
│   ├── __init__.py
│   ├── root_finding.py             # Bisection, Newton-Raphson, Secant, Simultaneous
│   ├── interpolation.py            # Lagrange, Newton Divided Difference, Cubic Spline
│   ├── eigenvalue.py               # Power Method, QR Iteration
│   ├── differentiation.py          # Finite Difference
│   └── integration.py              # Newton-Cotes, Gaussian Quadrature
├── demonstration/
│   ├── __init__.py
│   ├── ode_solver.py               # RK4 and Multi-step IVP solvers
│   └── bvp_solver.py               # Finite Difference BVP and Shooting Method
├── utils/
│   ├── __init__.py
│   ├── data_generator.py           # Synthetic dataset generation
│   └── visualization.py            # Plotting and display helpers
└── tests/
    ├── __init__.py
    ├── test_least_squares.py
    ├── test_matrix_solver.py
    ├── test_error_analysis.py
    ├── test_root_finding.py
    ├── test_interpolation.py
    ├── test_eigenvalue.py
    ├── test_differentiation.py
    ├── test_integration.py
    ├── test_ode_solver.py
    └── test_bvp_solver.py
```

---

## Mathematical Background

### Least Squares and Normal Equations

The least squares method finds the coefficient vector **β** that minimizes the sum of squared residuals between observed and predicted values:

```
minimize ‖Aβ − b‖²
```

Setting the gradient to zero yields the normal equations:

```
(AᵀA)β = Aᵀb
```

This system is solved using one of three matrix decomposition methods implemented in the project.

### Matrix Decomposition Methods

| Method | Approach | Complexity | Stability |
|--------|----------|------------|-----------|
| **Gauss-Jordan** | Row reduction to reduced row echelon form | O(n³) | Moderate |
| **LU Decomposition** | Factors A = LU, then forward/back substitution | O(n³) | Good |
| **Cholesky** | Factors A = LLᵀ for symmetric positive-definite A | O(n³/3) | Excellent |

The Cholesky method is the most efficient for the normal equations since AᵀA is always symmetric positive-definite (when A has full column rank).

### Root Finding Methods

- **Bisection**: Bracket-based, guaranteed convergence at rate O(log₂). Robust but slow.
- **Newton-Raphson**: Quadratic convergence using f(x) and f'(x). Fast but requires derivative.
- **Secant**: Superlinear convergence (order ≈ 1.618) without requiring derivatives.
- **Simultaneous Newton**: Multi-dimensional Newton's method for systems of nonlinear equations, using the Jacobian matrix.

These methods are applied to solve for target prices given specific property constraints.

### Interpolation vs Regression

- **Interpolation** passes exactly through all data points. Suitable when data is exact or when estimating values within the range of known data.
  - *Lagrange*: Polynomial interpolation using basis polynomials.
  - *Newton Divided Difference*: Builds polynomial incrementally using divided differences.
  - *Cubic Spline*: Piecewise cubic polynomials with continuous first and second derivatives, producing smooth curves without oscillation.
- **Regression** (Least Squares) fits a model that minimizes overall error but does not necessarily pass through every point. Better for noisy real-world data.

### Eigenvalue Methods

- **Power Method**: Iteratively computes the dominant eigenvalue and eigenvector of a matrix. Applied to the feature covariance matrix to identify the most influential features.
- **QR Iteration**: Iteratively decomposes A = QR and forms A' = RQ, converging to the Schur form. Provides all eigenvalues of the feature covariance matrix.

### Numerical Differentiation and Integration

- **Finite Difference Approximations**: Approximate first and second derivatives of the price function using discrete data points. Used to compute price sensitivity to each property feature.
- **Newton-Cotes Formulas**: Approximate definite integrals using equally spaced function evaluations (trapezoidal rule, Simpson's rule).
- **Gaussian Quadrature**: Achieves higher accuracy by choosing optimal evaluation points and weights (Legendre polynomials), approximating integrals with fewer function evaluations.

### ODE and BVP Solving

- **Runge-Kutta 4th Order (RK4)**: A single-step method for initial value problems (IVPs) that achieves O(h⁴) accuracy per step.
- **Multi-step Methods**: Use information from multiple previous steps (Adams-Bashforth, Adams-Moulton) for efficient IVP solving.
- **Finite Difference BVP**: Discretizes boundary value problems into a system of linear equations and solves the resulting tridiagonal system.
- **Shooting Method**: Converts a BVP into an IVP by guessing unknown initial conditions and iterating until boundary conditions are satisfied.

---

## Dataset

The project uses a **synthetic dataset** generated programmatically to demonstrate the numerical methods under controlled conditions.

| Property | Description |
|----------|-------------|
| **Records** | 120 |
| **Features** | 8 columns |
| **Generation** | Programmatic with known mathematical relationships |

### Feature Descriptions

| Column | Description | Unit | Range |
|--------|-------------|------|-------|
| `area` | Total living area | sq ft | 600 – 4,500 |
| `bedrooms` | Number of bedrooms | count | 1 – 6 |
| `bathrooms` | Number of bathrooms | count | 1 – 4 |
| `age` | Age of the property | years | 0 – 50 |
| `parking` | Number of parking spaces | count | 0 – 3 |
| `location_score` | Neighborhood quality rating | score | 1 – 10 |
| `distance_center` | Distance to city center | km | 0.5 – 40.0 |
| `price` | Property sale price | currency | varies |

The synthetic data follows a realistic price model with controlled noise, ensuring that numerical methods can be validated against known ground truth.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/username/house-price-numerical-methods.git
cd house-price-numerical-methods

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages (listed in `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting and visualization |
| `scikit-learn` | Evaluation metrics |

---

## Usage

### Running the Application

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Launch the Streamlit application
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Application Sections

The Streamlit app provides **14 interactive sections** accessible via the sidebar navigation:

| # | Section | Numerical Methods Covered |
|---|---------|--------------------------|
| 1 | Overview | Project introduction and architecture |
| 2 | Dataset | Synthetic data exploration and statistics |
| 3 | Least Squares Regression | Core prediction engine |
| 4 | Matrix Solvers | Gauss-Jordan, LU, Cholesky comparison |
| 5 | Error Analysis | Residuals, MSE, R² evaluation |
| 6 | Root Finding | Bisection, Newton-Raphson, Secant |
| 7 | Interpolation | Lagrange, Newton Divided Difference, Cubic Spline |
| 8 | Eigenvalue Analysis | Power Method, QR Iteration |
| 9 | Numerical Differentiation | Finite Difference price sensitivity |
| 10 | Numerical Integration | Newton-Cotes, Gaussian Quadrature |
| 11 | ODE Solving | RK4 and Multi-step IVP |
| 12 | Boundary Value Problems | Finite Difference BVP, Shooting Method |
| 13 | Method Comparison | Benchmarking and accuracy analysis |
| 14 | Conclusion | Summary and future directions |

---

## Testing

### Running the Test Suite

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_least_squares.py -v
pytest tests/test_matrix_solver.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

### Test Coverage

The test suite covers all implemented numerical methods:

| Test File | Module Tested | Test Cases |
|-----------|--------------|------------|
| `test_least_squares.py` | Least squares regression | Coefficient accuracy, prediction |
| `test_matrix_solver.py` | Gauss-Jordan, LU, Cholesky | Solution correctness, decomposition |
| `test_error_analysis.py` | Error metrics | Residuals, MSE, R² computation |
| `test_root_finding.py` | Bisection, Newton-Raphson, Secant | Root accuracy, convergence |
| `test_interpolation.py` | Lagrange, Newton, Cubic Spline | Interpolation accuracy |
| `test_eigenvalue.py` | Power Method, QR Iteration | Eigenvalue/eigenvector accuracy |
| `test_differentiation.py` | Finite Difference | Derivative approximation |
| `test_integration.py` | Newton-Cotes, Gauss Quadrature | Integral approximation |
| `test_ode_solver.py` | RK4, Multi-step | IVP solution accuracy |
| `test_bvp_solver.py` | BVP, Shooting | BVP boundary condition satisfaction |

---

## Limitations

1. **Synthetic Data**: The dataset is programmatically generated rather than collected from real-world listings. While it follows realistic mathematical relationships, it may not capture all nuances of actual housing markets.

2. **Simplified Price Model**: The underlying price function is a linear combination of features with additive noise. Real estate pricing involves nonlinear interactions, market dynamics, and external factors not modeled here.

3. **Educational Purpose**: This project is designed as a teaching tool to demonstrate numerical methods. The implementations prioritize clarity and correctness over production-grade performance optimization.

4. **Static Dataset**: The data is fixed at generation time and does not update with changing market conditions.

5. **No External APIs**: The application does not connect to real estate databases, census data, or geographic information systems.

---

## Future Work

1. **Production Web Interface**: Deploy as a full-stack web application with user authentication, property listing management, and real-time predictions.

2. **Real-World Datasets**: Integrate data from public housing databases (e.g., Zillow, Realtor.com APIs) with thousands of records and diverse feature sets.

3. **Geographic Data**: Incorporate GPS coordinates, neighborhood demographics, school ratings, and proximity to amenities using geospatial analysis.

4. **Cloud Deployment**: Deploy to cloud platforms (AWS, GCP, Azure) with auto-scaling, database persistence, and CI/CD pipelines.

5. **Advanced Methods**: Implement regularization (Ridge, Lasso), polynomial regression, gradient boosting, and neural network comparison alongside classical numerical methods.

6. **Interactive Sensitivity**: Allow users to adjust individual property features and observe real-time price changes through interactive sliders.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Authors

**University Project — Numerical Methods Course**

| Role | Name |
|------|------|
| Developer | *[Your Name]* |
| Supervisor | *[Supervisor Name]* |
| Institution | *[University Name]* |
| Course | Numerical Methods |
| Semester | *[Semester/Year]* |

---

<p align="center">
  Built with Python, Streamlit, and NumPy
</p>
# house-price-prediction-model
