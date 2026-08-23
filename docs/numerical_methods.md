# Mathematical Background

This document provides the mathematical foundation for all numerical methods implemented in the project.

---

## 1. Linear Algebra: Solving Ax = b

### 1.1 Least Squares and Normal Equations

Given an overdetermined system Ax = b (more equations than unknowns), we find the best approximate solution by minimizing the squared residual ||Ax - b||².

Taking the gradient and setting it to zero:

```
∇ ||Ax - b||² = 2 A^T (Ax - b) = 0
A^T A x = A^T b
```

This produces the **normal equations**:

```
X^T X β = X^T y
```

where X is the design matrix (with a column of 1s for the intercept), y is the target vector, and β is the coefficient vector.

**In our project:** `RegressionEngine.fit_linear()` constructs X^T X and X^T y, then delegates the linear solve to `MatrixSolver`.

### 1.2 Gauss-Jordan Elimination

Solves Ax = b by transforming the augmented matrix [A | b] to reduced row echelon form.

**Algorithm:**
1. Form the augmented matrix [A | b]
2. For each column j = 0, 1, ..., n-1:
   a. **Partial pivoting:** Find the row with the largest |A[i][j]| below the diagonal; swap rows
   b. **Normalize:** Divide the pivot row by the pivot element A[j][j]
   c. **Eliminate:** For every other row i ≠ j, subtract A[i][j] times the pivot row
3. The right side of the augmented matrix is the solution

**Computational cost:** O(n³) operations

**Example (2×2):**

```
[2  1 | 5]     [1  0 | 1]     x₁ = 1
[4  3 | 11] →  [0  1 | 3]     x₂ = 3
```

### 1.3 LU Decomposition (Doolittle's Method)

Factors A = LU where:
- **L** is lower triangular with 1s on the diagonal
- **U** is upper triangular

**Algorithm:**
```
For i = 0 to n-1:
    U[i][j] = A[i][j] - Σ(k=0 to i-1) L[i][k] * U[k][j]    for j = i to n-1
    L[j][i] = (A[j][i] - Σ(k=0 to i-1) L[j][k] * U[k][i]) / U[i][i]    for j = i+1 to n-1
```

Then solve two triangular systems:
1. **Forward substitution:** Solve Ly = b
   ```
   y[i] = (b[i] - Σ(j=0 to i-1) L[i][j] * y[j]) / L[i][i]
   ```
2. **Back substitution:** Solve Ux = y
   ```
   x[i] = (y[i] - Σ(j=i+1 to n-1) U[i][j] * x[j]) / U[i][i]
   ```

**Advantage:** Once A = LU is computed, solving for multiple RHS vectors costs only O(n²) per vector instead of O(n³).

### 1.4 Cholesky Decomposition (A = LL^T)

For symmetric positive-definite matrices A, we can decompose A = LL^T where L is lower triangular.

**Algorithm:**
```
For i = 0 to n-1:
    L[i][i] = sqrt(A[i][i] - Σ(k=0 to i-1) L[i][k]²)
    L[j][i] = (A[j][i] - Σ(k=0 to i-1) L[j][k] * L[i][k]) / L[i][i]    for j = i+1 to n-1
```

**Requirements:**
- A must be symmetric: A = A^T
- A must be positive definite: all eigenvalues > 0, equivalently all leading principal minors > 0

**Advantage:** Approximately half the computation and storage of LU decomposition. Numerically more stable for well-conditioned symmetric systems.

---

## 2. Eigenvalue Problems

### 2.1 Power Method

Finds the **dominant eigenvalue** (largest in absolute value) and its corresponding eigenvector.

**Algorithm:**
1. Start with a random vector v₀ (normalized)
2. Iterate:
   ```
   w = A · vₖ
   λₖ = ||w||        (eigenvalue estimate)
   vₖ₊₁ = w / λₖ     (normalized eigenvector)
   ```
3. Stop when |λₖ - λₖ₋₁| < tolerance

**Convergence rate:** Linear, proportional to |λ₂/λ₁| where λ₂ is the second-largest eigenvalue.

**Limitation:** Only finds the single dominant eigenvalue. If the dominant eigenvalue has multiplicity > 1, or if |λ₁| = |λ₂|, convergence may fail.

### 2.2 QR Iteration

Finds **all eigenvalues** by iteratively factoring and recombining.

**Algorithm:**
1. A₀ = A
2. Repeat:
   ```
   Q, R = QR_factorization(Aₖ)     (Aₖ = Q · R)
   Aₖ₊₁ = R · Q                     (similar matrix)
   ```
3. Eigenvalues converge to the diagonal of Aₖ

**With Wilkinson Shift:**
Compute shift μ from the bottom-right 2×2 block of Aₖ:
```
a = Aₖ[n-2,n-2],  b = Aₖ[n-2,n-1],  c = Aₖ[n-1,n-2],  d = Aₖ[n-1,n-1]
trace = a + d
det = a·d - b·c
discriminant = trace² - 4·det
μ = eigenvalue of 2×2 block closest to d
```
Then factor (Aₖ - μI) = QR and form Aₖ₊₁ = RQ + μI.

**QR Factorization (Modified Gram-Schmidt):**
For each column j of A:
```
v = A[:, j]
For i = 0 to j-1:
    R[i][j] = Q[:, i]^T · v
    v = v - R[i][j] · Q[:, i]
R[j][j] = ||v||
Q[:, j] = v / R[j][j]
```

**Convergence:** Cubic with Wilkinson shift for real eigenvalues. The off-diagonal norm decreases to zero, making the matrix diagonal.

---

## 3. Root Finding

### 3.1 Bisection Method

Based on the Intermediate Value Theorem: if f is continuous on [a,b] and f(a)·f(b) < 0, there exists at least one root in (a,b).

**Algorithm:**
1. Verify f(a)·f(b) < 0
2. Repeat:
   ```
   c = (a + b) / 2
   if f(c) = 0 or (b - a)/2 < tolerance: return c
   if f(a)·f(c) < 0: b = c
   else: a = c
   ```

**Convergence:** Linear. Error halves each iteration: |eₙ| ≤ (b-a)/2ⁿ

**Iterations needed:** n ≥ log₂((b-a)/ε) for tolerance ε

### 3.2 Newton-Raphson Method

Uses the tangent line at each iteration to find the next approximation.

**Formula:**
```
xₙ₊₁ = xₙ - f(xₙ) / f'(xₙ)
```

**Derivation:** From the Taylor expansion f(x) ≈ f(xₙ) + f'(xₙ)(x - xₙ), set f(x) = 0 and solve for x.

**Convergence:** Quadratic (error squares each step) when:
- f'(root) ≠ 0
- Starting guess is sufficiently close to the root

**Risk:** May diverge if:
- Starting point is far from the root
- f'(xₙ) ≈ 0 (flat tangent)

### 3.3 Secant Method

Approximates the derivative using finite differences of function values.

**Formula:**
```
xₙ₊₁ = xₙ - f(xₙ) · (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))
```

**Convergence:** Superlinear with order φ ≈ 1.618 (golden ratio).

**Advantage:** No derivative needed (unlike Newton-Raphson).
**Disadvantage:** Requires two initial guesses; may not converge for all functions.

---

## 4. Interpolation

### 4.1 Lagrange Interpolation

Given n+1 data points (x₀,y₀), (x₁,y₁), ..., (xₙ,yₙ), the Lagrange polynomial is:

```
P(x) = Σᵢ₌₀ⁿ yᵢ · Lᵢ(x)
```

where the basis polynomials are:

```
Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ) / (xᵢ - xⱼ)
```

**Properties:**
- Passes exactly through all data points
- Unique polynomial of degree ≤ n
- Computationally O(n²) per evaluation
- Prone to Runge's phenomenon for equidistant points with high n

### 4.2 Newton Divided Difference

An equivalent polynomial form using divided differences:

```
P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
```

**Divided difference table:**
```
x₀  f[x₀]
         f[x₀,x₁] = (f[x₁] - f[x₀]) / (x₁ - x₀)
x₁  f[x₁]         f[x₀,x₁,x₂] = (f[x₁,x₂] - f[x₀,x₁]) / (x₂ - x₀)
         f[x₁,x₂]
x₂  f[x₂]
```

**Advantages over Lagrange:**
- Coefficients are the top row of the divided difference table
- Adding a new data point requires only one new column of computations
- Evaluated using nested multiplication (Horner's method): O(n) per evaluation

### 4.3 Cubic Spline Interpolation

Fits piecewise cubic polynomials Sᵢ(x) between each pair of data points, ensuring:
- Sᵢ(xᵢ) = yᵢ and Sᵢ(xᵢ₊₁) = yᵢ₊₁ (interpolation)
- S, S', S'' are continuous at interior knots

**For each interval [xᵢ, xᵢ₊₁]:**
```
Sᵢ(x) = aᵢ + bᵢ(x - xᵢ) + cᵢ(x - xᵢ)² + dᵢ(x - xᵢ)³
```

**Coefficient computation:**
1. Set hᵢ = xᵢ₊₁ - xᵢ
2. Solve a tridiagonal system for the c coefficients (second-derivative related):
   ```
   2(h₀+h₁)c₁ + h₁c₂ = 6[(y₂-y₁)/h₁ - (y₁-y₀)/h₀]
   hᵢ₋₁cᵢ₋₁ + 2(hᵢ₋₁+hᵢ)cᵢ + hᵢcᵢ₊₁ = 6[(yᵢ₊₁-yᵢ)/hᵢ - (yᵢ-yᵢ₋₁)/hᵢ₋₁]
   ```
3. Apply natural boundary conditions: c₀ = cₙ = 0
4. Compute remaining coefficients:
   ```
   aᵢ = yᵢ
   bᵢ = (yᵢ₊₁ - yᵢ)/hᵢ - hᵢ(cᵢ₊₁ + 2cᵢ)/3
   dᵢ = (cᵢ₊₁ - cᵢ) / (3hᵢ)
   ```

**Tridiagonal system solved via Thomas Algorithm (TDMA):**
```
Forward sweep:
    c'₀ = c₀/b₀,  d'₀ = d₀/b₀
    c'ᵢ = cᵢ / (bᵢ - aᵢ·c'ᵢ₋₁)
    d'ᵢ = (dᵢ - aᵢ·d'ᵢ₋₁) / (bᵢ - aᵢ·c'ᵢ₋₁)

Back substitution:
    xₙ = d'ₙ
    xᵢ = d'ᵢ - c'ᵢ·xᵢ₊₁
```

---

## 5. Numerical Differentiation

### 5.1 Forward Difference

```
f'(x) ≈ [f(x+h) - f(x)] / h
```

**Error:** O(h) — derived from Taylor expansion:
```
f(x+h) = f(x) + hf'(x) + (h²/2)f''(ξ)
```

### 5.2 Backward Difference

```
f'(x) ≈ [f(x) - f(x-h)] / h
```

**Error:** O(h) — same order as forward difference.

### 5.3 Central Difference

```
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
```

**Error:** O(h²) — the h² terms cancel in the Taylor expansion:
```
f(x+h) = f(x) + hf'(x) + (h²/2)f''(x) + (h³/6)f'''(x) + ...
f(x-h) = f(x) - hf'(x) + (h²/2)f''(x) - (h³/6)f'''(x) + ...
f(x+h) - f(x-h) = 2hf'(x) + O(h³)
```

Central difference is preferred for better accuracy at the same step size.

### 5.4 Second Derivative (Central Difference)

```
f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
```

**Error:** O(h²)

---

## 6. Numerical Integration

### 6.1 Trapezoidal Rule

Approximates the integral by summing areas of trapezoids:

```
∫ₐᵇ f(x)dx ≈ h/2 · [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]
```

where h = (b-a)/n.

**Error:** O(h²) — proportional to f''(ξ)h²

### 6.2 Simpson's 1/3 Rule

Fits parabolas through consecutive triples of points (n must be even):

```
∫ₐᵇ f(x)dx ≈ h/3 · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]
```

**Error:** O(h⁴) — exact for polynomials up to degree 3

### 6.3 Simpson's 3/8 Rule

Uses cubic interpolation over groups of 4 points (n must be a multiple of 3):

```
∫ₐᵇ f(x)dx ≈ 3h/8 · [f(x₀) + 3f(x₁) + 3f(x₂) + 2f(x₃) + 3f(x₄) + ... + f(xₙ)]
```

**Error:** O(h⁴) — similar accuracy to 1/3 rule but uses different grouping

### 6.4 Gaussian Quadrature

Instead of equally spaced points, selects optimal points (roots of Legendre polynomials) and weights.

**Transformation to [a,b]:**
```
∫ₐᵇ f(x)dx = (b-a)/2 · ∫₋₁¹ f((b-a)t/2 + (b+a)/2) dt ≈ (b-a)/2 · Σ wᵢ · f(xᵢ)
```

**2-point Gauss-Legendre:**
```
Points:  x = ±1/√3 ≈ ±0.5774
Weights: w = 1.0, 1.0
Exact for polynomials up to degree 3
```

**3-point Gauss-Legendre:**
```
Points:  x = 0, ±√(3/5) ≈ ±0.7746
Weights: w = 8/9, 5/9, 5/9
Exact for polynomials up to degree 5
```

---

## 7. Ordinary Differential Equations

### 7.1 Runge-Kutta 4th Order (RK4)

Solves the initial value problem dy/dt = f(t, y), y(t₀) = y₀.

**Algorithm:**
```
k₁ = h · f(tₙ, yₙ)
k₂ = h · f(tₙ + h/2, yₙ + k₁/2)
k₃ = h · f(tₙ + h/2, yₙ + k₂/2)
k₄ = h · f(tₙ + h, yₙ + k₃)

yₙ₊₁ = yₙ + (k₁ + 2k₂ + 2k₃ + k₄) / 6
```

**Error:** O(h⁴) per step, O(h⁴) global (single-step method)

**Interpretation:** Weighted average of four slope estimates (left endpoint, two midpoints, right endpoint).

### 7.2 Adams-Bashforth 4-Step Method

A multi-step explicit method that uses the four most recent function evaluations.

**Formula:**
```
yₙ₊₁ = yₙ + h/24 · [55f(tₙ, yₙ) - 59f(tₙ₋₁, yₙ₋₁) + 37f(tₙ₋₂, yₙ₋₂) - 9f(tₙ₋₃, yₙ₋₃)]
```

**Bootstrapping:** The first 3 steps are computed using RK4 to provide the necessary starting values.

**Advantage:** Only one function evaluation per step (after bootstrapping), compared to 4 for RK4.
**Disadvantage:** Not self-starting; requires stored history; may be less stable than single-step methods.

---

## 8. Boundary Value Problems

### 8.1 Finite Difference Method

Solves y'' + p(x)y' + q(x)y = r(x) on [a, b] with y(a) = α, y(b) = β.

**Discretization:** Replace derivatives at interior points xᵢ with finite differences:
```
y''(xᵢ) ≈ (yᵢ₋₁ - 2yᵢ + yᵢ₊₁) / h²
y'(xᵢ)  ≈ (yᵢ₊₁ - yᵢ₋₁) / (2h)
```

**Substituting into the ODE** at each interior point yields a tridiagonal system:
```
(1/h² - pᵢ/2h) · yᵢ₋₁ + (-2/h² + qᵢ) · yᵢ + (1/h² + pᵢ/2h) · yᵢ₊₁ = rᵢ
```

The boundary conditions are applied by modifying the first and last equations of the system. The tridiagonal system is then solved using the Thomas algorithm.

### 8.2 Shooting Method

Converts the BVP to an initial value problem (IVP) and adjusts the unknown initial condition.

**Algorithm:**
1. Convert y'' = f(x, y, y') to a system: let u₁ = y, u₂ = y'
   ```
   du₁/dx = u₂
   du₂/dx = f(x, u₁, u₂)
   ```
2. Guess initial slope s₀, solve IVP with y(a) = α, y'(a) = s₀ using RK4
3. Check if y(b) ≈ β
4. Use the **secant method** to adjust s:
   ```
   sₙ₊₁ = sₙ - [y(b; sₙ) - β] · (sₙ - sₙ₋₁) / [y(b; sₙ) - y(b; sₙ₋₁)]
   ```
5. Repeat until |y(b) - β| < tolerance

**Advantage:** Reuses existing IVP solvers; no need to construct a new linear system.
**Disadvantage:** May require many "shots" for stiff problems; convergence depends on the quality of initial guesses.

---

## 9. Error Analysis Metrics

### Mean Absolute Error (MAE)
```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

### Root Mean Squared Error (RMSE)
```
RMSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]
```

### R² Score (Coefficient of Determination)
```
R² = 1 - SS_res / SS_tot
SS_res = Σ (yᵢ - ŷᵢ)²
SS_tot = Σ (yᵢ - ȳ)²
```
R² = 1.0 means perfect prediction; R² = 0.0 means the model is no better than predicting the mean.

### Error Propagation

For z = x + y with independent errors σₓ and σᵧ:
```
σ_z = √(σₓ² + σᵧ²)
```

For z = x · y:
```
σ_z/|z| = √((σₓ/x)² + (σᵧ/y)²)
```
