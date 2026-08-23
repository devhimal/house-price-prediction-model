# Viva / Cross-Questions Document

> **Numerical Methods in Scientific Computing — House Price Prediction Project**
> Comprehensive Q&A covering all modules, algorithms, and design decisions.

---

## Category 1: General Project Understanding

**Q1. Why did you implement numerical algorithms from scratch instead of using libraries like scikit-learn?**

The primary purpose is academic learning — implementing from scratch forces understanding of the underlying mathematics, numerical stability issues, and algorithmic trade-offs. In production, libraries are preferred, but this project demonstrates mastery of the fundamentals by solving a real regression problem without any ML framework abstractions.

**Q2. What is the scope of the project and what does it deliver?**

The project predicts house prices from 7 numerical features using hand-coded linear regression solved by Gauss-Jordan, LU decomposition, and Cholesky-based methods. It includes root finding, interpolation, eigenvalue analysis, numerical integration, differentiation, ODE solvers, and boundary value problems — all packaged in a 14-page Streamlit web application with 181 passing unit tests and an R² of 0.8696.

**Q3. How does this differ from a typical ML project that uses scikit-learn or TensorFlow?**

In a typical ML project, you call `model.fit(X, y)` and the library handles matrix decomposition, memory layout, and convergence internally. Here, every step — forming the normal equations, decomposing matrices, back-substituting, computing residuals — is written explicitly. This exposes issues like singular matrices, condition numbers, and floating-point precision that library calls hide.

**Q4. What were the main challenges you faced during implementation?**

Three key challenges: (1) ensuring numerical stability when solving the normal equations, since X^T X can be poorly conditioned; (2) handling edge cases such as singular matrices and zero denominators in root-finding methods; (3) getting all 181 tests to pass while keeping the code clean and free of external ML dependencies.

**Q5. What did you learn most from this project?**

The deepest insight was how sensitive results are to the choice of numerical method. Gauss-Jordan, LU, and Cholesky all solve the same system but have very different computational costs and stability profiles. I also learned that condition number — not just the algorithm — determines whether a solution is trustworthy.

**Q6. What real-world applications could this system be extended to?**

The same numerical pipeline applies to any linear regression domain: insurance pricing, salary prediction, energy consumption forecasting, or medical cost estimation. The root-finding module is directly applicable to inverse problems — for example, finding the required building area to meet a target sale price. Interpolation is useful for filling missing data points in sensor readings.

**Q7. How would you scale this system for larger datasets?**

For datasets with millions of rows, forming X^T X explicitly becomes prohibitive. I would switch to iterative solvers (conjugate gradient, stochastic gradient descent) and use sparse matrix representations. The current architecture already separates the solver interface from the model, so swapping in an iterative backend is straightforward.

**Q8. What is the single most important takeaway from building this project?**

Numerical methods are not abstract theory — they are engineering tools with real trade-offs. The choice between Gauss-Jordan and Cholesky isn't just about speed; it's about whether the matrix is symmetric positive definite, how much memory you have, and how sensitive the result is to rounding errors. Understanding these trade-offs is what separates a practitioner from a library user.

---

## Category 2: Linear Algebra & Matrix Solvers

**Q9. Describe the Gauss-Jordan elimination algorithm in your own words.**

Gauss-Jordan converts the augmented matrix [A|b] into reduced row echelon form by: (1) for each pivot row i, find the largest element in column i below the diagonal (partial pivoting); (2) swap it into position; (3) scale the pivot row so the diagonal element becomes 1; (4) eliminate all other entries in column i by row subtraction. The result is the identity matrix on the left and the solution vector on the right.

**Q10. What is LU decomposition (Doolittle's method) and why is it useful?**

Doolittle decomposes A into a lower triangular L (with 1s on the diagonal) and upper triangular U such that A = LU. This is useful because solving Ax = b becomes two cheap steps: forward substitution Ly = b, then back substitution Ux = y. For multiple right-hand sides, the decomposition is done once and reused.

**Q11. When would you choose Cholesky over LU decomposition?**

Cholesky is chosen when A is symmetric positive definite (SPD) — it decomposes A into LL^T, requiring roughly half the operations of LU (n³/6 vs n³/3) and half the storage. In house price regression, X^T X is theoretically SPD, but in practice, multicollinearity or ill-conditioning can violate this, causing Cholesky to fail — which is exactly what happens in this project.

**Q12. Why does the project need three different matrix solvers?**

Each solver serves a different purpose: Gauss-Jordan is conceptually simple and works on any invertible matrix; LU is computationally efficient and handles multiple right-hand sides; Cholesky is the fastest but only works on SPD matrices. Using all three demonstrates that different problems favor different methods, and it validates that they produce identical results.

**Q13. What is the condition number and why does it matter in this project?**

The condition number κ(A) = ||A|| · ||A⁻¹|| measures how sensitive the solution x is to small changes in A or b. If κ(A) is large (e.g., > 10⁶), the matrix is ill-conditioned and small floating-point errors get amplified. In this project, X^T X has a high condition number due to feature correlation, which is why the solvers are tested and their numerical stability matters.

**Q14. What happens if the matrix is singular (non-invertible)?**

A singular matrix means at least one row/column is linearly dependent on others, so the system either has no solution or infinitely many. In Gauss-Jordan, a zero pivot is encountered during elimination. In LU, the pivot becomes zero. The implementation must detect this (e.g., pivot below tolerance) and raise an error rather than producing garbage.

**Q15. How do the normal equations (X^T X β = X^T y) arise in linear regression?**

Linear regression minimizes the sum of squared residuals ||y - Xβ||². Taking the gradient with respect to β and setting it to zero gives the normal equations: X^T X β = X^T y. This is the linear system we solve for β. An alternative is the QR decomposition approach (β = (R⁻¹ Q^T y)), which is numerically more stable.

**Q16. Compare the computational complexity of the three solvers.**

Gauss-Jordan elimination: O(n³) with no reuse. LU decomposition: O(n³/3) for the decomposition, then O(n²) per back-substitution, reusable for multiple b. Cholesky: O(n³/6) for the decomposition, the fastest when applicable. For our problem (n ≈ 8 features), the difference is negligible, but for large n, Cholesky would be ~6× faster than Gauss-Jordan.

---

## Category 3: Regression & Model Performance

**Q17. State the linear regression formula and explain what each coefficient represents.**

The model is: price = β₀ + β₁·area + β₂·bedrooms + β₃·bathrooms + β₄·age + β₅·garage + β₆·location_score + β₇·distance_center. For example, β₁ ≈ 287.99 means each additional square foot of area adds approximately $288 to the predicted price, holding all other features constant.

**Q18. What does an R² of 0.8696 mean in plain English?**

R² = 0.8696 means approximately 87% of the variance in house prices is explained by the 7 features in the model. The remaining 13% is due to factors not captured (school quality, renovation status, market conditions, etc.). An R² above 0.8 is generally considered good for house price prediction.

**Q19. Write the R² formula and explain it.**

R² = 1 - (SS_res / SS_tot), where SS_res = Σ(yᵢ - ŷᵢ)² is the residual sum of squares and SS_tot = Σ(yᵢ - ȳ)² is the total sum of squares. SS_tot measures total variance; SS_res measures unexplained variance. R² = 1 means perfect fit; R² = 0 means the model is no better than predicting the mean.

**Q20. What is RMSE and why use it alongside R²?**

RMSE = √(Σ(yᵢ - ŷᵢ)² / n) measures the average prediction error in the original units (dollars). R² is unitless and relative. RMSE is absolute — it tells you that the average prediction is off by some dollar amount. Both are needed: R² tells you the model explains most variance; RMSE tells you the practical error magnitude.

**Q21. Why do all three solvers (Gauss-Jordan, LU, Cholesky) give the same R²?**

They all solve the same system of linear equations X^T X β = X^T y. The mathematical solution β is unique (assuming non-singularity), so any correct solver must produce the same β. The differences are in computational cost, numerical precision, and memory usage — not in the mathematical answer.

**Q22. What is the difference between the normal equations approach and gradient descent for solving regression?**

The normal equations solve for β analytically in one step (O(n³)). Gradient descent iteratively updates β by moving in the direction that reduces the loss (O(n·k) where k is iterations). Normal equations are exact but require matrix inversion; gradient descent is approximate but scales better to very large datasets and doesn't require forming X^T X.

**Q23. Could this model be overfitting or underfitting? How would you tell?**

With R² = 0.8696 and only 7 features on 2000 records, overfitting is unlikely — the feature-to-sample ratio is low. To confirm, we compare training vs test R²; if they differ significantly, overfitting is occurring. Underfitting would show low R² on both sets. Residual plots (pattern vs random) would also reveal model inadequacy.

**Q24. What does residual analysis reveal about model quality?**

Residuals (yᵢ - ŷᵢ) should be randomly distributed around zero with no visible pattern. If residuals show a curve, the relationship is non-linear and a linear model is inadequate. If residuals fan out (heteroscedasticity), prediction uncertainty varies with the target. If residuals are heavy-tailed, outliers are skewing the model. We check all three.

---

## Category 4: Root Finding

**Q25. What are the prerequisites for the bisection method?**

The function f(x) must be continuous on the interval [a, b] and f(a) and f(b) must have opposite signs (f(a)·f(b) < 0). This guarantees (by the Intermediate Value Theorem) that at least one root exists in [a, b]. Bisection halves the interval each iteration, converging linearly with a guaranteed rate.

**Q26. Explain the Newton-Raphson method and its convergence properties.**

Newton-Raphson iterates: x_{n+1} = x_n - f(x_n) / f'(x_n). It has quadratic convergence (the error roughly squares each iteration), meaning it doubles the correct digits each step. However, it requires the derivative f'(x) and can diverge if the initial guess is far from the root or if f'(x) ≈ 0.

**Q27. What advantage does the secant method have over Newton-Raphson?**

The secant method approximates the derivative using two previous points: f'(x) ≈ (f(x_n) - f(x_{n-1})) / (x_n - x_{n-1}). This eliminates the need to analytically compute f'(x), which is valuable when the derivative is complex or unknown. It converges with order ≈ 1.618 (superlinear), slightly slower than Newton-Raphson's quadratic rate.

**Q28. How does the simultaneous Newton method work for systems of equations?**

For a system F(x) = 0 where F: ℝⁿ → ℝⁿ, simultaneous Newton iterates: x_{n+1} = x_n - J(x_n)⁻¹ F(x_n), where J is the Jacobian matrix of partial derivatives. Each iteration requires solving a linear system (not explicitly inverting J). Convergence is quadratic when J is non-singular near the root.

**Q29. Why is root finding used in this project?**

Root finding solves the inverse problem: given a target price, find the required area (or other feature value) that achieves it. We define f(area) = price(area) - target_price and find the root. This is practically useful — a buyer can ask "what size house can I afford for $500K?" and get a direct answer.

**Q30. Compare convergence rates of bisection, secant, and Newton-Raphson.**

Bisection: linear convergence, order 1 — error halves each step; reliable but slow. Secant: superlinear convergence, order ≈ 1.618 — faster than bisection, no derivative needed. Newton-Raphson: quadratic convergence, order 2 — fastest when it works, but requires f'(x) and a good initial guess. For our 1D root-finding, all three converge in fewer than 50 iterations.

---

## Category 5: Interpolation

**Q31. What is the difference between Lagrange interpolation and Newton's divided difference interpolation?**

Both produce the same polynomial of degree ≤ n through n+1 points, but they differ numerically. Lagrange uses explicit basis polynomials: L(x) = Σ yᵢ · ℓᵢ(x). Newton uses a recursive divided-difference table: P(x) = a₀ + a₁(x-x₀) + a₂(x-x₀)(x-x₁) + ... Newton is preferred in practice because adding a new point doesn't require recomputing all coefficients.

**Q32. What is cubic spline interpolation and why is it preferred over high-degree polynomials?**

A cubic spline fits piecewise cubic polynomials between adjacent data points, with continuity of the first and second derivatives at the knots. Unlike a single high-degree polynomial, splines avoid Runge's phenomenon (wild oscillations at interval edges). The result is a smooth curve that interpolates all points without overshooting.

**Q33. What is Runge's phenomenon and when does it occur?**

Runge's phenomenon is the oscillation of high-degree polynomial interpolants near the edges of the interval, even when the underlying function is smooth. It occurs when equally spaced points are used with polynomial degrees above ~10. For example, interpolating f(x) = 1/(1+25x²) with 20 equally spaced points produces massive oscillations at the endpoints.

**Q34. When would you use least squares approximation instead of interpolation?**

Interpolation forces the curve through every data point, which amplifies noise. Least squares minimizes the overall fit error, allowing the curve to approximate rather than exact-fit. Use interpolation when data is exact (e.g., mathematical tables) and least squares when data has measurement noise (e.g., sensor readings, survey data).

**Q35. What was the comparison result between interpolation methods in this project?**

The interpolation comparison across Lagrange, Newton, and Cubic Spline showed a spread of approximately $69,000 in predicted prices at the same feature values. This illustrates that while all methods produce the same result at the data points, their behavior between and beyond points can differ significantly — underscoring why cubic spline is preferred for smooth, stable interpolation.

---

## Category 6: Advanced Methods

**Q36. What is eigenvalue analysis used for in this project?**

Eigenvalue analysis of the correlation matrix reveals multicollinearity — how strongly features correlate with each other. If eigenvalues are near zero, columns of X are nearly linearly dependent, making X^T X ill-conditioned. The eigenvectors identify the principal directions of variance in the feature space (PCA), which can guide dimensionality reduction.

**Q37. What is the power method and how does it differ from QR iteration?**

The power method finds only the dominant eigenvalue (largest magnitude) by repeatedly multiplying a vector by A and normalizing. QR iteration finds all eigenvalues simultaneously by factorizing A = QR, then forming A' = RQ, repeating until convergence. Power method: simple, one eigenvalue, linear convergence. QR: complex, all eigenvalues, cubic convergence.

**Q38. Compare forward, backward, and central finite differences for numerical differentiation.**

Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h, error O(h). Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h, error O(h). Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h), error O(h²). Central is most accurate because the first-order error terms cancel. Too small h causes floating-point cancellation error; too large h truncation error.

**Q39. What is the difference between trapezoidal, Simpson's, and Gaussian quadrature?**

Trapezoidal approximates the integral using linear segments between points: error O(h²). Simpson's uses quadratic polynomials over each pair of segments: error O(h⁴). Gaussian quadrature places points and weights optimally to exactly integrate polynomials of degree 2n-1 using n points: highest accuracy per evaluation. For smooth functions, Gaussian quadrature converges exponentially.

**Q40. Explain the difference between RK4 and Adams-Bashforth for ODE solvers.**

RK4 is a single-step explicit method: it evaluates f at 4 points per step and combines them with specific weights, achieving 4th-order accuracy. Adams-Bashforth is a multi-step method: it uses previous function evaluations to predict the next step, requiring fewer function evaluations per step but needing a startup phase (e.g., from RK4). RK4 is more stable; Adams-Bashforth is more efficient for long integrations.

**Q41. What boundary value problems (BVPs) are solved and how?**

The project solves second-order ODE BVPs using the finite difference method. The domain is discretized into n points, derivatives are replaced with central difference approximations, and the resulting tridiagonal system is solved. For example, solving y'' = f(x, y, y') with y(a) = α and y(b) = β reduces to a linear system that can be solved by Thomas algorithm (specialized tridiagonal LU).

---

## Category 7: Testing & Software Engineering

**Q42. Why 181 tests — what do they cover?**

The 181 tests cover every numerical method in the project: matrix solver correctness (Gauss-Jordan, LU, Cholesky), regression model accuracy, root-finding convergence, interpolation accuracy at known points, eigenvalue computation, differentiation and integration against analytical solutions, ODE solver convergence rates, and edge cases (singular matrices, zero inputs, empty arrays). Each test verifies against a known mathematical result.

**Q43. How do you verify that a numerical algorithm is correct?**

Three strategies: (1) test against closed-form solutions (e.g., derivative of x² is 2x, integral of x² is x³/3); (2) test known properties (e.g., R² must be in [0, 1], eigenvalues must satisfy det(A - λI) = 0); (3) compare implementations — if three solvers produce the same answer, the probability of all three being wrong in the same way is negligible.

**Q44. How are singular matrices tested?**

The test suite includes matrices with determinant zero (e.g., rows [1,2; 2,4]) and verifies that each solver raises a descriptive error rather than producing NaN or garbage. This ensures the code handles edge cases gracefully. The tolerance threshold for singularity detection is carefully chosen to balance false positives against missed singularities.

**Q45. What edge cases are tested beyond singular matrices?**

Edge cases include: single-element systems (n=1), empty inputs, very large condition numbers, NaN/Inf in input data, zero variance features, perfect collinearity (duplicate features), single data point (n=1 samples), and mismatched array dimensions. Each is expected to either produce a correct result or raise a specific error.

**Q46. Do you follow test-driven development (TDD)?**

Partially. For each new numerical method, I first wrote the mathematical specification (expected input/output), then wrote a test implementing that specification, then wrote the code to pass it. This ensured correctness from the start. However, some integration tests (verifying that the Streamlit app runs) were written after the implementation.

---

## Category 8: Data & Feature Engineering

**Q47. Why were these 7 features chosen and what does each represent?**

Area (sq ft), bedrooms, bathrooms, age (years), garage (capacity), location_score (1-10 composite of amenities), and distance_center (km from city center). These are the most commonly used features in real estate valuation models and capture the key drivers of price: size, condition, location, and accessibility.

**Q48. How are location_score and distance_center derived?**

location_score is a composite metric (1-10) derived from proximity to schools, hospitals, parks, shopping, and public transport, each weighted by importance. distance_center is the Euclidean distance from the property to the city center in kilometers. Both are derived from geospatial data during preprocessing.

**Q49. What data preprocessing steps are applied?**

The pipeline includes: (1) handling missing values via median imputation; (2) outlier detection using IQR filtering; (3) feature scaling via standardization (z-score normalization); (4) correlation analysis to identify multicollinearity. No categorical encoding is needed because all features are numerical.

**Q50. Why a 70/30 train/test split?**

The 70/30 split is a standard convention that provides enough training data (1400 samples) for the model to learn, while reserving 600 samples for unbiased evaluation. With 2000 records, 1400 training samples gives ~200 samples per feature (the rule-of-thumb minimum is 10-20 per feature), which is sufficient.

---

## Category 9: Streamlit Application

**Q51. Why Streamlit over Flask or Django?**

Streamlit was chosen because it's purpose-built for data science dashboards — it requires minimal boilerplate, handles session state automatically, and produces interactive web apps from pure Python. Flask/Django would require writing HTML/CSS/JS templates, which is unnecessary for a numerical methods demonstration. Streamlit's widget callbacks align naturally with parameter exploration.

**Q52. Describe the 14 pages and their roles.**

The pages are organized by method category: (1) Home/Overview, (2) Data Exploration with interactive plots, (3-5) Linear Algebra solvers (Gauss-Jordan, LU, Cholesky), (6) Regression with coefficient display, (7) Root Finding with target price search, (8-10) Interpolation methods, (11) Eigenvalue Analysis, (12) Numerical Integration, (13) ODE Solver, (14) Testing Dashboard showing 181 test results. Each page is self-contained with inputs and visualizations.

**Q53. How is session state managed across pages?**

Streamlit's `st.session_state` dictionary persists data across page navigations. When a user enters parameters on one page and navigates away, the results are stored in session state. This allows the regression model trained on one page to be used by the root-finding page without re-computation. Cache decorators (`@st.cache_data`) further prevent redundant calculations.

**Q54. How would the architecture change if this were deployed as a production API?**

The numerical core (solvers, models, interpolation) would be extracted into a pure Python library with no Streamlit dependency. The Streamlit app becomes one frontend; a FastAPI backend would serve the same core via REST endpoints. This separation allows mobile apps, batch scripts, and other clients to use the same numerical engine.

---

## Category 10: Tricky / Advanced Questions

**Q55. Why does Cholesky decomposition fail on the normal equations X^T X β = X^T y?**

X^T X is symmetric positive semi-definite, but multicollinearity (correlated features) makes it positive semi-definite rather than strictly positive definite. When eigenvalues are near zero, the Cholesky factorization L L^T encounters a zero or negative pivot and fails. This is a real diagnostic signal — it tells you the features are too correlated and you need regularization (Ridge/Lasso) or feature removal.

**Q56. What is the condition number of X^T X in this project and what does it imply?**

The condition number is large (typically 10⁴ to 10⁸ depending on scaling), meaning the problem is moderately ill-conditioned. This implies that small perturbations in the features (measurement noise of ~1%) could change the coefficients by a much larger percentage. However, the prediction accuracy (R² = 0.8696) remains good because the ill-conditioning affects coefficient estimation more than prediction.

**Q57. How would you extend this to polynomial regression?**

Polynomial regression creates additional features: x₁², x₁x₂, x₂², etc. The design matrix X grows but the method is identical — solve the normal equations with the expanded X. The risk is overfitting as the number of polynomial terms grows. Ridge regularization (adding λI to X^T X) controls this, and the condition number of X^T X + λI improves.

**Q58. What are the fundamental limitations of linear regression for house prices?**

(1) It assumes a linear relationship, but price-area curves are often logarithmic. (2) It cannot capture interactions (e.g., a pool adds more value in Arizona than Alaska). (3) It's sensitive to outliers — one mansion skews the coefficients. (4) It cannot handle categorical features without encoding. (5) It doesn't model market dynamics (time, supply, demand).

**Q59. How would you handle categorical features (e.g., neighborhood, house style)?**

Categorical features require encoding: one-hot encoding creates binary columns (neighborhood_A, neighborhood_B), ordinal encoding assigns integers for ordered categories. One-hot is preferred for nominal categories (no natural order). However, one-hot encoding increases dimensionality, which worsens the condition number of X^T X — a direct trade-off between model expressiveness and numerical stability.

**Q60. If you had to pick one improvement to make the model more accurate, what would it be?**

Feature engineering — specifically, adding polynomial and interaction terms with regularization. For example, area × location_score captures that large houses in premium locations are disproportionately expensive. Ridge regression (L2 regularization) would prevent overfitting from these additional terms while improving the condition number of X^T X. This is the single highest-impact change.

---

*Document prepared for viva voce examination. All answers reference implementation details from the project codebase.*
