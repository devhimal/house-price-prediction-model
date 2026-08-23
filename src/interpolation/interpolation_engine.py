import numpy as np


class InterpolationEngine:
    def __init__(self):
        pass

    def lagrange(self, x_data: np.ndarray, y_data: np.ndarray, x: float) -> float:
        """Lagrange interpolation.

        P(x) = sum_{i=0}^{n} y_i * L_i(x)
        where L_i(x) = prod_{j!=i} (x - x_j) / (x_i - x_j)

        Returns interpolated value.
        """
        n = len(x_data)
        result = 0.0

        for i in range(n):
            basis = 1.0
            for j in range(n):
                if j != i:
                    basis *= (x - x_data[j]) / (x_data[i] - x_data[j])
            result += y_data[i] * basis

        return result

    def newton_divided_difference(self, x_data: np.ndarray, y_data: np.ndarray, x: float) -> dict:
        """Newton divided difference interpolation.

        1. Build divided difference table
        2. Compute polynomial coefficients
        3. Evaluate at x using nested multiplication

        Returns dict: value, coefficients, divided_diff_table
        """
        n = len(x_data)

        # Build divided difference table
        table = np.zeros((n, n))
        for i in range(n):
            table[i][0] = y_data[i]

        for j in range(1, n):
            for i in range(n - j):
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_data[i + j] - x_data[i])

        # Coefficients are the top row of the table
        coefficients = [table[0][j] for j in range(n)]

        # Evaluate using nested multiplication
        result = coefficients[n - 1]
        for i in range(n - 2, -1, -1):
            result = result * (x - x_data[i]) + coefficients[i]

        return {
            "value": result,
            "coefficients": coefficients,
            "divided_diff_table": table,
        }

    def cubic_spline(self, x_data: np.ndarray, y_data: np.ndarray, x: float) -> dict:
        """Natural cubic spline interpolation.

        Solve for spline coefficients using tridiagonal system.

        Returns dict: value, spline_coefficients (a,b,c,d for each interval)

        Algorithm:
        1. Compute h_i = x_{i+1} - x_i
        2. Set up tridiagonal system for second derivatives (c coefficients)
        3. Solve tridiagonal system
        4. Compute remaining coefficients
        5. Find correct interval and evaluate
        """
        n = len(x_data) - 1  # number of intervals

        h = np.array([x_data[i + 1] - x_data[i] for i in range(n)])

        # Build tridiagonal system: A * c = rhs
        # where c are the second-derivative related coefficients
        # Size of system: (n-1) x (n-1)
        size = n - 1

        if size == 0:
            # Only 2 points: linear interpolation
            slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
            a = y_data[0]
            b = slope
            c_val = 0.0
            d_val = 0.0
            interval_idx = 0
            t = x - x_data[0]
            value = a + b * t + c_val * t**2 + d_val * t**3
            return {
                "value": value,
                "spline_coefficients": [{"a": a, "b": b, "c": c_val, "d": d_val, "x_left": x_data[0]}],
            }

        # Tridiagonal matrix coefficients
        # Main diagonal
        diag_main = np.zeros(size)
        for i in range(size):
            diag_main[i] = 2.0 * (h[i] + h[i + 1])

        # Lower diagonal
        diag_lower = np.zeros(size - 1)
        for i in range(size - 1):
            diag_lower[i] = h[i + 1]

        # Upper diagonal
        diag_upper = np.zeros(size - 1)
        for i in range(size - 1):
            diag_upper[i] = h[i + 1]

        # Right-hand side
        rhs = np.zeros(size)
        for i in range(size):
            rhs[i] = 3.0 * ((y_data[i + 2] - y_data[i + 1]) / h[i + 1] - (y_data[i + 1] - y_data[i]) / h[i])

        # Solve tridiagonal system using Thomas algorithm
        c_deriv = self._thomas_algorithm(diag_lower, diag_main, diag_upper, rhs)

        # Full c array (second derivatives related), with natural boundary conditions
        c_full = np.zeros(n + 1)
        c_full[1:n] = c_deriv
        c_full[0] = 0.0
        c_full[n] = 0.0

        # Compute a, b, d coefficients for each interval
        spline_coefficients = []
        for i in range(n):
            a_i = y_data[i]
            b_i = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (c_full[i + 1] + 2.0 * c_full[i]) / 3.0
            c_i = c_full[i]
            d_i = (c_full[i + 1] - c_full[i]) / (3.0 * h[i])
            spline_coefficients.append({
                "a": a_i,
                "b": b_i,
                "c": c_i,
                "d": d_i,
                "x_left": x_data[i],
            })

        # Find the correct interval for x
        interval_idx = 0
        for i in range(n):
            if x >= x_data[i] and x <= x_data[i + 1]:
                interval_idx = i
                break
        else:
            # Clamp to nearest interval
            if x < x_data[0]:
                interval_idx = 0
            else:
                interval_idx = n - 1

        coeff = spline_coefficients[interval_idx]
        t = x - coeff["x_left"]
        value = coeff["a"] + coeff["b"] * t + coeff["c"] * t**2 + coeff["d"] * t**3

        return {
            "value": value,
            "spline_coefficients": spline_coefficients,
        }

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm (TDMA).

        System:
            b[0]*x[0] + c[0]*x[1]                   = d[0]
            a[i]*x[i] + b[i]*x[i+1] + c[i]*x[i+2]   = d[i]   for i in 1..n-2
            a[n-2]*x[n-2] + b[n-2]*x[n-1]            = d[n-2]

        Parameters:
            a: lower diagonal (length n-1)
            b: main diagonal (length n)
            c: upper diagonal (length n-1)
            d: right-hand side (length n)

        Returns:
            solution x (length n)
        """
        n = len(b)

        # Forward sweep
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            m = b[i] - a[i - 1] * c_prime[i - 1] if i < n else b[i] - a[i - 1] * c_prime[i - 1]
            if i < n - 1:
                c_prime[i] = c[i] / m
            if i < n:
                d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / m

        # Back substitution
        x = np.zeros(n)
        x[n - 1] = d_prime[n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def least_squares_fit(self, x_data: np.ndarray, y_data: np.ndarray, degree: int,
                          x: float) -> dict:
        """Least squares polynomial fit (regression, not interpolation).

        Fit polynomial of given degree to all data points.
        Solves the normal equations: (X^T X) a = X^T y

        Returns dict: value, coefficients
        """
        n = len(x_data)
        m = degree + 1  # number of coefficients

        # Build Vandermonde matrix
        X = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                X[i][j] = x_data[i] ** j

        # Compute X^T X
        XtX = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                for k in range(n):
                    XtX[i][j] += X[k][i] * X[k][j]

        # Compute X^T y
        Xty = np.zeros(m)
        for i in range(m):
            for k in range(n):
                Xty[i] += X[k][i] * y_data[k]

        # Solve using normal equations with Gaussian elimination
        coefficients = self._solve_linear_system(XtX, Xty)

        # Evaluate polynomial at x
        result = 0.0
        for i in range(m):
            result += coefficients[i] * x**i

        return {
            "value": result,
            "coefficients": coefficients,
        }

    def _solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b using Gaussian elimination with partial pivoting.

        Parameters:
            A: coefficient matrix (m x m)
            b: right-hand side vector (length m)

        Returns:
            solution vector x (length m)
        """
        m = len(b)
        # Augmented matrix
        aug = np.zeros((m, m + 1))
        for i in range(m):
            for j in range(m):
                aug[i][j] = A[i][j]
            aug[i][m] = b[i]

        # Forward elimination with partial pivoting
        for col in range(m):
            # Find pivot
            max_row = col
            for row in range(col + 1, m):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            # Swap rows
            aug[[col, max_row]] = aug[[max_row, col]]

            # Eliminate below
            for row in range(col + 1, m):
                factor = aug[row][col] / aug[col][col]
                for j in range(col, m + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = np.zeros(m)
        for i in range(m - 1, -1, -1):
            x[i] = aug[i][m]
            for j in range(i + 1, m):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]

        return x
