import numpy as np
from typing import Callable


class BVPSolver:
    def __init__(self):
        pass

    def finite_difference(self, p_func: Callable, q_func: Callable, r_func: Callable,
                          a: float, b: float, alpha: float, beta: float, n: int = 20) -> dict:
        """Solve y'' + p(x)*y' + q(x)*y = r(x) using finite difference method.

        Boundary conditions: y(a) = alpha, y(b) = beta

        Discretization:
        y'' ≈ (y_{i-1} - 2y_i + y_{i+1}) / h^2
        y'  ≈ (y_{i+1} - y_{i-1}) / (2h)

        This creates a tridiagonal system solved using Thomas algorithm.

        Returns dict: x_values, y_values, n_points, method
        """
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)

        # Build tridiagonal system for interior points i = 1, ..., n-1
        size = n - 1
        sub = np.zeros(size)    # sub-diagonal
        main = np.zeros(size)   # main diagonal
        sup = np.zeros(size)    # super-diagonal
        rhs = np.zeros(size)    # right-hand side

        for i in range(size):
            xi = x[i + 1]
            p_val = p_func(xi)
            q_val = q_func(xi)
            r_val = r_func(xi)

            sub[i] = 1.0 / h**2 - p_val / (2.0 * h)
            main[i] = -2.0 / h**2 + q_val
            sup[i] = 1.0 / h**2 + p_val / (2.0 * h)
            rhs[i] = r_val

        # Apply boundary conditions
        rhs[0] -= sub[0] * alpha
        rhs[-1] -= sup[-1] * beta

        # Solve using Thomas algorithm
        y_interior = self._thomas_algorithm(sub, main, sup, rhs)

        # Full solution including boundary values
        y_values = np.zeros(n + 1)
        y_values[0] = alpha
        y_values[-1] = beta
        y_values[1:-1] = y_interior

        return {
            'x_values': x,
            'y_values': y_values,
            'n_points': n + 1,
            'method': 'finite_difference'
        }

    def shooting_method(self, f: Callable, a: float, b: float, alpha: float, beta: float,
                        n_steps: int = 100, n_shots: int = 50) -> dict:
        """Solve BVP using shooting method.

        Converts BVP to IVP: y'' = f(x, y, y')
        Guess initial slope s, solve as IVP using RK4.
        Adjust s using secant method until y(b) ≈ beta.

        Returns dict: x_values, y_values, initial_slope, n_shots, method
        """
        h = (b - a) / n_steps

        def solve_ivp(slope: float) -> float:
            x = a
            y = alpha
            dy = slope
            for _ in range(n_steps):
                y, dy = self._rk4_step(f, x, y, dy, h)
                x += h
            return y

        # Secant method to find correct initial slope
        s0 = 0.0
        s1 = 1.0
        y0 = solve_ivp(s0) - beta
        y1 = solve_ivp(s1) - beta

        slope = s1
        for _ in range(n_shots):
            if abs(y1 - y0) < 1e-12:
                break
            slope = s1 - y1 * (s1 - s0) / (y1 - y0)
            s0, y0 = s1, y1
            s1 = slope
            y1 = solve_ivp(s1)

        # Solve with the converged slope
        x_values = np.linspace(a, b, n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        y_values[0] = alpha
        dy = slope
        y = alpha
        x = a
        for i in range(n_steps):
            y, dy = self._rk4_step(f, x, y, dy, h)
            y_values[i + 1] = y
            x += h

        return {
            'x_values': x_values,
            'y_values': y_values,
            'initial_slope': slope,
            'n_shots': n_shots,
            'method': 'shooting_method'
        }

    def _rk4_step(self, f: Callable, x: float, y: float, dy: float, h: float) -> tuple:
        """Single RK4 step for second-order ODE converted to system.
        Returns (y_new, dy_new)
        """
        k1_y = dy
        k1_dy = f(x, y, dy)

        k2_y = dy + 0.5 * h * k1_dy
        k2_dy = f(x + 0.5 * h, y + 0.5 * h * k1_y, dy + 0.5 * h * k1_dy)

        k3_y = dy + 0.5 * h * k2_dy
        k3_dy = f(x + 0.5 * h, y + 0.5 * h * k2_y, dy + 0.5 * h * k2_dy)

        k4_y = dy + h * k3_dy
        k4_dy = f(x + h, y + h * k3_y, dy + h * k3_dy)

        y_new = y + (h / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        dy_new = dy + (h / 6.0) * (k1_dy + 2.0 * k2_dy + 2.0 * k3_dy + k4_dy)

        return y_new, dy_new

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                          d: np.ndarray) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm.
        a: sub-diagonal, b: main diagonal, c: super-diagonal, d: right-hand side
        Returns solution vector.
        """
        n = len(b)

        # Forward sweep
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i - 1]
            if i < n - 1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x
