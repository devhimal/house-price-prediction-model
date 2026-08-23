import numpy as np
from typing import Callable, Optional


class ODESolver:
    """Numerical solver for ordinary differential equations using RK4 and Adams-Bashforth methods."""

    def __init__(self) -> None:
        pass

    def runge_kutta_4(
        self,
        f: Callable[[float, float], float],
        t_span: tuple[float, float],
        y0: float,
        n_steps: int = 100,
    ) -> dict:
        """Solve dy/dt = f(t,y) using RK4 method.

        k1 = h * f(t_n, y_n)
        k2 = h * f(t_n + h/2, y_n + k1/2)
        k3 = h * f(t_n + h/2, y_n + k2/2)
        k4 = h * f(t_n + h, y_n + k3)
        y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6

        Returns dict: t_values, y_values, n_steps, method
        """
        t_start, t_end = t_span
        h = (t_end - t_start) / n_steps

        t_values = np.linspace(t_start, t_end, n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        y_values[0] = y0

        for i in range(n_steps):
            t_n = t_values[i]
            y_n = y_values[i]

            k1 = h * f(t_n, y_n)
            k2 = h * f(t_n + h / 2, y_n + k1 / 2)
            k3 = h * f(t_n + h / 2, y_n + k2 / 2)
            k4 = h * f(t_n + h, y_n + k3)

            y_values[i + 1] = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return {
            "t_values": t_values,
            "y_values": y_values,
            "n_steps": n_steps,
            "method": "Runge-Kutta 4",
        }

    def adams_bashforth_4(
        self,
        f: Callable[[float, float], float],
        t_span: tuple[float, float],
        y0: float,
        n_steps: int = 100,
    ) -> dict:
        """Solve dy/dt = f(t,y) using 4-step Adams-Bashforth method.

        Uses RK4 for first 3 steps, then:
        y_{n+1} = y_n + h/24 * [55*f(t_n,y_n) - 59*f(t_{n-1},y_{n-1}) +
                                  37*f(t_{n-2},y_{n-2}) - 9*f(t_{n-3},y_{n-3})]

        Returns dict: t_values, y_values, n_steps, method
        """
        t_start, t_end = t_span
        h = (t_end - t_start) / n_steps

        t_values = np.linspace(t_start, t_end, n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        y_values[0] = y0

        # Bootstrap first 3 steps with RK4
        num_bootstrap = min(3, n_steps)
        for i in range(num_bootstrap):
            t_n = t_values[i]
            y_n = y_values[i]

            k1 = h * f(t_n, y_n)
            k2 = h * f(t_n + h / 2, y_n + k1 / 2)
            k3 = h * f(t_n + h / 2, y_n + k2 / 2)
            k4 = h * f(t_n + h, y_n + k3)

            y_values[i + 1] = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Adams-Bashforth 4-step for remaining steps
        for i in range(num_bootstrap, n_steps):
            t_n = t_values[i]
            y_n = y_values[i]

            f0 = f(t_n, y_n)
            f1 = f(t_values[i - 1], y_values[i - 1])
            f2 = f(t_values[i - 2], y_values[i - 2])
            f3 = f(t_values[i - 3], y_values[i - 3])

            y_values[i + 1] = y_n + h / 24 * (55 * f0 - 59 * f1 + 37 * f2 - 9 * f3)

        return {
            "t_values": t_values,
            "y_values": y_values,
            "n_steps": n_steps,
            "method": "Adams-Bashforth 4",
        }

    def compare_methods(
        self,
        f: Callable[[float, float], float],
        t_span: tuple[float, float],
        y0: float,
        n_steps: int = 100,
        exact_solution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> dict:
        """Compare RK4 and Adams-Bashforth.

        Returns dict with both methods' results and comparison.
        """
        rk4_result = self.runge_kutta_4(f, t_span, y0, n_steps)
        ab4_result = self.adams_bashforth_4(f, t_span, y0, n_steps)

        comparison: dict = {
            "rk4": rk4_result,
            "adams_bashforth_4": ab4_result,
        }

        if exact_solution is not None:
            exact_values = exact_solution(rk4_result["t_values"])
            comparison["exact_values"] = exact_values

            rk4_error = np.max(np.abs(rk4_result["y_values"] - exact_values))
            ab4_error = np.max(np.abs(ab4_result["y_values"] - exact_values))
            comparison["rk4_max_error"] = rk4_error
            comparison["ab4_max_error"] = ab4_error
            comparison["rk4_l2_error"] = np.sqrt(np.mean((rk4_result["y_values"] - exact_values) ** 2))
            comparison["ab4_l2_error"] = np.sqrt(np.mean((ab4_result["y_values"] - exact_values) ** 2))

        return comparison
