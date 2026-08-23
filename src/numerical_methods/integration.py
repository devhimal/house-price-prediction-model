import numpy as np


class NumericalIntegration:
    def __init__(self):
        pass

    def trapezoidal(self, f, a: float, b: float, n: int = 100) -> dict:
        h = (b - a) / n
        x = [a + i * h for i in range(n + 1)]
        fx = [f(xi) for xi in x]
        integral = h / 2 * (fx[0] + 2 * sum(fx[1:-1]) + fx[-1])
        return {
            "result": integral,
            "n_intervals": n,
            "h": h,
            "method": "Trapezoidal"
        }

    def simpson_one_third(self, f, a: float, b: float, n: int = 100) -> dict:
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        x = [a + i * h for i in range(n + 1)]
        fx = [f(xi) for xi in x]
        odd_sum = sum(fx[i] for i in range(1, n, 2))
        even_sum = sum(fx[i] for i in range(2, n, 2))
        integral = h / 3 * (fx[0] + 4 * odd_sum + 2 * even_sum + fx[-1])
        return {
            "result": integral,
            "n_intervals": n,
            "h": h,
            "method": "Simpson's 1/3"
        }

    def simpson_three_eighth(self, f, a: float, b: float, n: int = 100) -> dict:
        remainder = n % 3
        if remainder != 0:
            n += 3 - remainder
        h = (b - a) / n
        x = [a + i * h for i in range(n + 1)]
        fx = [f(xi) for xi in x]
        integral = fx[0] + fx[-1]
        for i in range(1, n):
            if i % 3 == 0:
                integral += 2 * fx[i]
            else:
                integral += 3 * fx[i]
        integral *= 3 * h / 8
        return {
            "result": integral,
            "n_intervals": n,
            "h": h,
            "method": "Simpson's 3/8"
        }

    def gaussian_quadrature_2point(self, f, a: float, b: float) -> dict:
        mid = (a + b) / 2
        half_len = (b - a) / 2
        inv_sqrt3 = 1.0 / np.sqrt(3)
        x1 = mid - half_len * inv_sqrt3
        x2 = mid + half_len * inv_sqrt3
        weights = [1.0, 1.0]
        integral = half_len * (f(x1) + f(x2))
        return {
            "result": integral,
            "method": "Gaussian Quadrature (2-point)",
            "points": [x1, x2],
            "weights": weights
        }

    def gaussian_quadrature_3point(self, f, a: float, b: float) -> dict:
        mid = (a + b) / 2
        half_len = (b - a) / 2
        sqrt_3_5 = np.sqrt(3.0 / 5.0)
        x0 = mid - half_len * sqrt_3_5
        x1 = mid
        x2 = mid + half_len * sqrt_3_5
        weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        integral = half_len * (weights[0] * f(x0) + weights[1] * f(x1) + weights[2] * f(x2))
        return {
            "result": integral,
            "method": "Gaussian Quadrature (3-point)",
            "points": [x0, x1, x2],
            "weights": weights
        }

    def compare_methods(self, f, a: float, b: float, exact: float = None) -> dict:
        trap = self.trapezoidal(f, a, b)
        s13 = self.simpson_one_third(f, a, b)
        s38 = self.simpson_three_eighth(f, a, b)
        g2 = self.gaussian_quadrature_2point(f, a, b)
        g3 = self.gaussian_quadrature_3point(f, a, b)

        results = {
            "Trapezoidal": trap["result"],
            "Simpson's 1/3": s13["result"],
            "Simpson's 3/8": s38["result"],
            "Gaussian 2-point": g2["result"],
            "Gaussian 3-point": g3["result"],
        }

        if exact is not None:
            results["errors"] = {
                "Trapezoidal": abs(exact - trap["result"]),
                "Simpson's 1/3": abs(exact - s13["result"]),
                "Simpson's 3/8": abs(exact - s38["result"]),
                "Gaussian 2-point": abs(exact - g2["result"]),
                "Gaussian 3-point": abs(exact - g3["result"]),
            }
            results["exact"] = exact

        return results
