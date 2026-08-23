import numpy as np
from typing import Callable, List, Dict, Optional


class RootFinder:
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def bisection(self, f: Callable[[float], float], a: float, b: float,
                  tolerance: Optional[float] = None,
                  max_iterations: Optional[int] = None) -> Dict:
        """Find root of f(x) = 0 in [a,b] using bisection method.
        Validates f(a)*f(b) < 0.
        Returns dict: root, iterations, error, convergence_history, success, message
        """
        tol = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        convergence_history = []

        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            return {
                "root": None,
                "iterations": 0,
                "error": None,
                "convergence_history": convergence_history,
                "success": False,
                "message": "f(a) and f(b) must have opposite signs"
            }

        if abs(fa) < tol:
            return {
                "root": a,
                "iterations": 0,
                "error": abs(fa),
                "convergence_history": convergence_history,
                "success": True,
                "message": "a is already a root"
            }

        if abs(fb) < tol:
            return {
                "root": b,
                "iterations": 0,
                "error": abs(fb),
                "convergence_history": convergence_history,
                "success": True,
                "message": "b is already a root"
            }

        root = a
        for i in range(1, max_iter + 1):
            root = (a + b) / 2.0
            fr = f(root)
            error = (b - a) / 2.0
            convergence_history.append(error)

            if abs(fr) < tol or error < tol:
                return {
                    "root": root,
                    "iterations": i,
                    "error": error,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "Converged"
                }

            if fa * fr < 0:
                b = root
                fb = fr
            else:
                a = root
                fa = fr

        return {
            "root": root,
            "iterations": max_iter,
            "error": (b - a) / 2.0,
            "convergence_history": convergence_history,
            "success": False,
            "message": "Maximum iterations reached without convergence"
        }

    def newton_raphson(self, f: Callable[[float], float],
                       df: Callable[[float], float], x0: float,
                       tolerance: Optional[float] = None,
                       max_iterations: Optional[int] = None) -> Dict:
        """Find root using Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n)
        Returns dict: root, iterations, error, convergence_history, success, message
        """
        tol = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        convergence_history = []

        x = x0
        for i in range(1, max_iter + 1):
            fx = f(x)
            dfx = df(x)

            if abs(dfx) < 1e-15:
                return {
                    "root": x,
                    "iterations": i - 1,
                    "error": abs(fx),
                    "convergence_history": convergence_history,
                    "success": False,
                    "message": "Derivative is zero; Newton-Raphson cannot continue"
                }

            x_new = x - fx / dfx
            error = abs(x_new - x)
            convergence_history.append(error)

            if abs(f(x_new)) < tol or error < tol:
                return {
                    "root": x_new,
                    "iterations": i,
                    "error": error,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "Converged"
                }

            x = x_new

        return {
            "root": x,
            "iterations": max_iter,
            "error": abs(f(x)),
            "convergence_history": convergence_history,
            "success": False,
            "message": "Maximum iterations reached without convergence"
        }

    def secant(self, f: Callable[[float], float], x0: float, x1: float,
               tolerance: Optional[float] = None,
               max_iterations: Optional[int] = None) -> Dict:
        """Find root using secant method.
        Returns dict: root, iterations, error, convergence_history, success, message
        """
        tol = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        convergence_history = []

        f0 = f(x0)
        f1 = f(x1)

        if abs(f0) < tol:
            return {
                "root": x0,
                "iterations": 0,
                "error": abs(f0),
                "convergence_history": convergence_history,
                "success": True,
                "message": "x0 is already a root"
            }

        if abs(f1) < tol:
            return {
                "root": x1,
                "iterations": 0,
                "error": abs(f1),
                "convergence_history": convergence_history,
                "success": True,
                "message": "x1 is already a root"
            }

        for i in range(1, max_iter + 1):
            denom = f1 - f0
            if abs(denom) < 1e-15:
                return {
                    "root": x1,
                    "iterations": i,
                    "error": abs(f1),
                    "convergence_history": convergence_history,
                    "success": False,
                    "message": "Denominator is zero; secant method cannot continue"
                }

            x2 = x1 - f1 * (x1 - x0) / denom
            error = abs(x2 - x1)
            convergence_history.append(error)

            f2 = f(x2)

            if abs(f2) < tol or error < tol:
                return {
                    "root": x2,
                    "iterations": i,
                    "error": error,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "Converged"
                }

            x0, f0 = x1, f1
            x1, f1 = x2, f2

        return {
            "root": x1,
            "iterations": max_iter,
            "error": abs(f1),
            "convergence_history": convergence_history,
            "success": False,
            "message": "Maximum iterations reached without convergence"
        }

    def simultaneous_newton(self, functions: List[Callable],
                            jacobian: Callable,
                            initial_guess: List[float],
                            tolerance: Optional[float] = None,
                            max_iterations: Optional[int] = None) -> Dict:
        """Solve system of nonlinear equations using Newton's method for systems.
        J*dx = -F, x_{new} = x + dx
        Returns dict: solution, iterations, error, convergence_history, success, message
        """
        tol = tolerance if tolerance is not None else self.tolerance
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        convergence_history = []

        x = np.array(initial_guess, dtype=float)

        for i in range(1, max_iter + 1):
            F = np.array([fn(*x) for fn in functions], dtype=float)
            J = np.array(jacobian(*x), dtype=float)

            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-15:
                return {
                    "solution": x.tolist(),
                    "iterations": i - 1,
                    "error": float(np.linalg.norm(F)),
                    "convergence_history": convergence_history,
                    "success": False,
                    "message": "Jacobian is singular; Newton's method cannot continue"
                }

            dx = np.linalg.solve(J, -F)
            x = x + dx
            error = float(np.linalg.norm(dx))
            convergence_history.append(error)

            F_new = np.array([fn(*x) for fn in functions], dtype=float)
            if float(np.linalg.norm(F_new)) < tol or error < tol:
                return {
                    "solution": x.tolist(),
                    "iterations": i,
                    "error": error,
                    "convergence_history": convergence_history,
                    "success": True,
                    "message": "Converged"
                }

        return {
            "solution": x.tolist(),
            "iterations": max_iter,
            "error": float(np.linalg.norm(np.array([fn(*x) for fn in functions], dtype=float))),
            "convergence_history": convergence_history,
            "success": False,
            "message": "Maximum iterations reached without convergence"
        }
