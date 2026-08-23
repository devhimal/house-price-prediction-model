import numpy as np


class NumericalDifferentiation:
    """Numerical differentiation methods using finite differences.

    Provides forward, backward, and central difference approximations
    for first and second derivatives, along with comparison and
    sensitivity analysis utilities.
    """

    def __init__(self, h: float = 1e-5):
        """Initialize with default step size.

        Args:
            h: Step size for finite difference approximations.
        """
        self.h = h

    def forward_difference(self, f, x: float, h: float = None) -> float:
        """Compute first derivative using forward difference.

        f'(x) approx (f(x+h) - f(x)) / h

        Accuracy: O(h)

        Args:
            f: Function to differentiate.
            x: Point at which to evaluate the derivative.
            h: Step size. Uses instance default if None.

        Returns:
            Approximate value of f'(x).
        """
        if h is None:
            h = self.h
        return (f(x + h) - f(x)) / h

    def backward_difference(self, f, x: float, h: float = None) -> float:
        """Compute first derivative using backward difference.

        f'(x) approx (f(x) - f(x-h)) / h

        Accuracy: O(h)

        Args:
            f: Function to differentiate.
            x: Point at which to evaluate the derivative.
            h: Step size. Uses instance default if None.

        Returns:
            Approximate value of f'(x).
        """
        if h is None:
            h = self.h
        return (f(x) - f(x - h)) / h

    def central_difference(self, f, x: float, h: float = None) -> float:
        """Compute first derivative using central difference.

        f'(x) approx (f(x+h) - f(x-h)) / (2h)

        Accuracy: O(h^2) - more accurate than forward/backward.

        Args:
            f: Function to differentiate.
            x: Point at which to evaluate the derivative.
            h: Step size. Uses instance default if None.

        Returns:
            Approximate value of f'(x).
        """
        if h is None:
            h = self.h
        return (f(x + h) - f(x - h)) / (2 * h)

    def second_derivative(self, f, x: float, h: float = None) -> float:
        """Compute second derivative using central difference.

        f''(x) approx (f(x+h) - 2f(x) + f(x-h)) / h^2

        Accuracy: O(h^2)

        Args:
            f: Function to differentiate.
            x: Point at which to evaluate the second derivative.
            h: Step size. Uses instance default if None.

        Returns:
            Approximate value of f''(x).
        """
        if h is None:
            h = self.h
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

    def compare_methods(self, f, df_exact, x: float) -> dict:
        """Compare all three first-derivative methods against the exact derivative.

        Args:
            f: Function to differentiate.
            df_exact: Exact derivative function.
            x: Point at which to evaluate.

        Returns:
            Dictionary with keys:
                forward: result from forward_difference
                backward: result from backward_difference
                central: result from central_difference
                exact: value from df_exact(x)
                error_forward: absolute error of forward method
                error_backward: absolute error of backward method
                error_central: absolute error of central method
        """
        exact = df_exact(x)
        fwd = self.forward_difference(f, x)
        bwd = self.backward_difference(f, x)
        cen = self.central_difference(f, x)

        return {
            "forward": fwd,
            "backward": bwd,
            "central": cen,
            "exact": exact,
            "error_forward": abs(fwd - exact),
            "error_backward": abs(bwd - exact),
            "error_central": abs(cen - exact),
        }

    def price_sensitivity(self, price_function, feature_value: float,
                          feature_name: str, h: float = None) -> dict:
        """Analyze sensitivity of house price to a single feature.

        Computes dPrice/dFeature at the given feature_value using
        the central difference method for best accuracy.

        Args:
            price_function: Function that takes a feature value and returns price.
            feature_value: The value of the feature at which to evaluate sensitivity.
            feature_name: Human-readable name of the feature being analyzed.
            h: Step size. Uses instance default if None.

        Returns:
            Dictionary with keys:
                feature: the feature_name
                feature_value: the input feature_value
                sensitivity: approximate dPrice/dFeature
                interpretation: string describing the sensitivity
        """
        sensitivity = self.central_difference(price_function, feature_value, h)

        if sensitivity > 0:
            interpretation = (
                f"Increasing {feature_name} by 1 unit increases price by "
                f"approximately ${abs(sensitivity):,.2f}"
            )
        elif sensitivity < 0:
            interpretation = (
                f"Increasing {feature_name} by 1 unit decreases price by "
                f"approximately ${abs(sensitivity):,.2f}"
            )
        else:
            interpretation = (
                f"Price is insensitive to changes in {feature_name} "
                f"at value {feature_value}"
            )

        return {
            "feature": feature_name,
            "feature_value": feature_value,
            "sensitivity": sensitivity,
            "interpretation": interpretation,
        }
