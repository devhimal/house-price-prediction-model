import numpy as np

class ErrorAnalyzer:
    def __init__(self):
        pass
    
    def absolute_error(self, true_val: float, approx_val: float) -> float:
        """|true - approx|"""
        return abs(true_val - approx_val)
    
    def relative_error(self, true_val: float, approx_val: float) -> float:
        """|true - approx| / |true|, handles division by zero"""
        if true_val == 0:
            return float('inf') if approx_val != 0 else 0.0
        return abs(true_val - approx_val) / abs(true_val)
    
    def percentage_error(self, true_val: float, approx_val: float) -> float:
        """relative_error * 100"""
        return self.relative_error(true_val, approx_val) * 100
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAE = mean(|y_true - y_pred|)"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE = sqrt(mean((y_true - y_pred)^2))"""
        return float(np.sqrt(np.mean((y_true - y_pred)**2)))
    
    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R^2 = 1 - SS_res / SS_tot"""
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0:
            return float('inf') if ss_res != 0 else 0.0
        return float(1 - ss_res / ss_tot)
    
    def mean_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MPE = mean(|y_true - y_pred| / |y_true|) * 100"""
        abs_errors = np.abs(y_true - y_pred)
        abs_true = np.abs(y_true)
        # Handle division by zero
        mask = abs_true != 0
        if not np.any(mask):
            return 0.0
        percentages = abs_errors[mask] / abs_true[mask] * 100
        return float(np.mean(percentages))
    
    def prediction_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Comprehensive error analysis.
        Returns dict with all metrics, individual errors, and summary.
        """
        individual_errors = y_true - y_pred
        abs_individual_errors = np.abs(individual_errors)
        
        # Basic metrics
        mae = self.mean_absolute_error(y_true, y_pred)
        rmse = self.root_mean_squared_error(y_true, y_pred)
        r2 = self.r_squared(y_true, y_pred)
        mpe = self.mean_percentage_error(y_true, y_pred)
        
        # Summary statistics
        summary = {
            'mae': mae,
            'rmse': rmse,
            'r_squared': r2,
            'mean_percentage_error': mpe,
            'mean_error': float(np.mean(individual_errors)),
            'std_error': float(np.std(individual_errors)),
            'max_abs_error': float(np.max(abs_individual_errors)),
            'min_abs_error': float(np.min(abs_individual_errors)),
            'median_abs_error': float(np.median(abs_individual_errors)),
            'n_samples': len(y_true)
        }
        
        return {
            'individual_errors': individual_errors,
            'abs_individual_errors': abs_individual_errors,
            'summary': summary
        }
    
    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Analyze residuals: mean, std, min, max, normality check (basic)."""
        residuals = y_true - y_pred
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'median': float(np.median(residuals))
        }
        
        # Basic normality check: skewness and kurtosis (approximate)
        n = len(residuals)
        if n > 2:
            mean = np.mean(residuals)
            std = np.std(residuals)
            if std == 0:
                skewness = 0.0
                kurtosis = 0.0
            else:
                skewness = float(np.mean(((residuals - mean) / std)**3))
                kurtosis = float(np.mean(((residuals - mean) / std)**4) - 3)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        return {
            'stats': stats,
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'is_symmetric': abs(skewness) < 0.5,
            'is_mesokurtic': abs(kurtosis) < 1.0
        }
    
    def error_propagation_addition(self, errors: list) -> float:
        """For z = x + y: sigma_z = sqrt(sigma_x^2 + sigma_y^2)"""
        if not errors:
            return 0.0
        sum_sq = sum(e**2 for e in errors)
        return float(np.sqrt(sum_sq))
    
    def error_propagation_multiplication(self, values: list, errors: list) -> float:
        """For z = x*y: sigma_z/|z| = sqrt((sigma_x/x)^2 + (sigma_y/y)^2)"""
        if len(values) != len(errors):
            raise ValueError("values and errors must have same length")
        if not values:
            return 0.0
        
        # Calculate product
        z = 1.0
        for v in values:
            z *= v
        
        # Calculate relative errors squared
        sum_rel_sq = 0.0
        for v, e in zip(values, errors):
            if v == 0:
                raise ValueError("Value cannot be zero for multiplication propagation")
            sum_rel_sq += (e / v)**2
        
        # sigma_z = |z| * sqrt(sum of relative errors squared)
        sigma_z = abs(z) * np.sqrt(sum_rel_sq)
        return float(sigma_z)