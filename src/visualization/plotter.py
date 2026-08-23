import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')

    def dataset_scatter(self, X, y, feature_names, target_name, feature_index=0):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, feature_index], y, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax.set_xlabel(feature_names[feature_index])
        ax.set_ylabel(target_name)
        ax.set_title(f'{feature_names[feature_index]} vs {target_name}')
        fig.tight_layout()
        return fig

    def actual_vs_predicted(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', linewidths=0.5)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        fig.tight_layout()
        return fig

    def residual_plot(self, y_true, y_pred):
        residuals = np.array(y_true) - np.array(y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        fig.tight_layout()
        return fig

    def regression_curve(self, X, y, coefficients, feature_index=0, feature_name="Feature"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, feature_index], y, alpha=0.7, edgecolors='k', linewidths=0.5, label='Data Points')

        x_range = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), 300)
        coefficients = np.array(coefficients)
        if len(coefficients) == 1:
            y_line = coefficients[0] * x_range
        else:
            y_line = np.polyval(coefficients[::-1], x_range)

        ax.plot(x_range, y_line, 'r-', linewidth=2, label='Regression Curve')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Target')
        ax.set_title(f'Regression Curve: {feature_name}')
        ax.legend()
        fig.tight_layout()
        return fig

    def root_finding_convergence(self, history, method_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        iterations = np.arange(1, len(history) + 1)
        ax.semilogy(iterations, history, 'b-o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error (log scale)')
        ax.set_title(f'Root Finding Convergence: {method_name}')
        fig.tight_layout()
        return fig

    def interpolation_comparison(self, x_data, y_data, x_fine, results_dict):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_data, y_data, color='black', zorder=5, label='Original Data', edgecolors='k', linewidths=0.5)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        for (method_name, y_values), color in zip(results_dict.items(), colors):
            ax.plot(x_fine, y_values, label=method_name, linewidth=2, color=color)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Interpolation Comparison')
        ax.legend()
        fig.tight_layout()
        return fig

    def eigenvalue_convergence(self, convergence_history, method_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        iterations = np.arange(1, len(convergence_history) + 1)
        ax.semilogy(iterations, convergence_history, 'b-o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Change in Eigenvalue (log scale)')
        ax.set_title(f'Eigenvalue Convergence: {method_name}')
        fig.tight_layout()
        return fig

    def integration_comparison(self, results_dict):
        fig, ax = plt.subplots(figsize=(8, 6))
        methods = list(results_dict.keys())
        values = list(results_dict.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        bars = ax.bar(methods, values, color=colors, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.6f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Method')
        ax.set_ylabel('Result')
        ax.set_title('Numerical Integration Comparison')
        fig.tight_layout()
        return fig

    def ode_solution(self, t_values, solutions_dict):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(solutions_dict)))
        for (method_name, y_values), color in zip(solutions_dict.items(), colors):
            ax.plot(t_values, y_values, label=method_name, linewidth=2, color=color)
        ax.set_xlabel('t')
        ax.set_ylabel('y(t)')
        ax.set_title('ODE Solution Comparison')
        ax.legend()
        fig.tight_layout()
        return fig

    def bvp_solution(self, x_values, y_values, method_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_values, y_values, 'b-o', markersize=4, linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.set_title(f'BVP Solution: {method_name}')
        fig.tight_layout()
        return fig

    def error_distribution(self, errors):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        fig.tight_layout()
        return fig

    def feature_importance(self, feature_names, coefficients):
        fig, ax = plt.subplots(figsize=(8, 6))
        coefficients = np.array(coefficients)
        sorted_indices = np.argsort(np.abs(coefficients))
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_coeffs = coefficients[sorted_indices]
        colors = ['green' if c > 0 else 'red' for c in sorted_coeffs]
        ax.barh(sorted_names, sorted_coeffs, color=colors, edgecolor='black')
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Importance (Regression Coefficients)')
        fig.tight_layout()
        return fig

    def correlation_heatmap(self, data, feature_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = np.corrcoef(data.T)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        n = len(feature_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Feature Correlation Heatmap')
        fig.tight_layout()
        return fig
