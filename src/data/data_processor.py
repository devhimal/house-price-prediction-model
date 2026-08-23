"""
DataProcessor class for loading, validating, and preprocessing house price data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class DataProcessor:
    """Handles data loading, validation, cleaning, and preprocessing for house price prediction."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._target: str = "price"
        self._features: Optional[List[str]] = None

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame.

        Args:
            filepath: Path to the CSV file.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed as CSV.
        """
        self._df = pd.read_csv(filepath)
        self._features = [c for c in self._df.columns if c != self._target]
        return self._df

    def validate_data(self) -> Dict[str, object]:
        """Validate the loaded DataFrame for common issues.

        Returns:
            A dict describing the validation result.

        Raises:
            RuntimeError: If no data has been loaded.
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        issues: List[str] = []
        if self._df.empty:
            issues.append("DataFrame is empty.")
        if self._df.duplicated().any():
            issues.append(f"{self._df.duplicated().sum()} duplicate rows found.")
        if self._df.isnull().any().any():
            cols = self._df.columns[self._df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {cols}")
        if self._target not in self._df.columns:
            issues.append(f"Target column '{self._target}' not found.")

        return {"valid": len(issues) == 0, "issues": issues}

    def get_summary(self) -> Dict[str, object]:
        """Return a summary dict of the dataset.

        Returns:
            Dict with keys: records, features, target, missing_values, duplicate_rows.
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        return {
            "records": len(self._df),
            "features": len(self._features) if self._features else 0,
            "target": self._target,
            "missing_values": int(self._df.isnull().sum().sum()),
            "duplicate_rows": int(self._df.duplicated().sum()),
        }

    def handle_missing_values(self) -> pd.DataFrame:
        """Fill missing values with the median of each column.

        Returns:
            The DataFrame with missing values filled.
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        for col in self._df.columns:
            if self._df[col].isnull().any():
                self._df[col].fillna(self._df[col].median(), inplace=True)
        return self._df

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the dataset.

        Returns:
            The DataFrame with duplicates removed.
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        self._df.drop_duplicates(inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        return self._df

    def report_outliers(self) -> Dict[str, Dict[str, int]]:
        """Report outliers using the IQR method.

        Returns:
            A dict mapping column name to a dict with 'count', 'lower', and 'upper' bounds.
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        outlier_report: Dict[str, Dict[str, int]] = {}
        for col in self._df.select_dtypes(include=[np.number]).columns:
            q1 = self._df[col].quantile(0.25)
            q3 = self._df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((self._df[col] < lower) | (self._df[col] > upper)).sum())
            outlier_report[col] = {"count": count, "lower": lower, "upper": upper}
        return outlier_report

    def get_feature_names(self) -> List[str]:
        """Return the list of feature column names.

        Returns:
            List of feature column names (excluding target).
        """
        if self._features is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")
        return self._features.copy()

    def get_target_name(self) -> str:
        """Return the name of the target column.

        Returns:
            The target column name (default 'price').
        """
        return self._target

    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to feature matrix.

        Args:
            X: 2D numpy array of features.

        Returns:
            Normalized array with values scaled to [0, 1].
        """
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (X - mins) / ranges

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split arrays into random train and test subsets.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            test_size: Fraction of data to use as test set.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        n = X.shape[0]
        indices = np.random.permutation(n)
        split = int(n * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def preprocess(self, filepath: str) -> Dict[str, object]:
        """Run the full preprocessing pipeline.

        Steps: load -> validate -> fill missing -> remove duplicates -> return arrays.

        Args:
            filepath: Path to the CSV file.

        Returns:
            Dict with keys: X, y, feature_names, summary, validation, outliers.
        """
        self.load_csv(filepath)
        validation = self.validate_data()
        self.handle_missing_values()
        self.remove_duplicates()
        summary = self.get_summary()
        outliers = self.report_outliers()

        X = self._df[self._features].to_numpy().astype(float)
        y = self._df[self._target].to_numpy().astype(float)

        return {
            "X": X,
            "y": y,
            "feature_names": self._features,
            "summary": summary,
            "validation": validation,
            "outliers": outliers,
        }
