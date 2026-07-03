# outred/engines/incremental.py
# Out-of-core outlier detection for datasets that don't fit in memory.
# Uses SGDOneClassSVM with true incremental partial_fit.

import polars as pl
import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.feature_extraction import FeatureHasher
from typing import Optional, List
from outred.preprocessing import select_numeric_columns, select_categorical_columns

class IncrementalOutlierDetector:
    """
    True out-of-core outlier detector using SGDOneClassSVM.
    Handles numeric + categorical columns incrementally via partial_fit.
    Memory footprint stays flat regardless of dataset size.
    """

    def __init__(self, nu: float = 0.05, n_components: int = 100, n_features: int = 20,
                 exclude_columns: Optional[List[str]] = None):
        self.nu = nu
        self.n_components = n_components

        # Incremental scaler for numeric columns
        self.scaler = StandardScaler()

        # Kernel approximation for non-linearity
        self.nystroem = Nystroem(n_components=n_components, random_state=42)

        # The core SVM model
        self.svm = SGDOneClassSVM(nu=nu, random_state=42)

        # Categorical encoder (hashing trick  - no need to see all categories upfront)
        self.hasher = FeatureHasher(n_features=n_features, input_type='dict')

        self.is_fitted = False
        self.nystroem_fitted = False
        self.n_features = n_features

        # FIX: previously this engine ignored config.exclude_columns entirely,
        # meaning detected ID columns (e.g. a sequential 'id' field) leaked
        # into the feature matrix unlike every other detection path. See
        # dispatcher.py's _merge_id_columns_into_exclude docstring for the
        # measured impact of this class of bug.
        self.exclude_columns = exclude_columns or []

    def _prepare_features(self, df: pl.DataFrame) -> np.ndarray:
        """
        Combines numeric (scaled) + categorical (hashed) features into one array.
        Uses the shared column-selection helpers from preprocessing.
        """
        numeric_cols = select_numeric_columns(df, exclude=self.exclude_columns)
        cat_cols = select_categorical_columns(df, exclude=self.exclude_columns)

        parts = []

        # Numeric part
        if numeric_cols:
            X_num = df.select(numeric_cols).to_numpy().astype(np.float64)
            # F1 FIX: sanitize inf BEFORE nan_to_num  - inf values would
            # contaminate the percentile calculation below (np.percentile
            # with inf produces inf bounds, making clip a no-op).
            X_num = np.where(np.isinf(X_num), np.nan, X_num)
            X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip each column to its 1st-99th percentile range (computed
            # per-chunk) as a guard against extreme values that would
            # overflow Nystroem's RBF kernel squared-distance computation.
            if X_num.shape[0] >= 10:
                p1 = np.percentile(X_num, 1, axis=0)
                p99 = np.percentile(X_num, 99, axis=0)
                for j in range(X_num.shape[1]):
                    if p1[j] < p99[j]:
                        X_num[:, j] = np.clip(X_num[:, j], p1[j], p99[j])
            parts.append(X_num)
            
        # Categorical part via hashing trick
        if cat_cols:
            records = [
                {col: str(row[i]) for i, col in enumerate(cat_cols)}
                for row in df.select(cat_cols).iter_rows()
            ]
            X_cat = self.hasher.transform(records).toarray()
            parts.append(X_cat)

        if not parts:
            return np.zeros((len(df), 1))

        return np.hstack(parts)

    def partial_fit(self, df: pl.DataFrame):
        """
        Feed one chunk to update the model incrementally.

        FIX: The scaler now calls partial_fit on EVERY chunk, not just the
        first one.  This ensures the scaler's running mean/variance
        converges to the true population statistics across the whole file.
        """
        X = self._prepare_features(df)

        # Always update the scaler incrementally
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)

        if not self.nystroem_fitted:
            # First chunk  - fit the Nystroem approximation kernel
            self.nystroem.fit(X_scaled)
            self.nystroem_fitted = True

        X_mapped = self.nystroem.transform(X_scaled)
        self.svm.partial_fit(X_mapped)
        self.is_fitted = True

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Score a chunk  - must call partial_fit at least once first."""
        if not self.is_fitted:
            raise RuntimeError("Call partial_fit before predict.")

        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        X_mapped = self.nystroem.transform(X_scaled)

        scores = self.svm.score_samples(X_mapped)
        labels = self.svm.predict(X_mapped)  # 1 = normal, -1 = outlier

        # Normalize scores to 0.0–1.0
        score_min, score_max = scores.min(), scores.max()
        if score_max != score_min:
            normalized = (scores - score_min) / (score_max - score_min)
        else:
            normalized = np.zeros_like(scores)

        return df.with_columns([
            pl.Series("anomaly_score", normalized.round(4)),
            pl.Series("is_outlier", labels == -1)
        ])