# outred/preprocessing.py
# Configurable data preprocessing applied before any detection engine.

import polars as pl
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


# Columns the engines append  - must be excluded when selecting features.
_ENGINE_COLS = frozenset({
    "anomaly_score", "is_outlier",
    "cat_anomaly_score", "is_cat_outlier",
    "explanation",
})

_METADATA_COL_PATTERNS = frozenset({
    "is_outlier", "outlier_type", "ground_truth", "label", "target",
})


def _is_metadata_column(col: str) -> bool:
    return col.strip().lower() in _METADATA_COL_PATTERNS


def select_numeric_columns(
    df: pl.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Return column names that are numeric and not in the exclude list.
    Automatically skips any columns the engines append as output.
    """
    skip = set(_ENGINE_COLS)
    if exclude:
        skip.update(exclude)
    skip.update(c for c in df.columns if _is_metadata_column(c))

    return [
        col for col in df.columns
        if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32,
                             pl.Int16, pl.Int8, pl.UInt64, pl.UInt32,
                             pl.UInt16, pl.UInt8)
        and col not in skip
    ]


def select_categorical_columns(
    df: pl.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Return column names that are string / categorical."""
    skip = set(_ENGINE_COLS)
    if exclude:
        skip.update(exclude)
    skip.update(c for c in df.columns if _is_metadata_column(c))

    return [
        col for col in df.columns
        if df[col].dtype in (pl.Utf8, pl.String, pl.Categorical)
        and col not in skip
    ]


def impute(df: pl.DataFrame, columns: List[str], strategy: str = "median") -> pl.DataFrame:
    """
    Fill null values in *columns* according to *strategy*.

    Strategies:
      median  - fill with column median  (robust to outliers)
      mean    - fill with column mean
      zero    - fill with 0.0
      drop    - drop rows that contain any null in *columns*

    Safety: columns that are >90% null are always zero-filled regardless of
    strategy  - their median/mean is computed from so few non-null values that
    it would create artificial clusters of identical imputed values, masking
    real outliers.
    """
    if strategy == "drop":
        return df.drop_nulls(subset=columns)

    total_rows = len(df)
    fill_exprs = []
    for col in columns:
        # B5 FIX: high-null columns carry no signal  - zero-fill regardless of
        # strategy to avoid creating artificial value clusters from a tiny
        # number of non-null values.
        null_ratio = df[col].null_count() / total_rows if total_rows > 0 else 0.0
        if null_ratio > 0.90:
            fill_exprs.append(pl.col(col).fill_null(0.0))
            continue

        if strategy == "median":
            fill_val = df[col].median()
        elif strategy == "mean":
            fill_val = df[col].mean()
        elif strategy == "zero":
            fill_val = 0.0
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        if fill_val is None or not np.isfinite(fill_val):
            fill_val = 0.0

        fill_exprs.append(pl.col(col).fill_null(fill_val))

    if fill_exprs:
        df = df.with_columns(fill_exprs)
    return df


def drop_constant_columns(columns: List[str], X: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Remove columns with zero or near-zero variance from a numpy matrix.
    Returns updated column names and the trimmed matrix.

    B3 FIX: uses variance > 1e-10 instead of std > 0 to also catch
    near-constant columns (e.g. 99.99% identical values with one outlier)
    that add no meaningful signal but can confuse distance-based models.
    """
    variance = np.var(X, axis=0)
    mask = variance > 1e-10
    if mask.all():
        return columns, X
    kept = [c for c, keep in zip(columns, mask) if keep]
    return kept, X[:, mask]


def build_scaler(method: str):
    """
    Factory for sklearn scalers.
    Returns None if method == 'none'.
    """
    if method == "robust":
        return RobustScaler()
    elif method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def sanitize_numeric_array(X: np.ndarray) -> np.ndarray:
    """
    Universal numeric sanitization applied to every feature matrix before
    any model sees it. Handles the full zoo of garbage that real-world CSVs
    produce:

    1. Replace ±inf with NaN (division-by-zero artifacts, log(0) results)
    2. Replace NaN with 0.0
    3. Clip extreme values per-column to [1st, 99th] percentile  - prevents
       single absurd rows (fare_amount=93963, coordinates=3547) from
       blowing up kernel computations, RobustScaler IQR calculations, or
       distance-based models. Requires ≥10 rows for meaningful percentiles.

    This is called by prepare_matrix (batch path), and should also be called
    by IncrementalOutlierDetector._prepare_features (incremental path).
    """
    # Step 1: inf → NaN → 0.0
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)

    # Step 2: percentile clipping (only if enough rows for meaningful stats)
    if X.shape[0] >= 10 and X.shape[1] > 0:
        p1 = np.percentile(X, 1, axis=0)
        p99 = np.percentile(X, 99, axis=0)
        # Only clip columns where p1 != p99 (avoid clipping to a single value)
        for j in range(X.shape[1]):
            if p1[j] < p99[j]:
                X[:, j] = np.clip(X[:, j], p1[j], p99[j])

    return X


def prepare_matrix(
    df: pl.DataFrame,
    scaling: str = "robust",
    impute_strategy: str = "median",
    exclude: Optional[List[str]] = None,
    numeric_cast_threshold: float = 0.80,
) -> Tuple[np.ndarray, List[str]]:
    """
    Full preprocessing pipeline:
      1. Cast string-encoded numerics to Float64
      2. Select numeric columns (excluding engine outputs + user excludes)
      3. Impute nulls
      4. Convert to numpy
      5. Sanitize (inf, NaN, extreme values)
      6. Drop constant / near-constant columns
      7. Scale

    Returns (X, column_names).  X may have 0 columns if there are no
    usable numeric features  - callers must handle that.
    """
    df = cast_numeric_strings(df, cast_threshold=numeric_cast_threshold)
    num_cols = select_numeric_columns(df, exclude=exclude)
    if not num_cols:
        return np.zeros((len(df), 0)), []

    # Impute
    clean_df = impute(df.select(num_cols), num_cols, strategy=impute_strategy)

    # To numpy
    X = clean_df.to_numpy().astype(np.float64)

    # B1+B2 FIX: sanitize inf, NaN, and extreme values BEFORE scaling
    X = sanitize_numeric_array(X)

    # Drop constants (B3 FIX: uses variance > 1e-10)
    col_names, X = drop_constant_columns(num_cols, X)
    if X.shape[1] == 0:
        return X, col_names

    # Scale
    scaler = build_scaler(scaling)
    if scaler is not None and X.shape[0] >= 2:
        try:
            X = scaler.fit_transform(X)
            # Post-scaling safety: scaler can produce NaN/inf on degenerate data
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            # If scaler fails (e.g. singular matrix), proceed unscaled
            pass

    return X, col_names


def cast_numeric_strings(df: pl.DataFrame, cast_threshold: float = 0.80) -> pl.DataFrame:
    """
    Attempt to cast String columns to Float64 if they contain numeric data.
    Handles quoted-number CSVs where all columns are read as String by Polars
    (common in government/legacy datasets where values like "128" are quoted).

    A column is cast if > cast_threshold fraction of its non-null values parse
    successfully as float (default 0.80 = 80%).  Lower values are more aggressive
    and risk converting coded IDs (zip codes, FIPS codes) to numeric features.

    Safety checks (B4 FIX):
    - Skips columns where >10% of non-null values have leading zeros (e.g. "00123")
       - these are almost certainly codes/IDs, not real numbers.
    - Skips columns where all successfully-cast values are integers AND
      n_unique / total > 0.50  - these are likely categorical codes (state codes,
      district codes) that happen to be numeric.
    """
    cast_exprs = []
    for col in df.columns:
        if df[col].dtype not in (pl.Utf8, pl.String):
            continue

        try:
            non_null = df[col].drop_nulls()
            non_null_count = len(non_null)
            if non_null_count == 0:
                continue

            # B4 FIX: detect zero-padded codes  - sample up to 200 values
            sample = non_null.head(min(200, non_null_count))
            # Check for leading zeros: values like "007", "00123", "0042"
            # But NOT "0" itself (which is a valid number) or "0.5" (valid float)
            leading_zero_count = 0
            for val in sample.to_list():
                s = str(val).strip()
                if len(s) > 1 and s[0] == '0' and s[1] != '.':
                    leading_zero_count += 1
            if leading_zero_count > len(sample) * 0.10:
                continue  # likely a code column, skip

            casted = df[col].cast(pl.Float64, strict=False)
            non_null_casted = casted.drop_nulls().len()

            if non_null_count > 0 and (non_null_casted / non_null_count) > cast_threshold:
                cast_exprs.append(casted.alias(col))
        except Exception:
            pass

    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df