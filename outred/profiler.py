# outred/profiler.py
# Analyzes a dataset before detection to guide smart routing and give users
# a data quality overview.

import polars as pl
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from outred.preprocessing import select_numeric_columns, select_categorical_columns, cast_numeric_strings


@dataclass
class ColumnProfile:
    """Per-column statistics."""
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    cardinality_ratio: float  # unique / total rows (in the SAMPLE, see note below)

    # Numeric-only stats (None for categoricals)
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    iqr: Optional[float] = None

    # Categorical-only stats
    top_values: Optional[List[Dict[str, Any]]] = None  # [{"value": ..., "count": ..., "pct": ...}]


@dataclass
class DataProfile:
    """Full dataset profile used by the smart router."""
    file_path: str
    file_size_mb: float
    total_rows: int            # TRUE row count of the whole file (see note below)
    sample_rows: int           # rows actually used to compute the statistics below
    total_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    id_columns: List[str]          # detected high-cardinality ID-like columns
    constant_columns: List[str]    # zero-variance columns

    null_total: int
    null_pct: float
    data_quality_score: int        # 0–100

    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)

    # Aggregate numeric stats for smart routing
    avg_skewness: float = 0.0
    max_dimensionality: int = 0    # number of usable numeric columns

    # True if avg_skewness etc. were computed from a sample smaller than the
    # full file. Routing rules that key off these stats should treat a
    # near-threshold value with extra caution when this is True, since a
    # sample-derived statistic carries more uncertainty than a population
    # statistic  - see ROUTING NOTE in router/dispatcher.py.
    stats_are_sampled: bool = False

    def to_dict(self) -> dict:
        """Serialise for JSON API responses."""
        d = {
            "file_path": self.file_path,
            "file_size_mb": round(self.file_size_mb, 2),
            "total_rows": self.total_rows,
            "sample_rows": self.sample_rows,
            "stats_are_sampled": self.stats_are_sampled,
            "total_columns": self.total_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "id_columns": self.id_columns,
            "constant_columns": self.constant_columns,
            "null_total": self.null_total,
            "null_pct": round(self.null_pct, 2),
            "data_quality_score": self.data_quality_score,
            "avg_skewness": round(self.avg_skewness, 4),
            "max_dimensionality": self.max_dimensionality,
            "columns": {},
        }
        for name, cp in self.column_profiles.items():
            entry: Dict[str, Any] = {
                "dtype": cp.dtype,
                "null_count": cp.null_count,
                "null_pct": round(cp.null_pct, 2),
                "unique_count": cp.unique_count,
                "cardinality_ratio": round(cp.cardinality_ratio, 4),
            }
            if cp.mean is not None:
                entry.update({
                    "mean": round(cp.mean, 4),
                    "median": round(cp.median, 4) if cp.median is not None else None,
                    "std": round(cp.std, 4) if cp.std is not None else None,
                    "min": cp.min_val,
                    "max": cp.max_val,
                    "skewness": round(cp.skewness, 4) if cp.skewness is not None else None,
                    "kurtosis": round(cp.kurtosis, 4) if cp.kurtosis is not None else None,
                    "iqr": round(cp.iqr, 4) if cp.iqr is not None else None,
                })
            if cp.top_values is not None:
                entry["top_values"] = cp.top_values
            d["columns"][name] = entry
        return d


def profile_dataframe(
    df: pl.DataFrame,
    file_path: str = "<upload>",
    true_row_count: Optional[int] = None,
    id_cardinality_ratio: float = 0.50,
) -> DataProfile:
    """
    Build a DataProfile from a Polars DataFrame.

    For large files the caller should pass a representative SAMPLE (e.g. the
    first ~1,000-10,000 rows) for `df`  - computing per-column stats (mean,
    skewness, etc.) on the full file would be expensive and isn't necessary
    to capture the data's *shape*.

    BUT: total_rows must reflect the TRUE size of the whole file, not the
    sample, because the smart router uses total_rows for decisions like
    "rows > 1,000,000 -> use HBOS". If the caller doesn't pass
    `true_row_count`, this falls back to len(df) (the sample size)  - which
    will silently make any row-count-based routing rule impossible to
    trigger on files larger than the sample. Callers profiling a sample of
    a larger file should ALWAYS pass true_row_count explicitly (see
    outred.ingestion.chunker.count_csv_data_rows for a cheap way to get it).

    Args:
        id_cardinality_ratio: Flag a categorical column as 'ID-like' when
            unique_values / sample_rows exceeds this ratio (default 0.50).
    """
    sample_rows = len(df)
    total_rows = true_row_count if true_row_count is not None else sample_rows
    stats_are_sampled = total_rows != sample_rows

    total_cols = len(df.columns)

    # File size
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, TypeError):
        file_size_mb = 0.0

    # Column classification
    
    df = cast_numeric_strings(df)
    num_cols = select_numeric_columns(df)
    cat_cols = select_categorical_columns(df)

    dt_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Date, pl.Datetime)
    ]

    # Detect ID columns: high cardinality categoricals or sequential integers.
    # Uses configurable id_cardinality_ratio (default 0.50) instead of hardcoded.
    # NOTE: this still operates on the sample, since cardinality ratio in a
    # representative sample is a reasonable proxy for the full file's
    # cardinality shape (unlike total_rows, which is an absolute count that
    # a sample cannot represent).
    #
    # Numeric ID heuristic (three conditions -- ALL must hold):
    #   1. Integer dtype: population counts like POP_SC become Float64 after
    #      cast_numeric_strings, so they safely pass through. Real row IDs
    #      are almost always plain integers in the source CSV.
    #   2. Strictly monotonic with diff == 1: any constant-diff sequence works
    #      for the old check, but a real sequential ID has diff exactly 1
    #      (1, 2, 3 ...). A sorted population column may share a constant diff
    #      by coincidence, but it won't be 1.0 for every pair.
    #   3. Name-based: column name must contain a known ID-like token.
    #      This is the strongest signal and overrides the dtype gate --
    #      a float column literally named "row_id" should still be excluded.
    _ID_NAME_TOKENS = {"id", "idx", "index", "row_id", "row_num", "serial",
                       "record_id", "row_index", "rowid", "pk", "key"}

    id_cols: List[str] = []
    for c in cat_cols:
        if df[c].n_unique() > max(10, sample_rows * id_cardinality_ratio):
            id_cols.append(c)
    for c in num_cols:
        if df[c].n_unique() == sample_rows:
            # Every value in the sample is unique -- further checks required.
            is_integer_dtype = df[c].dtype in (
                pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
            )
            name_lower = c.strip().lower().replace(" ", "_")
            name_looks_like_id = any(
                tok == name_lower
                or name_lower.startswith(tok + "_")
                or name_lower.endswith("_" + tok)
                for tok in _ID_NAME_TOKENS
            )

            if is_integer_dtype or name_looks_like_id:
                vals = df[c].drop_nulls()
                if len(vals) > 1:
                    diffs = vals.diff().drop_nulls()
                    # Must be strictly increasing with constant diff of exactly 1
                    if diffs.n_unique() == 1 and float(diffs.min()) == 1.0:
                        id_cols.append(c)

    # Detect constant columns
    const_cols = [c for c in num_cols if df[c].n_unique() <= 1]

    # Null stats (from sample  - used as an estimate of the full file's null %)
    null_counts = df.null_count()
    null_total = int(null_counts.sum_horizontal()[0])
    total_cells = sample_rows * total_cols
    null_pct = (null_total / total_cells * 100) if total_cells > 0 else 0.0

    # Per-column profiles
    column_profiles: Dict[str, ColumnProfile] = {}
    skewness_values: List[float] = []

    for col in df.columns:
        n_unique = df[col].n_unique()
        n_null = df[col].null_count()
        null_p = (n_null / sample_rows * 100) if sample_rows > 0 else 0.0
        card_ratio = n_unique / sample_rows if sample_rows > 0 else 0.0

        cp = ColumnProfile(
            name=col,
            dtype=str(df[col].dtype),
            null_count=n_null,
            null_pct=null_p,
            unique_count=n_unique,
            cardinality_ratio=card_ratio,
        )

        if col in num_cols:
            series = df[col].drop_nulls().cast(pl.Float64)
            if len(series) > 0:
                arr = series.to_numpy()
                # C1 FIX: sanitize before computing stats  - corrupt CSVs can
                # produce inf/NaN values that propagate into skewness/kurtosis
                # and silently break routing decisions (NaN > 5.0 is False).
                arr = np.where(np.isinf(arr), np.nan, arr)
                arr = np.nan_to_num(arr, nan=0.0)

                cp.mean = float(np.mean(arr))
                cp.median = float(np.median(arr))
                cp.std = float(np.std(arr))
                cp.min_val = float(np.min(arr))
                cp.max_val = float(np.max(arr))
                q1 = float(np.percentile(arr, 25))
                q3 = float(np.percentile(arr, 75))
                cp.iqr = q3 - q1
                # Skewness (Fisher definition)
                if cp.std and cp.std > 0 and len(arr) >= 3:
                    n = len(arr)
                    m3 = float(np.mean((arr - cp.mean) ** 3))
                    cp.skewness = m3 / (cp.std ** 3)
                    # C1 FIX: guard against NaN propagation
                    if not np.isfinite(cp.skewness):
                        cp.skewness = 0.0
                    skewness_values.append(abs(cp.skewness))
                # Kurtosis (excess)
                if cp.std and cp.std > 0 and len(arr) >= 4:
                    m4 = float(np.mean((arr - cp.mean) ** 4))
                    cp.kurtosis = m4 / (cp.std ** 4) - 3.0
                    # C1 FIX: guard against NaN propagation
                    if not np.isfinite(cp.kurtosis):
                        cp.kurtosis = 0.0

        elif col in cat_cols:
            vc = df[col].value_counts().sort("count", descending=True).head(5)
            cp.top_values = [
                {
                    "value": str(vc[col][i]),
                    "count": int(vc["count"][i]),
                    "pct": round(int(vc["count"][i]) / sample_rows * 100, 2),
                }
                for i in range(len(vc))
            ]

        column_profiles[col] = cp

    # Data quality score  (simple heuristic: 100 minus penalties)
    quality = 100
    # Penalty for nulls
    quality -= min(30, int(null_pct))
    # Penalty for constant columns
    if num_cols:
        const_ratio = len(const_cols) / len(num_cols)
        quality -= int(const_ratio * 20)
    # Penalty for very few rows (uses TRUE total_rows, not the sample size —
    # a 500-row file profiled via a 1000-row sample request should be
    # penalized as "few rows"; a 10M-row file should not be, even though
    # both produce a sample of the same size).
    if total_rows < 50:
        quality -= 20
    elif total_rows < 500:
        quality -= 10
    # Penalty for high skewness
    avg_skew = float(np.mean(skewness_values)) if skewness_values else 0.0
    if avg_skew > 5.0:
        quality -= 15
    elif avg_skew > 2.0:
        quality -= 5
    quality = max(0, min(100, quality))

    usable_dims = len([c for c in num_cols if c not in const_cols and c not in id_cols])

    return DataProfile(
        file_path=file_path,
        file_size_mb=file_size_mb,
        total_rows=total_rows,
        sample_rows=sample_rows,
        total_columns=total_cols,
        numeric_columns=num_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        id_columns=id_cols,
        constant_columns=const_cols,
        null_total=null_total,
        null_pct=null_pct,
        data_quality_score=quality,
        column_profiles=column_profiles,
        avg_skewness=avg_skew,
        max_dimensionality=usable_dims,
        stats_are_sampled=stats_are_sampled,
    )