# outred/engines/categorical.py
# Rare-category outlier detector using frequency analysis.

import polars as pl
from typing import Dict, Optional
from outred.preprocessing import select_categorical_columns

# E2 FIX: safety cap on categorical columns to prevent OOM on wide datasets.
# No real dataset needs 500 string columns individually frequency-analyzed.
_MAX_CAT_COLS = 50


def detect_categorical_outliers(
    df: pl.DataFrame,
    threshold: float = 0.01,
    max_cardinality_ratio: float = 0.10,
    min_cardinality: int = 10,
    relative_multiplier: float = 0.3,
    global_freq_maps: Optional[Dict[str, Dict[str, float]]] = None,
) -> pl.DataFrame:
    """
    Flags rare categories as outliers using frequency analysis.

    A category is flagged as an outlier using an ADAPTIVE threshold:
    instead of a flat absolute threshold (which over-flags medium-cardinality
    columns), we compare each value's frequency against the *expected*
    frequency for that column's cardinality.

    Specifically, for a column with N unique values, the expected frequency
    is 1/N (uniform assumption). A value is flagged rare only if:
        actual_freq < expected_freq * relative_multiplier
    The absolute threshold is used as a floor so that ultra-rare values in
    low-cardinality columns (e.g. a typo in a 3-value column) are still caught.

    Args:
        threshold:  Absolute frequency floor below which a category is flagged
                    rare (default 0.01 = 1%).  Applied as a minimum alongside
                    the adaptive threshold.
        max_cardinality_ratio:  Skip columns with n_unique/total_rows > this
                                ratio (default 0.10 = 10%).
        min_cardinality:  Floor on the unique-count check (default 10).
        relative_multiplier:  A value is rare if its frequency is below this
                              fraction of the expected frequency.  Default 0.3
                              means "flagged if < 30% of expected frequency".
        global_freq_maps:  Optional pre-computed global frequency maps.
                           Dict mapping column_name -> {value: frequency}.
                           When provided, per-chunk value_counts are skipped
                           and these global frequencies are used instead,
                           solving the per-chunk frequency distortion bug.

    Adds two columns:
      cat_anomaly_score -- how rare the category is (0.0 = common, 1.0 = rarest)
      is_cat_outlier    -- True if frequency is below adaptive threshold

    Safety checks:
      E1: Null values are filled with a sentinel before frequency analysis so
          they don't get treated as an ultra-rare category that flags 50% of rows.
      E2: Caps the number of categorical columns processed at _MAX_CAT_COLS to
          prevent OOM on wide datasets. Keeps lowest-cardinality columns first
          (most likely to be real categories).
    """
    total_rows = len(df)
    if total_rows == 0:
        return df.with_columns([
            pl.lit(0.0).alias("cat_anomaly_score"),
            pl.lit(False).alias("is_cat_outlier"),
        ])

    cat_cols = select_categorical_columns(df)

    # Skip high-cardinality columns -- likely ID columns, not real categories.
    cat_cols = [
        col for col in cat_cols
        if df[col].n_unique() <= max(min_cardinality, int(total_rows * max_cardinality_ratio))
    ]

    # E2 FIX: cap the number of categorical columns to prevent OOM.
    if len(cat_cols) > _MAX_CAT_COLS:
        cat_cols.sort(key=lambda c: df[c].n_unique())
        cat_cols = cat_cols[:_MAX_CAT_COLS]

    if not cat_cols:
        return df.with_columns([
            pl.lit(0.0).alias("cat_anomaly_score"),
            pl.lit(False).alias("is_cat_outlier"),
        ])

    # E1 FIX: fill nulls with a sentinel so they don't appear as an ultra-rare
    # unique value.
    _SENTINEL = "__MISSING__"
    null_cols = []
    for col in cat_cols:
        if df[col].null_count() > 0:
            null_cols.append(col)
            df = df.with_columns(pl.col(col).fill_null(pl.lit(_SENTINEL)))

    rarity_exprs = []
    flag_exprs = []

    for col in cat_cols:
        if global_freq_maps is not None and col in global_freq_maps:
            # --- Use pre-computed global frequencies ---
            col_freqs = global_freq_maps[col]
            n_unique_global = len(col_freqs)

            # Map each row's value to its global frequency
            freq_series = df[col].map_elements(
                lambda v: col_freqs.get(v, 0.0),
                return_dtype=pl.Float64,
            )
            freq_col_name = f"__freq_{col}"
            df = df.with_columns(freq_series.alias(freq_col_name))
        else:
            # --- Compute per-chunk frequencies (single-pass / fallback) ---
            n_unique_global = df[col].n_unique()
            freq_map = (
                df[col]
                .value_counts()
                .with_columns(
                    (pl.col("count") / total_rows).alias(f"__freq_{col}")
                )
                .select([col, f"__freq_{col}"])
            )
            df = df.join(freq_map, on=col, how="left")
            freq_col_name = f"__freq_{col}"

        freq_expr = pl.col(freq_col_name)
        max_freq = df[freq_col_name].max()
        min_freq = df[freq_col_name].min()

        # Rarity score: 0.0 = most common, 1.0 = rarest
        if max_freq is not None and min_freq is not None and max_freq != min_freq:
            rarity_exprs.append(
                ((pl.lit(max_freq) - freq_expr) / (max_freq - min_freq))
                .alias(f"__rarity_{col}")
            )
        else:
            rarity_exprs.append(pl.lit(0.0).alias(f"__rarity_{col}"))

        # --- Adaptive threshold ---
        # expected_freq = 1 / n_unique (uniform assumption)
        # adaptive = expected_freq * relative_multiplier
        # effective_threshold = min(threshold, adaptive)
        # A value is rare if actual_freq < effective_threshold
        if n_unique_global > 0:
            expected_freq = 1.0 / n_unique_global
            adaptive_threshold = expected_freq * relative_multiplier
            effective_threshold = min(threshold, adaptive_threshold)
        else:
            effective_threshold = threshold

        # E1 FIX: don't flag the sentinel as rare -- null values are "missing",
        # not "anomalous". Only flag non-sentinel values below threshold.
        if col in null_cols:
            flag_exprs.append(
                ((freq_expr < effective_threshold) & (pl.col(col) != _SENTINEL))
                .alias(f"__flag_{col}")
            )
        else:
            flag_exprs.append(
                (freq_expr < effective_threshold).alias(f"__flag_{col}")
            )

    df = df.with_columns(rarity_exprs + flag_exprs)

    rarity_cols = [f"__rarity_{c}" for c in cat_cols]
    flag_cols = [f"__flag_{c}" for c in cat_cols]

    df = df.with_columns([
        pl.max_horizontal(rarity_cols).round(4).alias("cat_anomaly_score"),
        pl.any_horizontal(flag_cols).alias("is_cat_outlier"),
    ])

    # Drop intermediate helper columns
    helper_cols = [f"__freq_{c}" for c in cat_cols] + rarity_cols + flag_cols
    df = df.drop(helper_cols)

    # E1 FIX: restore nulls -- replace sentinel back with null so the output
    # DataFrame preserves the original null semantics.
    for col in null_cols:
        df = df.with_columns(
            pl.when(pl.col(col) == _SENTINEL)
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col)
        )

    return df