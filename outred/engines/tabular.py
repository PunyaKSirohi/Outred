# outred/engines/tabular.py
# Multi-algorithm outlier detection engine for in-memory (batch) datasets.
# Wraps PyOD models behind a unified interface with score normalisation.
#
# Architecture (Option B): the model is fitted ONCE on a representative
# global sample drawn uniformly from the file via reservoir sampling.
# Each chunk is then scored against that single fitted model -- no re-fitting.
# This produces globally consistent scores and a stable decision boundary
# instead of the per-chunk models that previously made scores incomparable.

import polars as pl
import numpy as np
from typing import Optional, Tuple, List

# PyOD models are imported lazily inside build_model() and fit_global_ensemble()
# to avoid loading all model classes at server startup (~30-50 MB saved at idle).

from outred.config import OutredConfig
from outred.preprocessing import prepare_matrix


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def _normalise_scores(
    scores: np.ndarray,
    global_lo: Optional[float] = None,
    global_hi: Optional[float] = None,
) -> np.ndarray:
    """
    Map raw decision_function scores to 0.0 - 1.0.

    If global_lo/global_hi are provided (from the model's training scores),
    those bounds are used for ALL chunks so that anomaly_score values are
    directly comparable across chunks.  Scores outside the training range
    are clipped to [0, 1].

    Falls back to chunk-local min/max only when no global bounds are given
    (legacy single-chunk use).
    """
    lo = global_lo if global_lo is not None else float(scores.min())
    hi = global_hi if global_hi is not None else float(scores.max())
    if hi != lo:
        normed = (scores - lo) / (hi - lo)
        return np.clip(normed, 0.0, 1.0).round(4)
    return np.zeros_like(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Global model builder
# ---------------------------------------------------------------------------

def build_model(algo: str, config: OutredConfig, n_rows: int):
    """
    Construct an unfitted PyOD model for the given algorithm.
    n_rows is used to bound n_neighbors / n_clusters safely.
    PyOD models are imported lazily here so they are only loaded into memory
    when a request actually triggers detection.
    """
    if algo == "iforest":
        from pyod.models.iforest import IForest
        return IForest(contamination=config.contamination, random_state=42)
    if algo == "hbos":
        from pyod.models.hbos import HBOS
        return HBOS(contamination=config.contamination)
    if algo == "lof":
        from pyod.models.lof import LOF
        n_neighbors = min(20, max(2, n_rows - 1))
        return LOF(contamination=config.contamination, n_neighbors=n_neighbors)
    if algo == "cblof":
        from pyod.models.cblof import CBLOF
        n_clusters = min(8, max(2, n_rows // 50))
        return CBLOF(contamination=config.contamination, n_clusters=n_clusters,
                     random_state=42)
    if algo == "ocsvm":
        from pyod.models.ocsvm import OCSVM
        return OCSVM(contamination=config.contamination)
    raise ValueError(f"Unknown algorithm: {algo}. Valid: iforest, hbos, lof, cblof, ocsvm")


# ---------------------------------------------------------------------------
# Option B: fit once on global sample, score each chunk
# ---------------------------------------------------------------------------

def fit_global_model(
    sample_df: pl.DataFrame,
    algo: str,
    config: OutredConfig,
) -> Tuple[Optional[object], float, float, float]:
    """
    Fit a PyOD model on the reservoir-sampled global sample.

    Returns (fitted_model, global_threshold, global_score_lo, global_score_hi).

    - global_threshold: the contamination-based decision boundary derived from
      the training scores.  Any chunk row whose decision_function score is
      >= this value is flagged as an outlier.
    - global_score_lo/hi: the min/max of the training scores, used to normalise
      anomaly_score to [0, 1] consistently across all chunks.

    Returns (None, 0, 0, 0) if the sample is empty or model fitting fails.
    """
    X, col_names = prepare_matrix(
        sample_df,
        scaling=config.scaling,
        impute_strategy=config.impute,
        exclude=config.exclude_columns,
        numeric_cast_threshold=config.numeric_cast_threshold,
    )

    if X.shape[1] == 0 or X.shape[0] < 2:
        return None, 0.0, 0.0, 0.0

    model = build_model(algo, config, X.shape[0])

    try:
        model.fit(X)
        scores = model.decision_function(X)
    except Exception as e:
        print(f"  Warning: Global model fit failed for {algo}: {e}")
        return None, 0.0, 0.0, 0.0

    global_lo = float(scores.min())
    global_hi = float(scores.max())
    # Threshold at the (1 - contamination) percentile of training scores.
    # Any score >= this on a new chunk row will be flagged as an outlier.
    threshold = float(np.percentile(scores, 100.0 * (1.0 - config.contamination)))

    return model, threshold, global_lo, global_hi


def score_chunk(
    df: pl.DataFrame,
    model,
    global_threshold: float,
    global_lo: float,
    global_hi: float,
    config: OutredConfig,
) -> pl.DataFrame:
    """
    Score one chunk against the pre-fitted global model.
    NO re-fitting occurs -- only decision_function() is called.

    anomaly_score is normalised using the training set's lo/hi bounds so
    values are directly comparable across chunks.
    """
    default = df.with_columns([
        pl.lit(0.0).alias("anomaly_score"),
        pl.lit(False).alias("is_outlier"),
    ])

    if model is None:
        return default

    X, col_names = prepare_matrix(
        df,
        scaling=config.scaling,
        impute_strategy=config.impute,
        exclude=config.exclude_columns,
        numeric_cast_threshold=config.numeric_cast_threshold,
    )

    if X.shape[1] == 0 or X.shape[0] < 1:
        return default

    try:
        scores = model.decision_function(X)
    except Exception as e:
        print(f"  Warning: Scoring failed on chunk: {e}. Returning zero scores.")
        return default

    normalized = _normalise_scores(scores, global_lo, global_hi)
    labels = scores >= global_threshold

    return df.with_columns([
        pl.Series("anomaly_score", normalized),
        pl.Series("is_outlier", labels),
    ])


# ---------------------------------------------------------------------------
# Option B: Ensemble (IForest + HBOS + LOF)
# ---------------------------------------------------------------------------

def fit_global_ensemble(
    sample_df: pl.DataFrame,
    config: OutredConfig,
) -> Tuple[Optional[List], float, float, float]:
    """
    Fit IForest + HBOS + LOF on the global sample.
    Normalises each model's training scores to [0,1], averages them,
    and derives a single threshold from the combined scores.

    Returns (fitted_models_list, threshold, avg_lo, avg_hi).
    fitted_models_list is a list of (name, model, train_lo, train_hi) tuples.
    """
    X, col_names = prepare_matrix(
        sample_df,
        scaling=config.scaling,
        impute_strategy=config.impute,
        exclude=config.exclude_columns,
        numeric_cast_threshold=config.numeric_cast_threshold,
    )

    if X.shape[1] == 0 or X.shape[0] < 2:
        return None, 0.0, 0.0, 0.0

    from pyod.models.iforest import IForest
    from pyod.models.hbos import HBOS
    from pyod.models.lof import LOF
    n_neighbors = min(20, max(2, X.shape[0] - 1))
    candidates = [
        ("IForest", IForest(contamination=config.contamination, random_state=42)),
        ("HBOS",    HBOS(contamination=config.contamination)),
        ("LOF",     LOF(contamination=config.contamination, n_neighbors=n_neighbors)),
    ]

    fitted = []
    all_normed = []
    for name, m in candidates:
        try:
            m.fit(X)
            raw = m.decision_function(X)
            lo, hi = float(raw.min()), float(raw.max())
            normed = (raw - lo) / (hi - lo) if hi != lo else np.zeros_like(raw)
            all_normed.append(normed)
            fitted.append((name, m, lo, hi))
        except Exception as e:
            print(f"  Warning: Ensemble member {name} failed on sample: {e} -- skipping.")

    if not fitted:
        return None, 0.0, 0.0, 0.0

    avg = np.mean(all_normed, axis=0)
    global_lo = float(avg.min())
    global_hi = float(avg.max())
    threshold = float(np.percentile(avg, 100.0 * (1.0 - config.contamination)))

    return fitted, threshold, global_lo, global_hi


def score_chunk_ensemble(
    df: pl.DataFrame,
    fitted_models: list,
    global_threshold: float,
    global_lo: float,
    global_hi: float,
    config: OutredConfig,
) -> pl.DataFrame:
    """Score a chunk against the pre-fitted ensemble using the training-derived lo/hi."""
    default = df.with_columns([
        pl.lit(0.0).alias("anomaly_score"),
        pl.lit(False).alias("is_outlier"),
    ])

    if not fitted_models:
        return default

    X, col_names = prepare_matrix(
        df,
        scaling=config.scaling,
        impute_strategy=config.impute,
        exclude=config.exclude_columns,
        numeric_cast_threshold=config.numeric_cast_threshold,
    )

    if X.shape[1] == 0 or X.shape[0] < 1:
        return default

    all_normed = []
    for name, m, train_lo, train_hi in fitted_models:
        try:
            raw = m.decision_function(X)
            if train_hi != train_lo:
                normed = (raw - train_lo) / (train_hi - train_lo)
            else:
                normed = np.zeros_like(raw)
            all_normed.append(np.clip(normed, 0.0, 1.0))
        except Exception as e:
            print(f"  Warning: Ensemble member {name} failed on chunk: {e} -- skipping.")

    if not all_normed:
        return default

    avg = np.mean(all_normed, axis=0).round(4)
    labels = avg >= global_threshold

    return df.with_columns([
        pl.Series("anomaly_score", avg),
        pl.Series("is_outlier", labels),
    ])


# ---------------------------------------------------------------------------
# Legacy per-chunk runners (kept for backward compat / direct model use)
# ---------------------------------------------------------------------------

def run_pyod_model(df: pl.DataFrame, model, config: OutredConfig) -> pl.DataFrame:
    """
    Legacy: fit + score on a single DataFrame.
    Used internally by tests and the legacy run_* helpers below.
    Scores are normalised locally (per-chunk) since there is no global context.
    """
    X, col_names = prepare_matrix(
        df, scaling=config.scaling, impute_strategy=config.impute,
        exclude=config.exclude_columns,
        numeric_cast_threshold=config.numeric_cast_threshold,
    )
    default = df.with_columns([
        pl.lit(0.0).alias("anomaly_score"),
        pl.lit(False).alias("is_outlier"),
    ])
    if X.shape[1] == 0 or X.shape[0] < 2:
        return default
    try:
        model.fit(X)
        scores = model.decision_function(X)
        labels = model.predict(X)
    except Exception as e:
        print(f"  Warning: {type(model).__name__} failed: {e}")
        return default
    return df.with_columns([
        pl.Series("anomaly_score", _normalise_scores(scores)),
        pl.Series("is_outlier", labels == 1),
    ])


def run_iforest(df: pl.DataFrame, config: OutredConfig) -> pl.DataFrame:
    from pyod.models.iforest import IForest
    return run_pyod_model(df, IForest(contamination=config.contamination, random_state=42), config)

def run_hbos(df: pl.DataFrame, config: OutredConfig) -> pl.DataFrame:
    from pyod.models.hbos import HBOS
    return run_pyod_model(df, HBOS(contamination=config.contamination), config)

def run_lof(df: pl.DataFrame, config: OutredConfig) -> pl.DataFrame:
    from pyod.models.lof import LOF
    n = min(20, max(2, df.height - 1))
    return run_pyod_model(df, LOF(contamination=config.contamination, n_neighbors=n), config)

def run_cblof(df: pl.DataFrame, config: OutredConfig) -> pl.DataFrame:
    from pyod.models.cblof import CBLOF
    n = min(8, max(2, df.height // 50))
    return run_pyod_model(df, CBLOF(contamination=config.contamination, n_clusters=n, random_state=42), config)

def run_ocsvm(df: pl.DataFrame, config: OutredConfig) -> pl.DataFrame:
    from pyod.models.ocsvm import OCSVM
    return run_pyod_model(df, OCSVM(contamination=config.contamination), config)


# Legacy name → runner map (kept for backward compat only)
_RUNNERS = {
    "iforest": run_iforest,
    "hbos": run_hbos,
    "lof": run_lof,
    "cblof": run_cblof,
    "ocsvm": run_ocsvm,
}


def get_runner(name: str):
    """Return the legacy per-chunk runner. Not used by the main pipeline (Option B)."""
    runner = _RUNNERS.get(name)
    if runner is None:
        raise ValueError(f"Unknown algorithm: {name}. Valid: {list(_RUNNERS.keys())}")
    return runner


def detect_outliers(df: pl.DataFrame, contamination: float = 0.05) -> pl.DataFrame:
    """Legacy wrapper -- runs IForest on a single DataFrame. For testing only."""
    cfg = OutredConfig(contamination=contamination)
    return run_iforest(df, cfg)