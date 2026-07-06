# outred/router/dispatcher.py
# Smart engine router -- profiles the data, selects the optimal algorithm,
# and orchestrates the detection pipeline.
#
# Key architectural changes (Option B + two-pass categorical):
#   1. Global numeric model: one reservoir sample is drawn from the full file,
#      the PyOD model is fitted once on that sample, then every chunk is scored
#      against the single fitted model (no re-fitting per chunk).
#   2. Global categorical frequencies: when cat_freq_mode == 'two-pass', a
#      lightweight pre-pass streams the CSV to count every categorical value
#      globally.  Per-chunk value_counts() are replaced by these global maps,
#      fixing the sorted-data distortion bug.  When cat_freq_mode == 'single-pass'
#      the profiling sample's value_counts are used as a proxy (faster, but
#      less accurate on large/sorted files).

import os
import sys
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Iterator, List, Dict, Optional

import polars as pl

from outred.config import OutredConfig
from outred.profiler import profile_dataframe, DataProfile
from outred.engines.tabular import (
    fit_global_model, score_chunk,
    fit_global_ensemble, score_chunk_ensemble,
)
from outred.engines.categorical import detect_categorical_outliers
from outred.engines.incremental import IncrementalOutlierDetector
from outred.explainer import explain_outliers
from outred.preprocessing import prepare_matrix, select_categorical_columns, cast_numeric_strings


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """
    Yielded by dispatch_batch() for every chunk processed.

    Attributes:
        chunk:        The original rows with anomaly_score / is_outlier columns appended.
        explanations: List of SHAP (or z-score fallback) explanation dicts,
                      one per flagged outlier row.  Empty when config.explain=False.
    """
    chunk: pl.DataFrame
    explanations: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def _merge_id_columns_into_exclude(config: OutredConfig, profile: DataProfile) -> OutredConfig:
    """
    Merge profiler-detected ID columns into config.exclude_columns so that
    every detection path (PyOD, ensemble, SHAP) automatically ignores them.
    Returns a NEW config (dataclasses.replace) to preserve immutability.
    """
    if not profile.id_columns:
        return config
    merged = list(dict.fromkeys(
        list(config.exclude_columns or []) + list(profile.id_columns)
    ))
    return replace(config, exclude_columns=merged)


# ---------------------------------------------------------------------------
# Reservoir sampling -- draw a globally representative sample
# ---------------------------------------------------------------------------

def _reservoir_sample(
    file_path: str,
    k: int,
    chunk_size: int,
) -> pl.DataFrame:
    """
    Stream the CSV and return a reservoir sample of k rows drawn uniformly
    at random from the entire file, using Knuth's Algorithm R.

    This is O(N) in time and O(k) in memory -- suitable for files too large
    to fit in RAM.  The returned DataFrame has the same schema as the CSV.

    Memory note: instead of converting chunks to Python dicts (5-10x overhead),
    we track selected row indices and accumulate matching Polars DataFrame slices.
    All data stays in Polars' efficient columnar Arrow memory throughout.
    """
    from outred.ingestion.chunker import stream_csv

    # reservoir_rows: list of single-row Polars DataFrames (stays in Arrow memory)
    reservoir_rows: List[pl.DataFrame] = []
    n_seen = 0

    for chunk in stream_csv(file_path, chunk_size):
        chunk_len = len(chunk)
        for local_i in range(chunk_len):
            n_seen += 1
            if len(reservoir_rows) < k:
                # Reservoir not yet full — always accept
                reservoir_rows.append(chunk.slice(local_i, 1))
            else:
                # Randomly decide whether to replace an existing slot
                j = random.randint(0, n_seen - 1)
                if j < k:
                    reservoir_rows[j] = chunk.slice(local_i, 1)

    if not reservoir_rows:
        return pl.DataFrame()

    return pl.concat(reservoir_rows, rechunk=True)


# ---------------------------------------------------------------------------
# Global categorical frequency pre-pass
# ---------------------------------------------------------------------------

def _build_global_freq_maps(
    file_path: str,
    cat_cols: List[str],
    chunk_size: int,
    max_cardinality_ratio: float,
    min_cardinality: int,
) -> Dict[str, Dict[str, float]]:
    """
    Stream the full CSV (read only the categorical columns) and accumulate
    global value counts.  Returns a dict mapping:
        column_name -> {value: frequency}
    where frequency = count / total_rows.

    Only columns that pass the cardinality filter are included.  High-
    cardinality columns (likely free-text / IDs) are excluded here as well
    so the maps align with what detect_categorical_outliers will process.
    """
    from outred.ingestion.chunker import stream_csv, count_csv_data_rows

    counters: Dict[str, Counter] = {c: Counter() for c in cat_cols}
    total_rows = 0

    for chunk in stream_csv(file_path, chunk_size):
        total_rows += len(chunk)
        for col in cat_cols:
            if col not in chunk.columns:
                continue
            series = chunk[col].drop_nulls()
            counters[col].update(series.to_list())

    if total_rows == 0:
        return {}

    freq_maps: Dict[str, Dict[str, float]] = {}
    for col, counter in counters.items():
        n_unique = len(counter)
        # Apply the same cardinality filter the engine uses
        if n_unique > max(min_cardinality, int(total_rows * max_cardinality_ratio)):
            continue  # skip -- high cardinality, engine will also skip
        freq_maps[col] = {val: cnt / total_rows for val, cnt in counter.items()}

    return freq_maps


def _build_sample_freq_maps(
    sample_df: pl.DataFrame,
    cat_cols: List[str],
    max_cardinality_ratio: float,
    min_cardinality: int,
) -> Dict[str, Dict[str, float]]:
    """
    Build frequency maps from the profiling sample only (single-pass mode).
    Faster but less accurate on large or sorted files.
    """
    sample_rows = len(sample_df)
    if sample_rows == 0:
        return {}

    freq_maps: Dict[str, Dict[str, float]] = {}
    for col in cat_cols:
        if col not in sample_df.columns:
            continue
        n_unique = sample_df[col].n_unique()
        if n_unique > max(min_cardinality, int(sample_rows * max_cardinality_ratio)):
            continue
        vc = sample_df[col].value_counts()
        col_map = {}
        for i in range(len(vc)):
            val = vc[col][i]
            cnt = int(vc["count"][i])
            if val is not None:
                col_map[str(val)] = cnt / sample_rows
        freq_maps[col] = col_map

    return freq_maps


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# ROUTING NOTE
# ------------
# profile.total_rows and profile.file_size_mb are EXACT (measured from the
# whole file).  profile.avg_skewness is sample-estimated (noisy).  The
# skewness rule uses a conservative threshold (5.0) to avoid flipping
# algorithms on borderline sample-noise values -- see profiler.py.

def _choose_algorithm(profile: DataProfile, config: OutredConfig) -> str:
    if profile.file_size_mb > config.route_incremental_size_mb:
        return "incremental"
    if profile.total_rows > config.route_hbos_row_threshold:
        return "hbos"
    if profile.max_dimensionality > config.route_high_dim_threshold:
        return "iforest"
    if profile.avg_skewness > config.route_skewness_threshold:
        return "lof"
    return "iforest"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dispatch_batch(
    file_path: str,
    config: OutredConfig,
) -> Iterator[ChunkResult]:
    """
    Main entry point.  Profiles the file, auto-selects an engine (or uses
    the user's explicit choice), draws a global sample, fits the model once,
    builds global categorical frequency maps, then streams and scores all
    chunks.
    """
    from outred.ingestion.chunker import stream_csv, validate_file, count_csv_data_rows

    validate_file(file_path)

    # -----------------------------------------------------------
    # 1. Sample + profile
    # -----------------------------------------------------------
    try:
        sample_n = config.profiler_sample_rows
        try:
            sample = pl.read_csv(file_path, n_rows=sample_n, infer_schema_length=0,
                                 encoding='utf8-lossy')
        except Exception:
            sample = pl.read_csv(file_path, n_rows=sample_n, infer_schema_length=0,
                                 encoding='latin1')
    except Exception:
        print("Error: CSV file has no data rows.")
        sys.exit(1)

    true_row_count = count_csv_data_rows(file_path)

    profile = profile_dataframe(
        sample, file_path,
        true_row_count=true_row_count,
        id_cardinality_ratio=config.profiler_id_cardinality_ratio,
    )

    # Merge profiler-detected ID columns into exclude list
    if profile.id_columns:
        print(f"  [Profiler] Excluding detected ID column(s): {profile.id_columns}")
    config = _merge_id_columns_into_exclude(config, profile)

    # -----------------------------------------------------------
    # 2. Datetime check -- time series is V2
    # -----------------------------------------------------------
    if profile.datetime_columns:
        print(f"  [Time Series] Datetime columns detected: {profile.datetime_columns}")
        print("  Time series anomaly detection is planned for V2 -- passing data through unchanged.")
        for chunk in stream_csv(file_path, config.chunk_size):
            yield ChunkResult(
                chunk=chunk.with_columns([
                    pl.lit(0.0).alias("anomaly_score"),
                    pl.lit(False).alias("is_outlier"),
                    pl.lit(0.0).alias("cat_anomaly_score"),
                    pl.lit(False).alias("is_cat_outlier"),
                ]),
                explanations=[],
            )
        return

    # -----------------------------------------------------------
    # 3. Choose algorithm
    # -----------------------------------------------------------
    algo = config.algorithm
    if algo == "auto":
        algo = _choose_algorithm(profile, config)
        print(f"  [Smart Router] Auto-selected algorithm: {algo}")
        print(f"    Rows={profile.total_rows:,} (sampled {profile.sample_rows:,} for stats)  "
              f"Dims={profile.max_dimensionality}  "
              f"Skew={profile.avg_skewness:.2f} (sample-estimated)  "
              f"Size={profile.file_size_mb:.1f}MB  "
              f"Quality={profile.data_quality_score}/100")
    else:
        print(f"  [Router] Using user-selected algorithm: {algo}")

    if profile.max_dimensionality == 0:
        print()
        print("  WARNING: No usable numeric features found in this dataset.")
        print("           Numeric outlier detection will produce all-zero scores.")
        print("           Only categorical (rare-value) detection will be active.")
        if profile.categorical_columns:
            print(f"           Categorical columns found: {profile.categorical_columns[:10]}")
        print()

    # -----------------------------------------------------------
    # 4. Run pipeline
    # -----------------------------------------------------------
    if algo == "incremental":
        yield from _incremental_pipeline(file_path, config, profile)
    else:
        yield from _batch_pipeline(file_path, config, algo, profile, sample)


# ---------------------------------------------------------------------------
# Pipeline implementations
# ---------------------------------------------------------------------------

def _batch_pipeline(
    file_path: str,
    config: OutredConfig,
    algo: str,
    profile: DataProfile,
    profiler_sample: pl.DataFrame,
) -> Iterator[ChunkResult]:
    """
    Option B batch pipeline:
      Pass 0a (numeric): reservoir-sample the file, fit the model once.
      Pass 0b (categorical): if two-pass mode, stream to count global freqs.
      Pass 1: stream each chunk, score with pre-fitted model + global freqs.
    """
    from outred.ingestion.chunker import stream_csv

    # --- Pass 0a: Draw global reservoir sample and fit numeric model ----------
    print(f"  [Global Model] Sampling {config.global_sample_rows:,} rows for model fitting...")
    global_sample = _reservoir_sample(file_path, config.global_sample_rows, config.chunk_size)

    if len(global_sample) == 0:
        print("  Warning: Reservoir sample is empty. Falling back to profiler sample.")
        global_sample = profiler_sample

    if algo == "ensemble":
        print(f"  [Global Model] Fitting ensemble (IForest + HBOS + LOF) on {len(global_sample):,} rows...")
        fitted_models, threshold, g_lo, g_hi = fit_global_ensemble(global_sample, config)
        print(f"  [Global Model] Ensemble ready. Decision threshold={threshold:.4f}")
    else:
        print(f"  [Global Model] Fitting {algo} on {len(global_sample):,} rows...")
        fitted_model, threshold, g_lo, g_hi = fit_global_model(global_sample, algo, config)
        fitted_models = None  # sentinel -- using single model path
        print(f"  [Global Model] Model ready. Decision threshold={threshold:.4f}  "
              f"Score range=[{g_lo:.4f}, {g_hi:.4f}]")

    # --- Pass 0b: Build global categorical frequency maps --------------------
    # Identify categorical columns from the sample (after casting)
    from outred.preprocessing import cast_numeric_strings, select_categorical_columns
    cast_sample = cast_numeric_strings(global_sample, cast_threshold=config.numeric_cast_threshold)
    cat_cols = select_categorical_columns(cast_sample, exclude=config.exclude_columns)

    global_freq_maps: Optional[Dict[str, Dict[str, float]]] = None

    if not cat_cols:
        print("  [Categorical] No categorical columns found -- skipping frequency pre-pass.")
    elif config.cat_freq_mode == "two-pass":
        print(f"  [Categorical] Two-pass mode: building global frequency maps for "
              f"{len(cat_cols)} column(s)...")
        global_freq_maps = _build_global_freq_maps(
            file_path, cat_cols, config.chunk_size,
            config.cat_max_cardinality_ratio, config.cat_min_cardinality,
        )
        print(f"  [Categorical] Global maps built for {len(global_freq_maps)} column(s).")
    else:
        # single-pass: use the profiler sample's counts as a proxy
        print(f"  [Categorical] Single-pass mode: using profiler sample for frequency estimates.")
        global_freq_maps = _build_sample_freq_maps(
            profiler_sample, cat_cols,
            config.cat_max_cardinality_ratio, config.cat_min_cardinality,
        )
        print(f"  [Categorical] Sample-based maps built for {len(global_freq_maps)} column(s).")

    # --- Pass 1: Stream, score, yield ----------------------------------------
    for chunk in stream_csv(file_path, config.chunk_size):
        # Numeric detection (score-only, no re-fitting)
        if algo == "ensemble":
            result = score_chunk_ensemble(
                chunk, fitted_models, threshold, g_lo, g_hi, config
            )
        else:
            result = score_chunk(
                chunk, fitted_model, threshold, g_lo, g_hi, config
            )

        # Categorical detection (with global freq maps)
        result = detect_categorical_outliers(
            result,
            threshold=config.cat_rare_threshold,
            max_cardinality_ratio=config.cat_max_cardinality_ratio,
            min_cardinality=config.cat_min_cardinality,
            global_freq_maps=global_freq_maps,
        )

        # Explainability (optional)
        explanations: List[dict] = []
        if config.explain and "is_outlier" in result.columns:
            explanations = _attach_explanations(result, config, profile, algo)

        yield ChunkResult(chunk=result, explanations=explanations)


def _incremental_pipeline(
    file_path: str,
    config: OutredConfig,
    profile: DataProfile,
) -> Iterator[ChunkResult]:
    """
    SGDOneClassSVM pipeline for very large files.
    Two-pass: Pass 1 trains incrementally, Pass 2 scores and yields results.
    Categorical detection uses global freq maps (same logic as batch).
    """
    from outred.ingestion.chunker import stream_csv
    from outred.preprocessing import cast_numeric_strings, select_categorical_columns

    detector = IncrementalOutlierDetector(
        nu=config.contamination,
        exclude_columns=config.exclude_columns,
    )

    # Pass 1: Train incrementally
    print("  [Incremental] Pass 1/2: Training model incrementally...")
    for chunk in stream_csv(file_path, config.chunk_size):
        detector.partial_fit(chunk)

    # Build global categorical frequency maps (re-use the same logic)
    # Use a small sample to identify cat_cols, then run pre-pass or sample mode
    try:
        sample = pl.read_csv(file_path, n_rows=config.profiler_sample_rows,
                             infer_schema_length=0, encoding='utf8-lossy')
    except Exception:
        sample = pl.DataFrame()

    cast_sample = cast_numeric_strings(sample, cast_threshold=config.numeric_cast_threshold) if len(sample) > 0 else sample
    cat_cols = select_categorical_columns(cast_sample, exclude=config.exclude_columns) if len(cast_sample) > 0 else []

    global_freq_maps: Optional[Dict[str, Dict[str, float]]] = None
    if cat_cols:
        if config.cat_freq_mode == "two-pass":
            global_freq_maps = _build_global_freq_maps(
                file_path, cat_cols, config.chunk_size,
                config.cat_max_cardinality_ratio, config.cat_min_cardinality,
            )
        else:
            global_freq_maps = _build_sample_freq_maps(
                sample, cat_cols,
                config.cat_max_cardinality_ratio, config.cat_min_cardinality,
            )

    # Pass 2: Score and yield
    print("  [Incremental] Pass 2/2: Scoring chunks...")
    for chunk in stream_csv(file_path, config.chunk_size):
        result = detector.predict(chunk)
        result = detect_categorical_outliers(
            result,
            threshold=config.cat_rare_threshold,
            max_cardinality_ratio=config.cat_max_cardinality_ratio,
            min_cardinality=config.cat_min_cardinality,
            global_freq_maps=global_freq_maps,
        )

        explanations: List[dict] = []
        if config.explain and "is_outlier" in result.columns:
            print("  Note: --explain is not yet supported for the incremental engine.")

        yield ChunkResult(chunk=result, explanations=explanations)


# ---------------------------------------------------------------------------
# Explanation helpers
# ---------------------------------------------------------------------------

def _build_model_for_explanation(algo: str, config: OutredConfig, n_rows: int):
    """
    Build a fresh unfitted model matching the algorithm used for detection,
    for SHAP/fallback explanation purposes.
    Returns None for algorithms that can't be explained this way.
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
        return CBLOF(contamination=config.contamination, n_clusters=n_clusters, random_state=42)
    if algo == "ocsvm":
        from pyod.models.ocsvm import OCSVM
        return OCSVM(contamination=config.contamination)
    return None


def _attach_explanations(
    result: pl.DataFrame,
    config: OutredConfig,
    profile: DataProfile,
    algo: str,
) -> List[dict]:
    """
    Compute SHAP (or fallback z-score) explanations for the outlier rows in
    this chunk.  Re-fits a matching model on the chunk for explanation only
    (the main detection already used the global model).
    """
    try:
        outlier_mask = result["is_outlier"].to_numpy()
        if not outlier_mask.any():
            return []

        X, col_names = prepare_matrix(
            result,
            scaling=config.scaling,
            impute_strategy=config.impute,
            exclude=config.exclude_columns,
            numeric_cast_threshold=config.numeric_cast_threshold,
        )
        if X.shape[1] == 0:
            return []

        model = _build_model_for_explanation(algo, config, X.shape[0])
        if model is None:
            print(f"  Note: --explain not supported for algorithm='{algo}'; skipping.")
            return []

        model.fit(X)

        explanations = explain_outliers(
            X, model, outlier_mask, col_names,
            max_rows=config.max_explain_rows,
        )

        for ex in explanations[:5]:
            row_idx = ex["row_index"]
            method = ex.get("method", "shap")
            features = ex["top_features"]
            parts = []
            for f in features:
                actual = f["actual_value"]
                median = f["median_value"]
                name = f["feature"]
                if median != 0:
                    ratio = abs(actual / median)
                    parts.append(f"{name}={actual} ({ratio:.1f}x median)")
                else:
                    parts.append(f"{name}={actual}")
            print(f"    Row {row_idx} [{method}]: {', '.join(parts)}")

        return explanations

    except Exception as e:
        print(f"  Warning: Could not generate explanations: {e}")
        return []