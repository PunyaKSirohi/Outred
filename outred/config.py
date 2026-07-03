# outred/config.py
# Central configuration for the Outred pipeline.

from dataclasses import dataclass, field
from typing import Optional, List

# All algorithm identifiers that the engine supports.
VALID_ALGORITHMS = ("auto", "iforest", "hbos", "lof", "cblof", "ocsvm", "ensemble")
VALID_SCALING = ("robust", "standard", "minmax", "none")
VALID_IMPUTE = ("median", "mean", "zero", "drop")
VALID_CAT_FREQ_MODES = ("two-pass", "single-pass")


@dataclass
class OutredConfig:
    """
    Immutable-ish configuration bag that flows through every layer of the
    pipeline.  Created once at startup (by the CLI parser or the API request
    handler) and threaded into profiler → router → engine → reporter.
    """

    # --- Algorithm ----------------------------------------------------------
    algorithm: str = "auto"
    """
    Which PyOD model to run.
    'auto'      - let the smart router decide based on data profile.
    'ensemble'  - run IForest + HBOS + LOF and average their scores.
    'iforest'   - Isolation Forest.
    'hbos'      - Histogram-Based Outlier Score.
    'lof'       - Local Outlier Factor.
    'cblof'     - Clustering-Based Local Outlier Factor.
    'ocsvm'     - One-Class SVM.
    """




    contamination: float = 0.05
    """Expected proportion of outliers (0.001 – 0.20)."""

    # --- Preprocessing ------------------------------------------------------
    scaling: str = "robust"
    """Feature scaling method applied before detection."""

    impute: str = "median"
    """How to handle null / NaN values in numeric columns."""

    # --- I/O ----------------------------------------------------------------
    chunk_size: int = 100_000
    """Number of rows per chunk during streaming ingestion."""

    output_path: str = "results/output.parquet"
    """Where the final Parquet file is written (CLI mode)."""

    # --- Explainability -----------------------------------------------------
    explain: bool = False
    """If True, compute SHAP feature contributions for flagged outliers."""

    max_explain_rows: int = 100
    """Cap on how many outlier rows get SHAP explanations per chunk."""

    # --- Web / API ----------------------------------------------------------
    max_file_mb: float = 50.0
    """Maximum upload size in MB for the web API."""

    # --- Categorical detection (Advanced) -----------------------------------
    cat_rare_threshold: float = 0.01
    """Frequency below which a category value is flagged as 'rare' (outlier).
    E.g. 0.01 = a value appearing in <1% of rows is rare.  Range: 0.001–0.50."""

    cat_max_cardinality_ratio: float = 0.10
    """Skip categorical columns whose unique_values / total_rows exceeds this
    ratio  - they are high-cardinality (names, addresses, free-text) and not
    meaningful categories.  Range: 0.01–1.0."""

    cat_min_cardinality: int = 10

    cat_freq_mode: str = "two-pass"
    """How to compute categorical value frequencies for rare-value detection.
    'two-pass'    -- (default) stream the full file to build exact global
                     frequency maps before detection.  More accurate on large
                     or sorted files, but adds an extra I/O pass.
    'single-pass' -- use the profiling sample's frequencies as a proxy for
                     the whole file.  Faster, but less accurate if the sample
                     is not representative.  Recommended for smaller datasets
                     (< 100K rows) where the sample already covers most values."""
    """Floor on the unique-value count used in the cardinality check.  Prevents
    tiny datasets from accidentally triggering the high-cardinality skip rule.
    Range: 1–10000."""

    # --- Profiling (Advanced) -----------------------------------------------
    profiler_id_cardinality_ratio: float = 0.50
    """In the profiling sample, flag a string column as 'ID-like' and exclude
    it from detection if unique_values / sample_rows exceeds this ratio.
    Range: 0.10–1.0."""

    numeric_cast_threshold: float = 0.80
    """Cast a String column to Float64 if this fraction of non-null values
    successfully parse as numbers.  Lower = more aggressive casting (risks
    converting coded IDs like zip codes).  Range: 0.50–1.0."""

    profiler_sample_rows: int = 1000

    global_sample_rows: int = 50_000
    """Number of rows to use for fitting the global numeric model (Option B).
    Rows are sampled uniformly from across the file via reservoir sampling.
    Larger = more representative but uses more RAM during model fitting.
    Range: 1000-500000."""
    """Number of rows to read for profiling.  Larger = more accurate
    cardinality / skewness estimates but slower startup.  Range: 100–100000."""

    # --- Routing thresholds (Advanced) --------------------------------------
    route_incremental_size_mb: float = 500.0
    """File size in MB above which the incremental (SGDOneClassSVM) engine is
    auto-selected.  Range: 50–10000."""

    route_hbos_row_threshold: int = 1_000_000
    """Row count above which HBOS is auto-selected (fastest PyOD model).
    Range: 10000–100000000."""

    route_high_dim_threshold: int = 50
    """Number of usable numeric columns above which IForest is preferred
    (handles high-dimensionality well).  Range: 5–1000."""

    route_skewness_threshold: float = 5.0
    """Average |skewness| above which LOF is auto-selected (handles varying
    densities).  Range: 1.0–20.0."""

    # --- Internals (not user-facing) ----------------------------------------
    exclude_columns: List[str] = field(default_factory=list)
    """Column names to exclude from analysis (e.g. IDs supplied by the user)."""

    def validate(self) -> None:
        """Raise ValueError if any setting is out of range."""
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"--algorithm must be one of {VALID_ALGORITHMS}. Got: {self.algorithm}"
            )
        if not (0.001 <= self.contamination <= 0.20):
            raise ValueError(
                f"--contamination must be between 0.001 and 0.20. Got: {self.contamination}"
            )
        if self.scaling not in VALID_SCALING:
            raise ValueError(
                f"--scaling must be one of {VALID_SCALING}. Got: {self.scaling}"
            )
        if self.impute not in VALID_IMPUTE:
            raise ValueError(
                f"--impute must be one of {VALID_IMPUTE}. Got: {self.impute}"
            )
        if self.chunk_size < 100:
            raise ValueError(
                f"--chunk-size must be at least 100. Got: {self.chunk_size}"
            )
        # --- Advanced setting validation ------------------------------------
        if not (0.001 <= self.cat_rare_threshold <= 0.50):
            raise ValueError(
                f"--cat-threshold must be between 0.001 and 0.50. Got: {self.cat_rare_threshold}"
            )
        if not (0.01 <= self.cat_max_cardinality_ratio <= 1.0):
            raise ValueError(
                f"--cat-max-cardinality must be between 0.01 and 1.0. Got: {self.cat_max_cardinality_ratio}"
            )
        if self.cat_min_cardinality < 1:
            raise ValueError(
                f"--cat-min-cardinality must be at least 1. Got: {self.cat_min_cardinality}"
            )
        if self.cat_freq_mode not in VALID_CAT_FREQ_MODES:
            raise ValueError(
                f"--cat-freq-mode must be one of {VALID_CAT_FREQ_MODES}. Got: {self.cat_freq_mode}"
            )
        if not (0.10 <= self.profiler_id_cardinality_ratio <= 1.0):
            raise ValueError(
                f"--id-cardinality must be between 0.10 and 1.0. Got: {self.profiler_id_cardinality_ratio}"
            )
        if not (0.50 <= self.numeric_cast_threshold <= 1.0):
            raise ValueError(
                f"--numeric-cast must be between 0.50 and 1.0. Got: {self.numeric_cast_threshold}"
            )
        if not (100 <= self.profiler_sample_rows <= 100_000):
            raise ValueError(
                f"--sample-rows must be between 100 and 100000. Got: {self.profiler_sample_rows}"
            )
        if not (50 <= self.route_incremental_size_mb <= 10_000):
            raise ValueError(
                f"--route-incremental-mb must be between 50 and 10000. Got: {self.route_incremental_size_mb}"
            )
        if self.route_hbos_row_threshold < 10_000:
            raise ValueError(
                f"--route-hbos-rows must be at least 10000. Got: {self.route_hbos_row_threshold}"
            )
        if not (5 <= self.route_high_dim_threshold <= 1000):
            raise ValueError(
                f"--route-high-dims must be between 5 and 1000. Got: {self.route_high_dim_threshold}"
            )
        if not (1.0 <= self.route_skewness_threshold <= 20.0):
            raise ValueError(
                f"--route-skewness must be between 1.0 and 20.0. Got: {self.route_skewness_threshold}"
            )
