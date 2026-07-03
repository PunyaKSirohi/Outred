# tests/test_engines.py

import polars as pl
import numpy as np
import pytest

from outred.config import OutredConfig
from outred.engines.tabular import (
    run_iforest, run_hbos, run_lof, run_cblof, run_ocsvm,
    run_ensemble, detect_outliers,
)


@pytest.fixture
def normal_df():
    """DataFrame with mostly normal data + a few clear outliers."""
    np.random.seed(42)
    n = 200
    amounts = np.concatenate([
        np.random.normal(50, 10, n - 5),  # normal data
        np.array([500, 600, 700, -100, -200]),  # outliers
    ])
    categories = np.random.choice(["A", "B", "C"], n)
    return pl.DataFrame({
        "amount": amounts,
        "category": categories.tolist(),
    })


@pytest.fixture
def config():
    return OutredConfig(contamination=0.05, scaling="robust", impute="median")


class TestIsolationForest:
    def test_runs_and_adds_columns(self, normal_df, config):
        result = run_iforest(normal_df, config)
        assert "anomaly_score" in result.columns
        assert "is_outlier" in result.columns
        assert len(result) == len(normal_df)

    def test_scores_normalised(self, normal_df, config):
        result = run_iforest(normal_df, config)
        scores = result["anomaly_score"].to_numpy()
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_finds_outliers(self, normal_df, config):
        result = run_iforest(normal_df, config)
        outlier_count = result["is_outlier"].sum()
        assert outlier_count > 0


class TestHBOS:
    def test_runs(self, normal_df, config):
        result = run_hbos(normal_df, config)
        assert "anomaly_score" in result.columns
        assert result["is_outlier"].sum() > 0


class TestLOF:
    def test_runs(self, normal_df, config):
        result = run_lof(normal_df, config)
        assert "anomaly_score" in result.columns
        assert result["is_outlier"].sum() > 0


class TestCBLOF:
    def test_runs(self, normal_df, config):
        result = run_cblof(normal_df, config)
        assert "anomaly_score" in result.columns


class TestOCSVM:
    def test_runs(self, normal_df, config):
        result = run_ocsvm(normal_df, config)
        assert "anomaly_score" in result.columns


class TestEnsemble:
    def test_runs(self, normal_df, config):
        result = run_ensemble(normal_df, config)
        assert "anomaly_score" in result.columns
        assert "is_outlier" in result.columns
        assert result["is_outlier"].sum() > 0

    def test_scores_normalised(self, normal_df, config):
        result = run_ensemble(normal_df, config)
        scores = result["anomaly_score"].to_numpy()
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0001  # allow tiny float rounding


class TestLegacyAPI:
    def test_detect_outliers_compat(self, normal_df):
        result = detect_outliers(normal_df, contamination=0.05)
        assert "anomaly_score" in result.columns
        assert "is_outlier" in result.columns


class TestEdgeCases:
    def test_no_numeric_columns(self, config):
        df = pl.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        result = run_iforest(df, config)
        assert result["anomaly_score"].sum() == 0.0
        assert result["is_outlier"].sum() == 0

    def test_single_row(self, config):
        df = pl.DataFrame({"val": [42.0]})
        result = run_iforest(df, config)
        assert len(result) == 1
        assert result["is_outlier"].sum() == 0

    def test_all_nulls(self, config):
        df = pl.DataFrame({"val": pl.Series([None, None, None], dtype=pl.Float64)})
        result = run_iforest(df, config)
        assert len(result) == 3

import os
import polars as pl
import numpy as np
import pytest
from outred.config import OutredConfig
from outred.engines.incremental import IncrementalOutlierDetector
from outred.engines.categorical import detect_categorical_outliers
from outred.ingestion.chunker import stream_csv


class TestIncrementalExcludeColumns:
    """Bug: IncrementalOutlierDetector ignored exclude_columns entirely —
    sequential id columns leaked into _prepare_features as real features."""

    def test_exclude_columns_respected(self):
        np.random.seed(42)
        df = pl.DataFrame({
            "id": list(range(500)),
            "amount": np.random.normal(50, 10, 500).tolist(),
        })
        det_with = IncrementalOutlierDetector(nu=0.05)
        det_without = IncrementalOutlierDetector(
            nu=0.05, exclude_columns=["id"]
        )
        det_with.partial_fit(df)
        det_without.partial_fit(df)

        result_with = det_with.predict(df)
        result_without = det_without.predict(df)

        flagged_with = set(result_with.filter(
            pl.col("is_outlier"))["id"].to_list())
        flagged_without = set(result_without.filter(
            pl.col("is_outlier"))["id"].to_list())

        # Results must differ  - id column was distorting detection
        assert flagged_with != flagged_without, (
            "Excluding the id column should change which rows are flagged"
        )

    def test_chunked_streaming_stable(self, tmp_path):
        np.random.seed(0)
        rows = 50_000
        df = pl.DataFrame({
            "id": list(range(rows)),
            "amount": np.concatenate([
                np.random.normal(50, 5, rows - 200),
                np.random.normal(500, 5, 200),  # clear outliers
            ]).tolist(),
        })
        csv_path = str(tmp_path / "test_stream.csv")
        df.write_csv(csv_path)

        det = IncrementalOutlierDetector(nu=0.05, exclude_columns=["id"])
        for chunk in stream_csv(csv_path, chunk_size=5000):
            det.partial_fit(chunk)

        rates = []
        for chunk in stream_csv(csv_path, chunk_size=5000):
            result = det.predict(chunk)
            rates.append(result["is_outlier"].mean())

        avg_rate = np.mean(rates)
        # SGDOneClassSVM's nu is an approximate bound, not exact  - allow wide range
        assert 0.001 <= avg_rate <= 0.20, (
            f"Expected flagged rate between 0.1% and 20%, got {avg_rate:.2%}"
        )

class TestCategoricalVectorizationCorrectness:
    """Regression: vectorized rewrite must produce identical output to the
    original Python-list implementation on all edge cases."""

    def test_rare_category_flagged(self):
        df = pl.DataFrame({
            "cat": ["A"] * 95 + ["B"] * 4 + ["RARE"],
        })
        result = detect_categorical_outliers(df, threshold=0.02)
        rare_rows = result.filter(pl.col("is_cat_outlier"))
        assert rare_rows.height == 1
        assert rare_rows["cat"][0] == "RARE"

    def test_common_categories_not_flagged(self):
        df = pl.DataFrame({
            "cat": ["A"] * 50 + ["B"] * 50,
        })
        result = detect_categorical_outliers(df, threshold=0.01)
        assert result["is_cat_outlier"].sum() == 0

    def test_score_range(self):
        df = pl.DataFrame({
            "cat": ["A"] * 90 + ["B"] * 9 + ["C"],
        })
        result = detect_categorical_outliers(df, threshold=0.02)
        scores = result["cat_anomaly_score"].to_numpy()
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_high_cardinality_skipped(self):
        # If every value is unique (cardinality > 10% of rows), it's an ID
        # column  - should be skipped, no outliers flagged.
        df = pl.DataFrame({
            "id_col": [str(i) for i in range(200)],
        })
        result = detect_categorical_outliers(df, threshold=0.01)
        assert result["is_cat_outlier"].sum() == 0

    @pytest.mark.skipif(
        not os.path.exists("t/creditcard.csv"),
        reason="requires t/creditcard.csv"
    )
    def test_no_categorical_outliers_on_numeric_dataset(self):
        df = pl.read_csv("t/creditcard.csv", n_rows=10_000,
                         infer_schema_length=None)
        result = detect_categorical_outliers(df)
        assert result["is_cat_outlier"].sum() == 0