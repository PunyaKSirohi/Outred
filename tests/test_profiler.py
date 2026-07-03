# tests/test_profiler.py

import polars as pl
import pytest

from outred.profiler import profile_dataframe


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "id": list(range(1, 101)),
        "amount": [float(x) for x in range(100)],
        "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["RARE"] * 5,
    })


class TestProfiler:
    def test_basic_profile(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert profile.total_rows == 100
        assert profile.total_columns == 3
        assert len(profile.numeric_columns) >= 1
        assert len(profile.categorical_columns) >= 1

    def test_null_stats(self):
        df = pl.DataFrame({
            "a": [1.0, None, 3.0, None, 5.0],
            "b": ["x", "y", None, "z", None],
        })
        profile = profile_dataframe(df)
        assert profile.null_total > 0
        assert profile.null_pct > 0

    def test_column_profiles(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert "amount" in profile.column_profiles
        cp = profile.column_profiles["amount"]
        assert cp.mean is not None
        assert cp.median is not None
        assert cp.std is not None

    def test_categorical_top_values(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert "category" in profile.column_profiles
        cp = profile.column_profiles["category"]
        assert cp.top_values is not None
        assert len(cp.top_values) > 0
        assert cp.top_values[0]["value"] == "A"  # most frequent

    def test_data_quality_score(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert 0 <= profile.data_quality_score <= 100

    def test_constant_detection(self):
        df = pl.DataFrame({
            "constant": [42.0] * 10,
            "varies": [float(x) for x in range(10)],
        })
        profile = profile_dataframe(df)
        assert "constant" in profile.constant_columns
        assert "varies" not in profile.constant_columns

    def test_to_dict(self, sample_df):
        profile = profile_dataframe(sample_df)
        d = profile.to_dict()
        assert "total_rows" in d
        assert "columns" in d
        assert isinstance(d["columns"], dict)

    def test_empty_df(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        profile = profile_dataframe(df)
        assert profile.total_rows == 0
        assert profile.data_quality_score <= 80  # penalised for few rows

import os
import tempfile
import polars as pl
import pytest
from outred.profiler import profile_dataframe
from outred.ingestion.chunker import count_csv_data_rows


class TestTrueRowCount:
    """Bug: total_rows used len(sample) instead of true file size,
    making the >1M rows -> HBOS routing rule permanently unreachable."""

    def test_true_row_count_passed_through(self):
        # Sample is 10 rows, but true file has 1,200,000  - profiler must
        # reflect the true count, not the sample size.
        df = pl.DataFrame({"a": range(10), "b": range(10)})
        profile = profile_dataframe(df, true_row_count=1_200_000)
        assert profile.total_rows == 1_200_000
        assert profile.sample_rows == 10
        assert profile.stats_are_sampled is True

    def test_falls_back_to_sample_when_not_provided(self):
        df = pl.DataFrame({"a": range(50), "b": range(50)})
        profile = profile_dataframe(df)
        assert profile.total_rows == 50
        assert profile.stats_are_sampled is False

    def test_count_csv_data_rows_matches_polars(self, tmp_path):
        # Confirm the cheap byte-count row counter agrees with polars on a
        # real file  - both with and without a trailing newline.
        csv_with_newline = tmp_path / "with_nl.csv"
        csv_with_newline.write_text("a,b\n1,2\n3,4\n5,6\n")
        assert count_csv_data_rows(str(csv_with_newline)) == 3

        csv_without_newline = tmp_path / "without_nl.csv"
        csv_without_newline.write_text("a,b\n1,2\n3,4\n5,6")
        assert count_csv_data_rows(str(csv_without_newline)) == 3

    @pytest.mark.skipif(
        not os.path.exists("t/nyc_taxi_1.2m.csv"),
        reason="requires t/nyc_taxi_1.2m.csv"
    )
    def test_real_file_row_count(self):
        count = count_csv_data_rows("t/nyc_taxi_1.2m.csv")
        assert count == 1_200_000


class TestSkewnessThreshold:
    """Bug: skewness threshold was 2.0, causing creditcard.csv (skew=2.11)
    to route to LOF which scored 20% recall vs IForest's 87%."""

    def test_threshold_is_conservative(self):
        # A dataset with skewness just over the old 2.0 threshold must NOT
        # route to lof under the new rules.
        from outred.router.dispatcher import _choose_algorithm
        from outred.profiler import DataProfile

        # Mimic creditcard.csv's profile: skew=2.11, small file, <50 dims
        profile = DataProfile(
            file_path="fake.csv",
            file_size_mb=143.0,
            total_rows=284_807,
            sample_rows=1000,
            total_columns=31,
            numeric_columns=list(range(30)),
            categorical_columns=[],
            datetime_columns=[],
            id_columns=[],
            constant_columns=[],
            null_total=0,
            null_pct=0.0,
            data_quality_score=95,
            avg_skewness=2.11,   # the value that caused the original misroute
            max_dimensionality=30,
            stats_are_sampled=True,
        )
        algo = _choose_algorithm(profile)
        assert algo == "iforest", (
            f"skewness=2.11 should route to iforest (not lof)  - "
            f"got {algo}. The old 2.0 threshold was empirically disproven "
            f"(LOF: 20% recall vs IForest: 87% on creditcard.csv)."
        )

    def test_high_skewness_still_routes_lof(self):
        from outred.router.dispatcher import _choose_algorithm
        from outred.profiler import DataProfile

        profile = DataProfile(
            file_path="fake.csv",
            file_size_mb=10.0,
            total_rows=50_000,
            sample_rows=1000,
            total_columns=5,
            numeric_columns=["a"],
            categorical_columns=[],
            datetime_columns=[],
            id_columns=[],
            constant_columns=[],
            null_total=0,
            null_pct=0.0,
            data_quality_score=90,
            avg_skewness=6.0,   # clearly over the new 5.0 bar
            max_dimensionality=5,
            stats_are_sampled=True,
        )
        algo = _choose_algorithm(profile)
        assert algo == "lof"


class TestIDColumnDetection:
    """Bug: profile.id_columns was computed but never consumed —
    sequential id columns leaked into every detection run as real features."""

    def test_sequential_int_detected_as_id(self):
        df = pl.DataFrame({
            "id": list(range(1, 101)),
            "amount": [float(x * x) for x in range(100)],  # non-sequential
        })
        profile = profile_dataframe(df)
        assert "id" in profile.id_columns
        assert "amount" not in profile.id_columns

    def test_id_excluded_from_usable_dimensions(self):
        np.random.seed(42)
        df = pl.DataFrame({
            "id":     list(range(1, 101)),
            "amount": np.random.normal(50, 10, 100).tolist(),
            "score":  np.random.normal(20, 5, 100).tolist(),
        })
        profile = profile_dataframe(df)
        assert profile.max_dimensionality == 2

    def test_no_id_columns_returns_same_config(self):
        from outred.router.dispatcher import _merge_id_columns_into_exclude
        from outred.config import OutredConfig

        np.random.seed(42)
        config = OutredConfig(exclude_columns=["user_col"])
        df = pl.DataFrame({
            "amount": np.random.normal(50, 10, 100).tolist(),
            "score":  np.random.normal(20, 5, 100).tolist(),
        })
        profile = profile_dataframe(df)
        new_config = _merge_id_columns_into_exclude(config, profile)
        assert new_config is config

    def test_id_merged_into_exclude_columns(self):
        from outred.router.dispatcher import _merge_id_columns_into_exclude
        from outred.config import OutredConfig

        config = OutredConfig(exclude_columns=[])
        df = pl.DataFrame({
            "id": list(range(1, 101)),
            "amount": [float(x) for x in range(100)],
        })
        profile = profile_dataframe(df)
        new_config = _merge_id_columns_into_exclude(config, profile)
        assert "id" in new_config.exclude_columns
        # original config must not be mutated
        assert "id" not in config.exclude_columns

    def test_no_id_columns_returns_same_config(self):
        from outred.router.dispatcher import _merge_id_columns_into_exclude
        from outred.config import OutredConfig

        config = OutredConfig(exclude_columns=["user_col"])
        df = pl.DataFrame({
            "amount": [float(x * x) for x in range(100)],
            "score":  [float(x * 1.5 + 3) for x in range(100)],
        })
        profile = profile_dataframe(df)
        new_config = _merge_id_columns_into_exclude(config, profile)
        assert new_config is config

    @pytest.mark.skipif(
        not os.path.exists("t/creditcard.csv"),
        reason="requires t/creditcard.csv"
    )
    def test_creditcard_has_no_id_columns(self):
        # creditcard.csv has no sequential id  - profiler should not
        # incorrectly exclude Time, Amount, or any V-column.
        sample = pl.read_csv("t/creditcard.csv", n_rows=1000,
                             infer_schema_length=None)
        profile = profile_dataframe(sample)
        assert profile.id_columns == []