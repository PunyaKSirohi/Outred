# tests/test_preprocessing.py

import polars as pl
import numpy as np
import pytest

from outred.preprocessing import (
    select_numeric_columns,
    select_categorical_columns,
    impute,
    drop_constant_columns,
    build_scaler,
    prepare_matrix,
)


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "amount": [10.0, 20.0, None, 40.0, 50.0],
        "count": [100, 200, 300, 400, 500],
        "category": ["A", "B", "A", "C", "B"],
        "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
    })


class TestColumnSelection:
    def test_numeric_selection(self, sample_df):
        cols = select_numeric_columns(sample_df)
        assert "amount" in cols
        assert "count" in cols
        assert "category" not in cols

    def test_numeric_excludes_engine_cols(self):
        df = pl.DataFrame({
            "value": [1.0, 2.0],
            "anomaly_score": [0.5, 0.1],
            "is_outlier": [True, False],
        })
        cols = select_numeric_columns(df)
        assert "value" in cols
        assert "anomaly_score" not in cols

    def test_numeric_with_user_excludes(self, sample_df):
        cols = select_numeric_columns(sample_df, exclude=["id"])
        assert "id" not in cols
        assert "amount" in cols

    def test_categorical_selection(self, sample_df):
        cols = select_categorical_columns(sample_df)
        assert "category" in cols
        assert "amount" not in cols


class TestImputation:
    def test_median_imputation(self, sample_df):
        result = impute(sample_df, ["amount"], "median")
        # Median of [10, 20, 40, 50] = 30
        assert result["amount"].null_count() == 0

    def test_mean_imputation(self, sample_df):
        result = impute(sample_df, ["amount"], "mean")
        assert result["amount"].null_count() == 0

    def test_zero_imputation(self, sample_df):
        result = impute(sample_df, ["amount"], "zero")
        assert result["amount"].null_count() == 0
        assert result["amount"][2] == 0.0

    def test_drop_imputation(self, sample_df):
        result = impute(sample_df, ["amount"], "drop")
        assert len(result) == 4  # one row dropped


class TestDropConstantColumns:
    def test_drops_constant(self):
        cols = ["a", "b"]
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        new_cols, new_X = drop_constant_columns(cols, X)
        assert new_cols == ["b"]
        assert new_X.shape[1] == 1

    def test_keeps_non_constant(self):
        cols = ["a", "b"]
        X = np.array([[1.0, 5.0], [2.0, 6.0]])
        new_cols, new_X = drop_constant_columns(cols, X)
        assert len(new_cols) == 2


class TestBuildScaler:
    def test_robust(self):
        scaler = build_scaler("robust")
        assert scaler is not None

    def test_none(self):
        assert build_scaler("none") is None

    def test_invalid(self):
        with pytest.raises(ValueError):
            build_scaler("banana")


class TestPrepareMatrix:
    def test_basic(self, sample_df):
        X, cols = prepare_matrix(sample_df, scaling="robust", impute_strategy="median")
        assert X.shape[0] == 5
        assert X.shape[1] > 0  # at least some non-constant numeric cols
        assert isinstance(cols, list)

    def test_no_numeric(self):
        df = pl.DataFrame({"name": ["Alice", "Bob"]})
        X, cols = prepare_matrix(df)
        assert X.shape[1] == 0
        assert cols == []

    def test_single_row(self):
        df = pl.DataFrame({"val": [42.0]})
        X, cols = prepare_matrix(df, scaling="none")
        # Single row has zero variance, so the column is dropped as constant
        assert X.shape == (1, 0)
