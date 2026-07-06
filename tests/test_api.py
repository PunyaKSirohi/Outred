# tests/test_api.py

import os
import pytest
from io import BytesIO


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


@pytest.fixture
def sample_csv_bytes():
    """A small CSV in bytes for upload testing."""
    csv = "id,amount,category\n"
    for i in range(50):
        csv += f"{i},{50 + i * 0.5},{'ABC'[i % 3]}\n"
    # Add a few outliers
    csv += "50,999,A\n51,1000,B\n52,-500,C\n"
    return csv.encode("utf-8")


class TestHealthEndpoint:
    def test_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestProfileEndpoint:
    def test_profile_success(self, client, sample_csv_bytes):
        res = client.post(
            "/api/profile",
            files={"file": ("test.csv", BytesIO(sample_csv_bytes), "text/csv")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "total_rows" in data
        assert "columns" in data
        assert data["total_rows"] == 53

    def test_profile_non_csv(self, client):
        res = client.post(
            "/api/profile",
            files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert res.status_code == 400


class TestAnalyzeEndpoint:
    def test_analyze_success(self, client, sample_csv_bytes):
        res = client.post(
            "/api/analyze",
            files={"file": ("test.csv", BytesIO(sample_csv_bytes), "text/csv")},
            data={
                "algorithm": "iforest",
                "contamination": "0.05",
                "scaling": "robust",
                "impute": "median",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "summary" in data
        assert "outliers" in data
        assert "explanations" in data          # key must always be present
        assert isinstance(data["explanations"], list)
        assert data["summary"]["total_rows"] == 53

    def test_analyze_with_explanations(self, client, sample_csv_bytes):
        """Explanations list is populated when explain=true is sent."""
        res = client.post(
            "/api/analyze",
            files={"file": ("test.csv", BytesIO(sample_csv_bytes), "text/csv")},
            data={
                "algorithm": "iforest",
                "contamination": "0.05",
                "explain": "true",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "explanations" in data
        explanations = data["explanations"]
        assert isinstance(explanations, list)
        assert len(explanations) > 0, "Expected at least one explanation for flagged outliers"

        # Validate structure of each explanation
        for ex in explanations:
            assert "row_index" in ex, "Missing row_index"
            assert isinstance(ex["row_index"], int)
            assert "top_features" in ex, "Missing top_features"
            assert isinstance(ex["top_features"], list)
            assert len(ex["top_features"]) > 0

            for f in ex["top_features"]:
                assert "feature" in f
                assert "value" in f
                assert "actual_value" in f
                assert "median_value" in f

    def test_analyze_auto_algorithm(self, client, sample_csv_bytes):
        res = client.post(
            "/api/analyze",
            files={"file": ("test.csv", BytesIO(sample_csv_bytes), "text/csv")},
            data={"algorithm": "auto", "contamination": "0.05"},
        )
        assert res.status_code == 200

    def test_analyze_invalid_algorithm(self, client, sample_csv_bytes):
        res = client.post(
            "/api/analyze",
            files={"file": ("test.csv", BytesIO(sample_csv_bytes), "text/csv")},
            data={"algorithm": "banana", "contamination": "0.05"},
        )
        assert res.status_code == 400

    def test_analyze_non_csv(self, client):
        res = client.post(
            "/api/analyze",
            files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
            data={"algorithm": "auto"},
        )
        assert res.status_code == 400

    def test_ephemeral_cleanup(self, client, sample_csv_bytes):
        """Verify no temp files are left behind after analysis."""
        import tempfile
        tmp_dir = tempfile.gettempdir()
        before = set(os.listdir(tmp_dir))

        client.post(
            "/api/analyze",
            files={"file": ("test.csv", BytesIO(sample_csv_bytes), "text/csv")},
            data={"algorithm": "hbos", "contamination": "0.05"},
        )

        after = set(os.listdir(tmp_dir))
        new_files = after - before
        # Filter for our file patterns
        leaked = [f for f in new_files if f.endswith(".csv") or f.endswith(".parquet")]
        assert len(leaked) == 0, f"Temp files not cleaned up: {leaked}"
