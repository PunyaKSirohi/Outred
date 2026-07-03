# server.py
# FastAPI web server for Outred  - ephemeral processing, rate limiting,
# and serving the web frontend.

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import polars as pl

from outred.config import OutredConfig
from outred.profiler import profile_dataframe
from outred.router.dispatcher import dispatch_batch
from outred.reporter.aggregator import ReportAggregator
from outred.ingestion.chunker import CSVError
from outred.ingestion.validator import validate_csv


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Outred",
    description="Outlier detection for tabular CSV data",
    version="1.0.0",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please wait before trying again."},
    )


# CORS  - allow the frontend served from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
_static_dir = Path(__file__).parent / "outred" / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Max upload size in bytes (1 GB)
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    algorithm: str = "auto"
    contamination: float = 0.05
    scaling: str = "robust"
    impute: str = "median"
    explain: bool = False


class AnalyzeResponse(BaseModel):
    summary: dict
    profile: dict
    outliers: list  # list of dicts for the top outlier rows
    explanations: list  # SHAP explanations (if requested)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the web frontend."""
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Outred API is running. Frontend not found."})


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/profile")
@limiter.limit("10/minute")
async def profile_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV and return a data profile (column stats, quality score).
    Ephemeral processing  - the file is deleted immediately after profiling.
    """
    tmp_path = None
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(400, "Only CSV files are supported.")

        # Stream to temp file, checking size incrementally to avoid
        # loading the entire file into RAM.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        _CHUNK = 1 * 1024 * 1024  # 1 MB chunks
        total = 0
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                raise HTTPException(413, "File too large. Maximum is 1 GB.")
            tmp.write(chunk)
        tmp.close()

        # Profile
        sample = pl.read_csv(tmp_path, n_rows=5000, infer_schema_length=None)
        profile = profile_dataframe(sample, file.filename or "<upload>")

        return JSONResponse(profile.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Profiling failed: {str(e)}")
    finally:
        # Ephemeral: guaranteed cleanup
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/analyze")
@limiter.limit("5/minute")
async def analyze_endpoint(
    request: Request,
    file: UploadFile = File(...),
    algorithm: str = Form("auto"),
    contamination: float = Form(0.05),
    scaling: str = Form("robust"),
    impute: str = Form("median"),
    explain: bool = Form(False),
    # --- Advanced Settings (all optional, sensible defaults) -----------------
    cat_threshold: float = Form(0.01),
    cat_max_cardinality: float = Form(0.10),
    cat_min_cardinality: int = Form(10),
    id_cardinality: float = Form(0.50),
    numeric_cast: float = Form(0.80),
    sample_rows: int = Form(1000),
    route_incremental_mb: float = Form(500.0),
    route_hbos_rows: int = Form(1_000_000),
    route_high_dims: int = Form(50),
    route_skewness: float = Form(5.0),
):
    """
    Upload a CSV, run outlier detection, return results as JSON.

    Ephemeral processing: the uploaded file and any temp output files are
    deleted immediately after processing. Zero data retention.
    """
    tmp_csv = None
    tmp_parquet = None
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(400, "Only CSV files are supported.")

        # Stream to temp file, checking size incrementally to avoid
        # loading the entire file into RAM.
        tmp_csv_f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_csv = tmp_csv_f.name
        _CHUNK = 1 * 1024 * 1024  # 1 MB chunks
        total = 0
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp_csv_f.close()
                raise HTTPException(413, "File too large. Maximum is 1 GB.")
            tmp_csv_f.write(chunk)
        tmp_csv_f.close()

        tmp_parquet_f = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        tmp_parquet = tmp_parquet_f.name
        tmp_parquet_f.close()

        # Build config  - all advanced settings wired through
        config = OutredConfig(
            algorithm=algorithm,
            contamination=contamination,
            scaling=scaling,
            impute=impute,
            explain=explain,
            output_path=tmp_parquet,
            cat_rare_threshold=cat_threshold,
            cat_max_cardinality_ratio=cat_max_cardinality,
            cat_min_cardinality=cat_min_cardinality,
            profiler_id_cardinality_ratio=id_cardinality,
            numeric_cast_threshold=numeric_cast,
            profiler_sample_rows=sample_rows,
            route_incremental_size_mb=route_incremental_mb,
            route_hbos_row_threshold=route_hbos_rows,
            route_high_dim_threshold=route_high_dims,
            route_skewness_threshold=route_skewness,
        )
        config.validate()

        # Profile
        sample = pl.read_csv(tmp_csv, n_rows=config.profiler_sample_rows,
                             infer_schema_length=None)
        profile = profile_dataframe(
            sample, file.filename or "<upload>",
            id_cardinality_ratio=config.profiler_id_cardinality_ratio,
        )

        # Run detection
        aggregator = ReportAggregator(output_path=tmp_parquet)
        all_outlier_rows = []
        all_explanations = []

        for result in dispatch_batch(file_path=tmp_csv, config=config):
            aggregator.add_chunk(result.chunk)

            # Collect outlier rows for the response (cap at 500 total)
            if "is_outlier" in result.chunk.columns and len(all_outlier_rows) < 500:
                outliers = result.chunk.filter(
                    pl.col("is_outlier") | pl.col("is_cat_outlier")
                )
                if len(outliers) > 0:
                    remaining = 500 - len(all_outlier_rows)
                    rows = outliers.head(remaining).to_dicts()
                    all_outlier_rows.extend(rows)

            # Collect explanations (cap at 200 total)
            if result.explanations and len(all_explanations) < 200:
                remaining = 200 - len(all_explanations)
                all_explanations.extend(result.explanations[:remaining])

        summary = aggregator.finalize()

        return JSONResponse({
            "summary": summary,
            "profile": profile.to_dict(),
            "outliers": all_outlier_rows,
            "explanations": all_explanations,
        })

    except HTTPException:
        raise
    except CSVError:
        # The CSV couldn't be parsed — run the validator on the still-live
        # temp file to collect every malformed row, then return them to the
        # UI so the user knows exactly what to fix. No files written to disk.
        validation_issues = []
        try:
            v_result = validate_csv(
                tmp_csv,
                write_reports=False,
                max_lines=100_000,  # cap scan time on huge uploads
            )
            validation_issues = [
                {
                    "line": iss.line_number,
                    "type": iss.issue_type,
                    "description": iss.description,
                    "raw_content": iss.raw_line[:300],
                }
                for iss in v_result.issues[:200]  # cap response size
            ]
        except Exception:
            pass
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Your CSV file contains structural issues that prevent parsing.",
                "issues": validation_issues,
                "total_issues": len(validation_issues),
            },
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        # Ephemeral: guaranteed cleanup of ALL temp files
        for path in (tmp_csv, tmp_parquet):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
