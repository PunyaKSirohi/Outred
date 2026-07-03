# outred/reporter/aggregator.py
# Collects scored chunks and writes the final Parquet output.
# Uses streaming PyArrow ParquetWriter to keep memory flat.

import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


class ReportAggregator:
    """
    Receives scored chunks one at a time, writes each directly to a Parquet
    file via PyArrow's streaming writer, and keeps only running counters in
    memory.  This fixes the old bug where all chunks were accumulated in a
    list and concatenated at the end (which blew up RAM on large files).
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.start_time = time.time()

        # Running counters (no DataFrames stored)
        self.total_rows: int = 0
        self.total_numeric_outliers: int = 0
        self.total_cat_outliers: int = 0
        self.total_combined_outliers: int = 0
        self.chunks_processed: int = 0

        # PyArrow writer  - initialised lazily on the first chunk
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunk(self, df: pl.DataFrame) -> None:
        """Write one chunk directly to the Parquet file and update counters."""
        self.chunks_processed += 1
        self.total_rows += len(df)

        if "is_outlier" in df.columns:
            self.total_numeric_outliers += int(df["is_outlier"].sum())

        if "is_cat_outlier" in df.columns:
            self.total_cat_outliers += int(df["is_cat_outlier"].sum())

        if "is_outlier" in df.columns and "is_cat_outlier" in df.columns:
            combined = df["is_outlier"] | df["is_cat_outlier"]
            self.total_combined_outliers += int(combined.sum())

        # Convert to Arrow and stream-write
        arrow_table = df.to_arrow()

        if self._writer is None:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._schema = arrow_table.schema
            self._writer = pq.ParquetWriter(self.output_path, self._schema)

        self._writer.write_table(arrow_table)

    def finalize(self) -> Dict[str, Any]:
        """Close the Parquet writer and print the summary report."""
        if self._writer is not None:
            self._writer.close()

        elapsed = round(time.time() - self.start_time, 2)

        if self.total_rows == 0:
            print("  Warning: No data was processed.")
            return self._summary_dict(elapsed)

        combined_pct = round((self.total_combined_outliers / self.total_rows) * 100, 2)
        numeric_pct = round((self.total_numeric_outliers / self.total_rows) * 100, 2)
        cat_pct = round((self.total_cat_outliers / self.total_rows) * 100, 2)

        print(f"""
    +------------------------------------------+
    |          OUTRED -- SCAN COMPLETE          |
    +------------------------------------------+
    |  Rows scanned        : {self.total_rows:>14,}   |
    |  Numeric outliers    : {self.total_numeric_outliers:>9,} ({numeric_pct}%)  |
    |  Categorical outliers: {self.total_cat_outliers:>9,} ({cat_pct}%)  |
    |  Combined outliers   : {self.total_combined_outliers:>9,} ({combined_pct}%)  |
    |  Chunks processed    : {self.chunks_processed:>14,}   |
    |  Runtime             : {elapsed:>12}s   |
    |  Output saved        : {self.output_path:<18} |
    +------------------------------------------+
        """)

        return self._summary_dict(elapsed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summary_dict(self, elapsed: float) -> Dict[str, Any]:
        """Structured summary for the API to serialise as JSON."""
        return {
            "total_rows": self.total_rows,
            "numeric_outliers": self.total_numeric_outliers,
            "categorical_outliers": self.total_cat_outliers,
            "combined_outliers": self.total_combined_outliers,
            "numeric_pct": round((self.total_numeric_outliers / max(1, self.total_rows)) * 100, 2),
            "categorical_pct": round((self.total_cat_outliers / max(1, self.total_rows)) * 100, 2),
            "combined_pct": round((self.total_combined_outliers / max(1, self.total_rows)) * 100, 2),
            "chunks_processed": self.chunks_processed,
            "runtime_seconds": elapsed,
            "output_path": self.output_path,
        }